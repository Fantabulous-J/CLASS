import copy
import glob
import json
import logging
import math
import os
import random
import re
import shutil
from itertools import chain
from statistics import mean
from typing import Dict, List, Tuple, Optional, Any, Union, Mapping

import numpy as np
import torch
import torch.distributed as dist
from packaging import version
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, get_constant_schedule, PreTrainedModel

from dataloader import GenericDataLoader, EncodeDataset, EncodeCollator, ReaderDataset, ReaderCollator
from evals.eval_xor_retrieve import read_jsonlines, evaluate_top_k_hit
from model import RRForConditionalGeneration
from retriever import FaissIPRetriever, write_ranking
from utils import AverageMeter, ProgressMeter, RandContext, compute_colbert_scores, group_corpus_by_langs

logger = logging.getLogger(__name__)


class ReaderTrainer(object):
    def __init__(self, model, train_dataset, data_collator, training_args, data_args, tokenizer):
        super(ReaderTrainer, self).__init__()
        self.training_args = training_args
        self.data_args = data_args
        self.args = training_args
        self.config = model.config
        self.epoch = 0

        if dist.is_initialized() and dist.get_world_size() > 1:
            assert self.training_args.negatives_x_device, self.training_args.negatives_x_device
        self._dist_loss_scale_factor = dist.get_world_size() if self.training_args.negatives_x_device else 1

        if self.training_args.negatives_x_device:
            assert dist.is_initialized() and dist.get_world_size() > 1, \
                ValueError('Distributed training has not been initialized for representation all gather.')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.data_collator = data_collator

        if isinstance(self.train_dataset, list):
            self.train_dataloader = []
            for idx, dataset in enumerate(train_dataset):
                self.train_dataloader.append(self.get_train_dataloader(dataset))
        else:
            self.train_dataloader = self.get_train_dataloader(self.train_dataset)

        if isinstance(self.train_dataloader, list):
            assert training_args.multi_task, \
                ValueError('can only have multiple datasets when using multi-task learning')
            self.num_training_steps = 0
            for dataloader in self.train_dataloader:
                self.num_training_steps += len(dataloader) // self.training_args.gradient_accumulation_steps
        else:
            self.num_training_steps = len(self.train_dataloader) // self.training_args.gradient_accumulation_steps
        if self.training_args.max_steps > 0:
            self.max_step = self.training_args.max_steps
            self.num_train_epochs = 1
        else:
            self.max_step = self.training_args.num_train_epochs * self.num_training_steps
            self.num_train_epochs = math.ceil(self.training_args.num_train_epochs)

        if training_args.gradient_checkpointing:
            model.encoder.gradient_checkpointing = training_args.gradient_checkpointing
        self.model = self.setup_model(model)
        self.optimizer = self.get_optimizer(self.model, self.training_args.weight_decay,
                                            self.training_args.learning_rate)
        if self.training_args.scheduler_type == "linear_warmup":
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.training_args.warmup_ratio * self.max_step,
                num_training_steps=self.max_step
            )
        else:
            assert self.training_args.scheduler_type == "constant", self.training_args.sheduler_type
            self.scheduler = get_constant_schedule(self.optimizer)

        os.makedirs(self.training_args.output_dir, exist_ok=True)

        self.use_amp = False
        self.amp_dtype = None
        self.scaler = None
        if self.training_args.fp16 or self.training_args.bf16:
            self.use_amp = True
            self.amp_dtype = torch.float16 if self.training_args.fp16 else torch.bfloat16
            self.scaler = torch.cuda.amp.GradScaler()

    def setup_model(self, model):
        model = model.to(self.training_args.device)
        if self.training_args.n_gpu > 1:
            model = nn.DataParallel(model)
        elif self.training_args.local_rank != -1:
            kwargs = {}
            if self.training_args.ddp_find_unused_parameters is not None:
                kwargs["find_unused_parameters"] = self.training_args.ddp_find_unused_parameters
            elif isinstance(model, PreTrainedModel):
                kwargs["find_unused_parameters"] = not model.is_gradient_checkpointing
            else:
                kwargs["find_unused_parameters"] = True

            if self.training_args.ddp_bucket_cap_mb is not None:
                kwargs["bucket_cap_mb"] = self.training_args.ddp_bucket_cap_mb
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.training_args.device] if self.training_args.n_gpu != 0 else None,
                output_device=self.training_args.device if self.training_args.n_gpu != 0 else None,
                broadcast_buffers=False,
                **kwargs,
            )
        return model

    def get_optimizer(self, model, weight_decay, lr):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if
                        p.requires_grad and not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        adam_kwargs = {
            "betas": (self.training_args.adam_beta1, self.training_args.adam_beta2),
            "eps": self.training_args.adam_epsilon,
        }
        return AdamW(optimizer_grouped_parameters, lr=lr, **adam_kwargs)

    def _save(self, model_to_save, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.training_args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        model_to_save = model_to_save.module if hasattr(model_to_save, 'module') else model_to_save
        if isinstance(model_to_save, PreTrainedModel):
            model_to_save.save_pretrained(output_dir)
        else:
            model_to_save.save(output_dir)

    def save_model(self):
        if self.is_world_process_zero():
            self._save(self.model)

    def is_world_process_zero(self) -> bool:
        return self.training_args.process_index == 0

    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            return data.to(self.training_args.device)
        return data

    def _prepare_inputs(
            self,
            inputs: Union[Tuple[Dict[str, Union[torch.Tensor, Any]], ...], Dict]
    ) -> List[Dict[str, Union[torch.Tensor, Any]]]:
        if isinstance(inputs, Mapping):
            return self._prepare_input(inputs)
        prepared = []
        for x in inputs:
            if isinstance(x, torch.Tensor):
                prepared.append(x.to(self.training_args.device))
            else:
                prepared.append(self._prepare_input(x))
        return prepared

    def get_train_dataloader(self, train_dataset) -> DataLoader:
        if train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        if self.training_args.world_size > 1:
            seed = self.training_args.data_seed if self.training_args.data_seed is not None else self.training_args.seed
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.training_args.world_size,
                rank=self.training_args.process_index,
                seed=seed,
            )
        else:
            train_sampler = RandomSampler(train_dataset)

        train_batch_size = self.training_args.train_batch_size

        return DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=False,
            num_workers=self.training_args.dataloader_num_workers,
        )

    def compute_loss(self, inputs, global_step=None):
        _, reader_inputs, _, batch_pids = inputs

        inputs_ids, attention_mask, independent_mask, query_mask, passage_mask, labels = \
            reader_inputs['input_ids'], reader_inputs['attention_mask'], reader_inputs['independent_mask'], \
            reader_inputs['query_mask'], reader_inputs['passage_mask'], reader_inputs['labels']
        self.training_args.gc_chunk_size = labels.size()[0]
        input_ids_chunks = torch.chunk(inputs_ids, chunks=self.training_args.gc_chunk_size, dim=0)
        attention_mask_chunks = torch.chunk(attention_mask, chunks=self.training_args.gc_chunk_size, dim=0)
        independent_mask_chunks = torch.chunk(independent_mask, chunks=self.training_args.gc_chunk_size, dim=0)
        labels_chunks = torch.chunk(labels, chunks=self.training_args.gc_chunk_size, dim=0)

        if isinstance(batch_pids, tuple):
            batch_bias = batch_pids[1]
        else:
            batch_bias = torch.zeros(inputs_ids.size(0), device=inputs_ids.device)
        bias_chunks = torch.chunk(batch_bias, chunks=self.training_args.gc_chunk_size, dim=0)

        if global_step < self.training_args.distillation_start_steps or self.training_args.only_reader:
            reader_loss = 0
            for idx, (input_ids, attention_mask, independent_mask, labels, bias_scores) in \
                    enumerate(zip(input_ids_chunks, attention_mask_chunks, independent_mask_chunks, labels_chunks, bias_chunks)):
                def chunk_forward():
                    with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                        loss = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            independent_mask=independent_mask,
                            labels=labels,
                            use_cache=False,
                        ).loss
                        loss /= self.training_args.gc_chunk_size
                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    return loss

                if idx != len(labels_chunks) - 1:
                    with self.model.no_sync():
                        loss = chunk_forward()
                else:
                    loss = chunk_forward()
                reader_loss = reader_loss + loss
            return reader_loss

        query_mask_chunks = torch.chunk(query_mask, chunks=self.training_args.gc_chunk_size, dim=0)
        passage_mask_chunks = torch.chunk(passage_mask, chunks=self.training_args.gc_chunk_size, dim=0)

        def compute_colbert_scores(query_vector, passage_vector, query_mask, passage_mask):
            # [num_query, num_passages, q_len, p_len]
            score_list = []
            chunk_query_vector = torch.chunk(query_vector, chunks=query_vector.size()[0], dim=0)
            for chunk in chunk_query_vector:
                scores = chunk.unsqueeze(1) @ passage_vector.unsqueeze(0).transpose(2, 3)
                scores = scores.masked_fill(~passage_mask[None, :, None].bool(), -1e9)
                scores = torch.max(scores, dim=-1).values
                score_list.append(scores)
            scores = torch.cat(score_list, dim=0)
            scores = scores.masked_fill(~query_mask[:, None].bool(), 0.0)
            scores = torch.sum(scores, dim=-1) / query_mask.sum(dim=1)[..., None]

            return scores

        reader_loss = 0
        for idx, (input_ids, attention_mask, independent_mask, query_mask, passage_mask, labels, bias_scores) in \
                enumerate(zip(input_ids_chunks, attention_mask_chunks, independent_mask_chunks, query_mask_chunks,
                              passage_mask_chunks, labels_chunks, bias_chunks)):
            def chunk_forward():
                with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        independent_mask=independent_mask,
                        query_mask=query_mask,
                        passage_mask=passage_mask,
                        labels=labels,
                        output_attentions=True,
                        return_dict=True,
                        use_cache=False,
                    )
                    loss = outputs.loss / self.training_args.gc_chunk_size
                    query_vector, passage_vector = outputs.query_vector, outputs.passage_vector
                    if len(query_vector.size()) == 3:
                        num_query, seq_len, _ = query_vector.size()
                        bsz = num_query // self.data_args.train_n_passages
                        query_vector = query_vector.view(bsz, self.data_args.train_n_passages, seq_len, -1).mean(1)
                    else:
                        bsz = query_vector.size()[0] // self.data_args.train_n_passages
                        query_vector = query_vector.view(bsz, self.data_args.train_n_passages, -1).mean(1)

                    cross_attention_scores = outputs.cross_attentions[-1][:, :, 0]
                    bsz, n_heads, _ = cross_attention_scores.size()
                    scores = cross_attention_scores.view(bsz, n_heads, self.data_args.train_n_passages, -1)
                    teacher_scores = scores.sum(dim=-1).mean(dim=1).detach()

                    if len(query_vector.size()) == 3:
                        retriever_scores = compute_colbert_scores(query_vector, passage_vector, query_mask,
                                                                  passage_mask)
                    else:
                        retriever_scores = torch.matmul(query_vector, passage_vector.transpose(0, 1))
                    retriever_logits = torch.log_softmax(retriever_scores, dim=-1)
                    retriever_loss = torch.nn.functional.kl_div(retriever_logits, teacher_scores,
                                                                reduction='batchmean') * self.training_args.retriever_weight
                    loss += retriever_loss / self.training_args.gc_chunk_size

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                return loss

            if idx != len(labels_chunks) - 1:
                with self.model.no_sync():
                    loss = chunk_forward()
            else:
                loss = chunk_forward()

            reader_loss = reader_loss + loss

        return reader_loss

    def dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def train_step(self, batch, global_step=None):
        if self.use_amp:
            if version.parse(torch.__version__) > version.parse("1.7.1"):
                with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                    kl_loss = self.compute_loss(batch, global_step)
            else:
                with torch.cuda.amp.autocast():
                    kl_loss = self.compute_loss(batch, global_step)
            self.scaler.scale(kl_loss).backward()
        else:
            kl_loss = self.compute_loss(batch, global_step)
            kl_loss.backward()

        return kl_loss

    def train(self):
        global_step = 0
        prev_global_step = 0
        best_eval = 0.0
        if self.training_args.eval_at_start:
            eval_result = self.refresh_passages(do_eval=True, global_step=global_step)
            if isinstance(eval_result, tuple):
                if self.data_args.task == "XOR-Retrieve":
                    if self.training_args.only_reader:
                        eval_result = eval_result[1]
                    else:
                        eval_result = eval_result[0]
                else:
                    eval_result = eval_result[1]
            best_eval = eval_result
        if self.training_args.refresh_passages:
            self.refresh_passages(do_eval=False)
        if self.training_args.max_steps > 0:
            dataset_epochs = [0] * len(self.train_dataloader)
        for epoch in range(self.num_train_epochs):
            if self.training_args.multi_task:
                for dataloader in self.train_dataloader:
                    if isinstance(dataloader.sampler, DistributedSampler):
                        dataloader.sampler.set_epoch(epoch)
            else:
                if isinstance(self.train_dataloader.sampler, DistributedSampler):
                    self.train_dataloader.sampler.set_epoch(epoch)
            self.epoch = copy.deepcopy(epoch)
            self.model.train()
            losses = AverageMeter('Loss', ':.4')
            progress = ProgressMeter(
                self.max_step if self.training_args.max_steps > 0 else self.num_training_steps,
                [losses],
                prefix="Epoch: [{}]".format(epoch))
            step = 0

            num_training_steps = [len(dataloader) for dataloader in self.train_dataloader]
            data_src_indices = []
            iterators = []
            for source, src_its in enumerate(num_training_steps):
                if self.training_args.max_steps > 0:
                    src_its = self.training_args.max_steps * self.training_args.gradient_accumulation_steps
                data_src_indices.extend([source] * src_its)
                train_dataloader = self.train_dataloader[source]
                iterators.append(iter(train_dataloader))

            epoch_rnd = random.Random(self.training_args.seed + epoch)
            epoch_rnd.shuffle(data_src_indices)

            for i, source_idx in enumerate(data_src_indices):
                try:
                    it = iterators[source_idx]
                    batch = next(it)
                except:
                    if self.training_args.max_steps > 0:
                        dataset_epochs[source_idx] += 1
                        dataloader = self.train_dataloader[source_idx]
                        if isinstance(dataloader.sampler, DistributedSampler):
                            dataloader.sampler.set_epoch(dataset_epochs[source_idx])
                    iterators[source_idx] = iter(self.train_dataloader[source_idx])
                    it = iterators[source_idx]
                    batch = next(it)
                batch = self._prepare_inputs(batch)
                kl_loss = self.train_step(batch, global_step)

                if self.is_world_process_zero() and not self.training_args.only_reader:
                    for name, param in self.model.named_parameters():
                        if param.grad is None:
                            print(name)

                if (step + 1) % self.training_args.gradient_accumulation_steps == 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_args.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_args.max_grad_norm)
                        self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    global_step += 1

                if self.training_args.negatives_x_device:
                    loss_list = [torch.zeros_like(kl_loss) for _ in range(dist.get_world_size())]
                    dist.all_gather(tensor_list=loss_list, tensor=kl_loss.contiguous())
                    loss = torch.mean(torch.stack(loss_list, dim=0), dim=0)
                    losses.update(loss.item())
                else:
                    losses.update(kl_loss.item())

                step += 1
                if self.training_args.max_steps > 0 and self.training_args.gradient_accumulation_steps > 1:
                    if global_step != 0 and global_step != prev_global_step \
                            and global_step % self.training_args.print_steps == 0 and \
                            self.training_args.process_index in [-1, 0]:
                        progress.display(global_step)
                        prev_global_step = global_step
                else:
                    if step % (self.training_args.print_steps * self.training_args.gradient_accumulation_steps) == 0 \
                            and self.training_args.process_index in [-1, 0]:
                        progress.display(step // self.training_args.gradient_accumulation_steps)

                if global_step != 0 and global_step % self.training_args.save_steps == 0 and \
                        global_step > self.training_args.distillation_start_steps:
                    eval_result = self.refresh_passages(do_eval=True, global_step=global_step)
                    if isinstance(eval_result, tuple):
                        if self.data_args.task == "XOR-Retrieve":
                            if self.training_args.only_reader:
                                eval_result = eval_result[1]
                            else:
                                eval_result = eval_result[0]
                        else:
                            eval_result = eval_result[1]
                    if self.is_world_process_zero() and eval_result > best_eval:
                        best_eval = eval_result
                        checkpoint_folder = f"checkpoint-best"
                        output_dir = os.path.join(self.training_args.output_dir, checkpoint_folder)
                        self._save(self.model, output_dir)
                        self.tokenizer.save_pretrained(output_dir)
                        shutil.copy2(os.path.join(self.training_args.output_dir, "dev_xor_retrieve_pids.jsonl"),
                                     output_dir)
                        shutil.copy2(
                            os.path.join(self.training_args.output_dir, "dev_reader_xor_eng_span_predictions.json"),
                            output_dir)
                    if self.data_args.load_partial:
                        idx = global_step // self.training_args.save_steps
                        num_examples = 800 * self.training_args.save_steps
                        data_dir = self.data_args.train_dir
                        train_path = os.path.join(data_dir, self.data_args.train_path)
                        with open(train_path) as f:
                            examples = [jsonline for jsonline in f.readlines()[idx * num_examples: (idx + 1) * num_examples]]
                        self.train_dataloader[0].dataset.examples = examples

                if self.training_args.refresh_passages and global_step != 0 and global_step != self.max_step and \
                        global_step % self.training_args.refresh_intervals == 0:
                    self.refresh_passages(do_eval=False, global_step=global_step)

                if global_step >= self.max_step:
                    break
            if global_step >= self.max_step:
                break

    def test(self):
        torch.distributed.barrier()
        global_step = self.training_args.max_steps
        if self.training_args.eval_on_test:
            logger.info('Evaluating on MKQA')
            self.model = RRForConditionalGeneration.from_pretrained(
                os.path.join(self.training_args.output_dir, f"checkpoint-best"),
                config=self.config,
            )
            self.model.to(self.training_args.device)
            if self.data_args.task == "XOR-Retrieve":
                self.data_args.eval_query_file = "mkqa_dev_retrieve_eng_span.jsonl"
            elif self.data_args.task == "XOR-Full":
                self.data_args.eval_query_file = "mkqa_dev_full.jsonl"
            else:
                raise NotImplementedError
            _ = self.refresh_passages(do_eval=True, eval_set="mkqa", global_step=global_step)

    def refresh_passages(self, do_eval=True, eval_set="dev", global_step=0):
        self.model.eval()
        torch.cuda.empty_cache()

        if ('nq-dpr' in self.data_args.train_dir or self.data_args.task not in self.data_args.train_dir) and do_eval:
            train_dir = 'data/XOR-Retrieve' if self.data_args.task == "XOR-Retrieve" else 'data/XOR-Full'
            corpus_file = 'psgs_w100.tsv' if self.data_args.task == "XOR-Retrieve" else 'all_w100.tsv'
            corpus = GenericDataLoader(train_dir, corpus_file=corpus_file,
                                       query_file=self.data_args.query_file).load_corpus()
        else:
            corpus = self.train_dataset[0].corpus
            train_dir = self.data_args.train_dir

        results = self.encode(do_eval=do_eval, corpus=corpus, global_step=global_step)
        logger.info(f"Process {self.training_args.process_index} Done encoding")

        def save_results(results, output_path, add_score=True):
            with open(output_path, 'w') as f:
                for qid in results:
                    sorted_indices_scores = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)[:100]
                    example = {
                        'qid': qid,
                        'pids': [(docid, score) if add_score else docid for docid, score in sorted_indices_scores]
                    }
                    f.write(json.dumps(example) + '\n')

        refresh = not (global_step == 0 and os.path.exists(os.path.join(self.training_args.output_dir, "train.jsonl")))

        if do_eval:
            output_path = os.path.join(self.training_args.output_dir, "{}.split{}.jsonl".format(
                eval_set, self.training_args.process_index))
            save_results(results, output_path)
        else:
            if refresh:
                if self.training_args.retrieve_from_each_lang:
                    assert len(results.keys()) == 13, len(results.keys())
                    for lang, lang_results in results.items():
                        output_path = os.path.join(self.training_args.output_dir,
                                                   f"train_{lang}.split{self.training_args.process_index}.jsonl")
                        save_results(lang_results, output_path)
                else:
                    output_path = os.path.join(self.training_args.output_dir,
                                               "train.split{}.jsonl".format(self.training_args.process_index))
                    save_results(results, output_path)

        torch.distributed.barrier()

        eval_result = 0.0
        if self.is_world_process_zero():
            def load_results(data_path):
                prediction_files = sorted(glob.glob(data_path))
                results = {}
                for path in prediction_files:
                    with open(path) as f:
                        for jsonline in f.readlines():
                            example = json.loads(jsonline)
                            qid = example['qid']
                            if qid not in results:
                                results[qid] = {}
                            for pid, score in example['pids']:
                                if not do_eval and 'mss' in self.data_args.query_file and len(qid.split("-")) >= 2 \
                                        and pid == qid.split("-")[1]:
                                    continue
                                results[qid][pid] = score
                return results

            if do_eval:
                results = load_results(os.path.join(self.training_args.output_dir, f"{eval_set}.split*.jsonl"))

                from retriever import parse_qa_jsonlines_file
                qas_file = os.path.join(train_dir, self.data_args.eval_query_file)
                qas = {}
                for question, qid, answers, lang in parse_qa_jsonlines_file(qas_file):
                    qas[qid] = (question, answers, lang)

                xor_output_prediction_format = []
                reader_evaluation_format = []
                output_path = os.path.join(self.training_args.output_dir, f"{eval_set}_xor_retrieve_results.json")
                with open(output_path, 'w') as f:
                    for qid in results:
                        sorted_indices_scores = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)[:100]
                        ctxs, pids = [], []
                        for docid, score in sorted_indices_scores:
                            ctxs.append(corpus[docid]["text"])
                            pids.append(docid)
                        question, answers, lang = qas[qid]
                        xor_output_prediction_format.append({"id": qid, "lang": lang, "ctxs": ctxs})
                        reader_evaluation_format.append({"qid": qid, "pids": pids})
                    json.dump(xor_output_prediction_format, f)
                output_path = os.path.join(self.training_args.output_dir, f"{eval_set}_xor_retrieve_pids.jsonl")
                with open(output_path, 'w') as f:
                    for example in reader_evaluation_format:
                        f.write(json.dumps(example) + '\n')

                if eval_set != "test":
                    predictions = json.load(open(os.path.join(self.training_args.output_dir,
                                                              "dev_xor_retrieve_results.json")))
                    input_data = read_jsonlines(
                        os.path.join(train_dir, self.data_args.eval_query_file))
                    qid2answers = {item["id"]: item["answers"] for item in input_data}
                    eval_results = {}
                    for topk in [2, 5, 100]:
                        logger.info("Evaluating R@{}kt".format(topk))
                        pred_per_lang_results = evaluate_top_k_hit(
                            predictions, qid2answers, topk * 1000)
                        avg_scores = []
                        for lang in pred_per_lang_results:
                            logger.info(
                                "performance on {0} ({1} examples)".format(lang, pred_per_lang_results[lang]["count"]))
                            per_lang_score = (pred_per_lang_results[lang]["hit"] / pred_per_lang_results[lang][
                                "count"]) * 100
                            logger.info(per_lang_score)

                            avg_scores.append(per_lang_score)

                        logger.info("Final macro averaged score: ")
                        logger.info(mean(avg_scores))
                        eval_results[topk] = mean(avg_scores)
                    eval_result = eval_results[2]
            else:
                if refresh:
                    results = load_results(os.path.join(self.training_args.output_dir, "train.split*.jsonl"))
                    output_path = os.path.join(self.training_args.output_dir, "train.jsonl")
                    save_results(results, output_path, add_score=False)

        torch.distributed.barrier()
        if not do_eval:
            logger.info(f"Process {self.training_args.process_index} Loading updated training passages")
            with open(os.path.join(self.training_args.output_dir, "train.jsonl"), 'r') as f:
                examples = [json.loads(jsonline) for jsonline in f]

            self.train_dataloader[0].dataset.examples = examples
        else:
            self.eval_reader(corpus=corpus, eval_set=eval_set)
            torch.distributed.barrier()

            if self.is_world_process_zero():
                data_path = os.path.join(self.training_args.output_dir,
                                         f"{eval_set}_reader_xor_eng_span_predictions.split*.json")
                prediction_files = sorted(glob.glob(data_path))
                results = {}
                for path in prediction_files:
                    with open(path) as f:
                        results.update(json.load(f))
                output_path = os.path.join(self.training_args.output_dir,
                                           f"{eval_set}_reader_xor_eng_span_predictions.json")
                with open(output_path, 'w') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                import subprocess

                if eval_set != "test":
                    if self.data_args.task == 'XOR-Retrieve':
                        file_path = 'evals/eval_xor_engspan.py'
                        args = f'--data_file {os.path.join(train_dir, self.data_args.eval_query_file)} ' \
                               f'--pred_file {output_path}'

                        result = subprocess.run(['python', file_path, *args.split()], capture_output=True, text=True)
                        output = result.stdout
                        pattern = re.compile(r'F1: (\d+\.\d+), EM:(\d+\.\d+)')
                        matches = pattern.findall(output)
                        f1_score = [float(match[0]) for match in matches][-1]
                        em_score = [float(match[1]) for match in matches][-1]
                        logger.info(output)
                        eval_result = (eval_result, f1_score, em_score)
                    else:
                        assert self.data_args.task == "XOR-Full", self.data_args.task
                        file_path = 'evals/eval_xor_full.py'
                        args = '--data_file data/XOR-Full/xor_dev_full_v1_1.jsonl ' \
                               f'--pred_file {output_path}'
                        result = subprocess.run(['python', file_path, *args.split()], capture_output=True, text=True)
                        output = result.stdout
                        pattern = re.compile(r'avg f1: (\d+\.\d+)\navg em: (\d+\.\d+)\navg bleu: (\d+\.\d+)')
                        matches = pattern.findall(output)
                        f1_score = [float(match[0]) for match in matches][-1]
                        em_score = [float(match[1]) for match in matches][-1]
                        bleu_score = [float(match[2]) for match in matches][-1]
                        logger.info(output)
                        eval_result = (eval_result, f1_score, em_score, bleu_score)
            torch.distributed.barrier()

        self.model.train()

        return eval_result

    def eval_reader(self, corpus, eval_set="dev"):
        if 'nq-dpr' in self.data_args.train_dir or self.data_args.task not in self.data_args.train_dir:
            train_dir = 'data/XOR-Retrieve' if self.data_args.task == "XOR-Retrieve" else 'data/XOR-Full'
        else:
            train_dir = self.data_args.train_dir
        queries = GenericDataLoader(train_dir, corpus_file=self.data_args.corpus_file,
                                    query_file=self.data_args.eval_query_file).load_queries()

        train_path = os.path.join(self.training_args.output_dir, f"{eval_set}_xor_retrieve_pids.jsonl")
        with open(train_path) as f:
            examples = [json.loads(jsonline) for jsonline in f.readlines()]
        shard_size = len(examples) // self.training_args.world_size
        start = self.training_args.process_index * shard_size
        end = (self.training_args.process_index + 1) * shard_size \
            if self.training_args.process_index + 1 != self.training_args.world_size else len(examples)
        examples = examples[start:end]

        eval_dataset = ReaderDataset(
            queries=queries,
            corpus=corpus,
            tokenizer=self.tokenizer,
            train_path=examples,
            data_args=self.data_args,
        )

        data_collator = ReaderCollator(
            self.tokenizer,
            max_query_length=self.data_args.max_query_length,
            max_passage_length=self.data_args.max_passage_length,
            max_query_passage_length=self.data_args.max_query_passage_length,
            max_answer_length=self.data_args.max_answer_length,
            separate_joint_encoding=self.training_args.separate_joint_encoding,
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=2,
            sampler=SequentialSampler(eval_dataset),
            collate_fn=data_collator,
            drop_last=False,
            num_workers=self.training_args.dataloader_num_workers,
        )
        model = self.model
        while hasattr(model, 'module'):
            model = model.module

        train_n_passages = 100 if self.data_args.train_n_passages == 1 else self.data_args.train_n_passages
        model.n_passages = train_n_passages
        predictions = {}
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                qids, reader_inputs, batch_answer, batch_pids = batch

                bsz, seq_len = reader_inputs["input_ids"].size()
                input_ids = reader_inputs["input_ids"].view(bsz // train_n_passages, train_n_passages, seq_len)
                model_inputs = {
                    "input_ids": input_ids.to(self.training_args.device),
                    "attention_mask": reader_inputs["attention_mask"].to(self.training_args.device),
                    "independent_mask": reader_inputs["independent_mask"].to(self.training_args.device),
                }

                outputs = model.generate(
                    **model_inputs,
                    max_length=self.data_args.max_answer_length,
                    num_beams=1,
                )
                for k, o in enumerate(outputs):
                    ans = self.tokenizer.decode(o, skip_special_tokens=True)
                    predictions[qids[k]] = ans

                torch.cuda.empty_cache()

        output_path = os.path.join(self.training_args.output_dir, "{}_reader_xor_eng_span_predictions.split{}.json".
                                   format(eval_set, self.training_args.process_index))
        with open(output_path, "w") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        model.n_passages = self.data_args.train_n_passages

    def encode_query(self, queries, start, end, do_eval=False):
        encode_dataset = EncodeDataset(queries, self.tokenizer, max_length=self.data_args.max_query_length,
                                       is_query=True, start=start, end=end,
                                       normalize_text=self.data_args.normalize_text,
                                       lower_case=self.data_args.lower_case, separate_joint_encoding=True,
                                       add_lang_token=self.data_args.add_lang_token)
        encode_loader = DataLoader(
            encode_dataset,
            batch_size=self.training_args.per_device_eval_batch_size * self.training_args.n_gpu,
            collate_fn=EncodeCollator(
                self.tokenizer,
                max_length=self.data_args.max_query_length,
                separate_joint_encoding=True,
            ),
            shuffle=False,
            drop_last=False,
            num_workers=self.training_args.dataloader_num_workers,
        )
        encoded = []
        mask = []
        lookup_indices = []

        for (batch_ids, batch) in tqdm(encode_loader):
            lookup_indices.extend(batch_ids)
            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                with torch.no_grad():
                    for k, v in batch.items():
                        batch[k] = v.to(self.training_args.device)
                    query_vector = self.model(query=batch, only_encoding=True).query_vector
                    encoded.append(query_vector)
                    mask.append(batch['attention_mask'])

        encoded = torch.cat(encoded)
        mask = torch.cat(mask)
        return encoded, mask, lookup_indices

    def encode(self, corpus, do_eval=True, global_step=0):
        if global_step == 0 and os.path.exists(os.path.join(self.training_args.output_dir, "train.jsonl")):
            return

        if do_eval:
            if 'nq-dpr' in self.data_args.train_dir or self.data_args.task not in self.data_args.train_dir:
                train_dir = 'data/XOR-Retrieve' if self.data_args.task == "XOR-Retrieve" else 'data/XOR-Full'
            else:
                train_dir = self.data_args.train_dir
            queries = GenericDataLoader(train_dir, corpus_file=self.data_args.corpus_file,
                                        query_file=self.data_args.eval_query_file).load_queries()
        else:
            queries = GenericDataLoader(self.data_args.train_dir, corpus_file=self.data_args.corpus_file,
                                        query_file=self.data_args.query_file).load_queries()
            if 'mss' in self.data_args.query_file:
                idx = global_step // self.training_args.refresh_intervals
                num_queries = 64 * self.training_args.refresh_intervals
                queries = dict(list(queries.items())[idx * num_queries: (idx + 1) * num_queries])

        if self.training_args.debug:
            queries = dict(list(queries.items())[:100])
        start, end = 0, len(queries)
        query_vector, q_mask, q_lookup_indices = self.encode_query(queries, start, end, do_eval=do_eval)

        results = {qid: {} for qid in q_lookup_indices}

        if self.training_args.debug:
            corpus = dict(list(corpus.items())[:1000])

        shard_size = len(corpus) // self.training_args.world_size
        start = self.training_args.process_index * shard_size
        end = (self.training_args.process_index + 1) * shard_size \
            if self.training_args.process_index + 1 != self.training_args.world_size else len(corpus)
        logger.info(
            f'Process {self.training_args.process_index} => Generate passage embeddings from {start} to {end}')

        encode_dataset = EncodeDataset(corpus, self.tokenizer, max_length=self.data_args.max_passage_length,
                                       is_query=False, start=start, end=end,
                                       normalize_text=self.data_args.normalize_text,
                                       lower_case=self.data_args.lower_case, separate_joint_encoding=True,
                                       add_lang_token=self.data_args.add_lang_token)
        encode_loader = DataLoader(
            encode_dataset,
            batch_size=self.training_args.per_device_eval_batch_size * self.training_args.n_gpu,
            collate_fn=EncodeCollator(
                self.tokenizer,
                max_length=self.data_args.max_passage_length,
                separate_joint_encoding=True,
                padding_to_max_length=True,
            ),
            shuffle=False,
            drop_last=False,
            num_workers=self.training_args.dataloader_num_workers,
        )

        lookup_indices, batch_scores = [], []
        for bidx, (batch_ids, batch) in enumerate(tqdm(encode_loader)):
            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                lookup_indices.extend(batch_ids)
                with torch.no_grad():
                    for k, v in batch.items():
                        batch[k] = v.to(self.training_args.device)
                    # [bz, p_len, dim]
                    passage_vector = self.model(passage=batch, only_encoding=True).passage_vector
                    passage_mask = batch['attention_mask']

                    if len(query_vector.size()) == 3:
                        scores = compute_colbert_scores(query_vector, passage_vector, q_mask, passage_mask)
                    else:
                        scores = torch.matmul(query_vector, passage_vector.transpose(0, 1))
                    batch_scores.append(scores)

            if len(batch_scores) % 100 == 0 or bidx == len(encode_loader) - 1:
                batch_scores = torch.cat(batch_scores, dim=1)
                if batch_scores.size()[0] > 64000:
                    batch_scores_list = torch.chunk(batch_scores, chunks=batch_scores.size()[0] // 32000, dim=0)
                    sorted_scores_list, sorted_indices_list = [], []
                    for batch_scores in batch_scores_list:
                        sorted_scores, sorted_indices = torch.topk(batch_scores, k=100, dim=-1)
                        sorted_scores_list.append(sorted_scores)
                        sorted_indices_list.append(sorted_indices)
                    sorted_scores = torch.cat(sorted_scores_list, dim=0)
                    sorted_indices = torch.cat(sorted_indices_list, dim=0)
                else:
                    sorted_scores, sorted_indices = torch.topk(batch_scores, k=min(100, batch_scores.size(-1)), dim=-1)
                sorted_scores = sorted_scores.cpu().numpy().tolist()
                sorted_indices = sorted_indices.cpu().numpy().tolist()
                for i, (scores, indices) in enumerate(zip(sorted_scores, sorted_indices)):
                    qid = q_lookup_indices[i]
                    for score, idx in zip(scores, indices):
                        docid = lookup_indices[idx]
                        results[qid][docid] = score
                    if len(results[qid]) > 100:
                        sorted_indices_scores = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)[:100]
                        results[qid] = {docid: score for docid, score in sorted_indices_scores}
                lookup_indices, batch_scores = [], []

        return results


class ReaderDistillTrainer(ReaderTrainer):
    def __init__(self, model, teacher_model, train_dataset, data_collator, training_args, data_args, tokenizer):
        super(ReaderDistillTrainer, self).__init__(model, train_dataset, data_collator, training_args, data_args,
                                                   tokenizer)
        self.teacher_model = teacher_model
        self.teacher_model.eval()

    def _forward(self, model, inputs, only_encoding=False, return_rnds=False):
        _, reader_inputs, _, _ = inputs

        inputs_ids, attention_mask, independent_mask, query_mask, passage_mask, labels = \
            reader_inputs['input_ids'], reader_inputs['attention_mask'], reader_inputs['independent_mask'], \
            reader_inputs['query_mask'], reader_inputs['passage_mask'], reader_inputs['labels']
        self.training_args.gc_chunk_size = labels.size()[0]
        input_ids_chunks = torch.chunk(inputs_ids, chunks=self.training_args.gc_chunk_size, dim=0)
        attention_mask_chunks = torch.chunk(attention_mask, chunks=self.training_args.gc_chunk_size, dim=0)
        independent_mask_chunks = torch.chunk(independent_mask, chunks=self.training_args.gc_chunk_size, dim=0)
        labels_chunks = torch.chunk(labels, chunks=self.training_args.gc_chunk_size, dim=0)

        query_mask_chunks = torch.chunk(query_mask, chunks=self.training_args.gc_chunk_size, dim=0)
        passage_mask_chunks = torch.chunk(passage_mask, chunks=self.training_args.gc_chunk_size, dim=0)

        query_vector_list, passage_vector_list, decoder_scores_list = [], [], []
        rnds = []
        for idx, (input_ids, attention_mask, independent_mask, query_mask, passage_mask, labels) in \
                enumerate(zip(input_ids_chunks, attention_mask_chunks, independent_mask_chunks, query_mask_chunks,
                              passage_mask_chunks, labels_chunks)):
            if return_rnds:
                rnds.append(RandContext(input_ids, attention_mask, independent_mask, query_mask, passage_mask, labels))
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        independent_mask=independent_mask,
                        query_mask=query_mask,
                        passage_mask=passage_mask,
                        labels=labels,
                        output_attentions=True,
                        return_dict=True,
                        use_cache=False,
                        requires_grad=False,
                        only_encoding=only_encoding,
                    )
                    if not only_encoding:
                        cross_attention_scores = outputs.cross_attentions[-1][:, :, 0]
                        bsz, n_heads, _ = cross_attention_scores.size()
                        scores = cross_attention_scores.view(bsz, n_heads, self.data_args.train_n_passages, -1)
                        decoder_scores = scores.sum(dim=-1).mean(dim=1).detach()
                        decoder_scores_list.append(decoder_scores)

            query_vector, passage_vector = outputs.query_vector, outputs.passage_vector
            if len(query_vector.size()) == 3:
                num_query, seq_len, _ = query_vector.size()
                bsz = num_query // self.data_args.train_n_passages
                query_vector = query_vector.view(bsz, self.data_args.train_n_passages, seq_len, -1).mean(1)
            else:
                bsz = query_vector.size()[0] // self.data_args.train_n_passages
                query_vector = query_vector.view(bsz, self.data_args.train_n_passages, -1).mean(1)

            query_vector_list.append(query_vector)
            passage_vector_list.append(passage_vector)

        if not only_encoding:
            decoder_scores = torch.cat(decoder_scores_list, dim=0)
        query_vector = torch.cat(query_vector_list, dim=0)
        passage_vector = torch.cat(passage_vector_list, dim=0)

        query_mask = torch.cat(query_mask_chunks, dim=0)
        query_mask = query_mask.view(query_mask.size(0) // self.data_args.train_n_passages,
                                     self.data_args.train_n_passages, -1)[:, 0]
        passage_mask = torch.cat(passage_mask_chunks, dim=0)

        if only_encoding:
            return query_vector, passage_vector, rnds, query_mask, passage_mask

        retriever_scores = self.compute_scores(query_vector, passage_vector, query_mask, passage_mask)
        return retriever_scores, decoder_scores

    def compute_scores(self, query_vector, passage_vector, query_mask, passage_mask):
        def compute_colbert_scores(query_vector, passage_vector, query_mask, passage_mask):
            # [num_query, num_passages, q_len, p_len]
            score_list = []
            chunk_query_vector = torch.chunk(query_vector, chunks=query_vector.size()[0], dim=0)
            for chunk in chunk_query_vector:
                scores = chunk.unsqueeze(1) @ passage_vector.unsqueeze(0).transpose(2, 3)
                scores = scores.masked_fill(~passage_mask[None, :, None].bool(), -1e9)
                scores = torch.max(scores, dim=-1).values
                score_list.append(scores)
            scores = torch.cat(score_list, dim=0)
            scores = scores.masked_fill(~query_mask[:, None].bool(), 0.0)
            scores = torch.sum(scores, dim=-1)

            return scores

        if self.training_args.negatives_x_device:
            all_query_vector = self.dist_gather_tensor(query_vector)
            all_passage_vector = self.dist_gather_tensor(passage_vector)
            all_query_mask = self.dist_gather_tensor(query_mask)
            all_passage_mask = self.dist_gather_tensor(passage_mask)
            assert all_query_mask.size()[0] == all_query_vector.size()[0], (
                all_query_mask.size(), all_query_vector.size())
            assert all_passage_mask.size()[0] == all_passage_vector.size()[0], (
                all_passage_mask.size(), all_passage_vector.size())
            if len(all_query_vector.size()) == 3:
                retriever_scores = compute_colbert_scores(all_query_vector, all_passage_vector, all_query_mask,
                                                          all_passage_mask)
            else:
                retriever_scores = torch.matmul(all_query_vector, all_passage_vector.transpose(0, 1))
        else:
            if len(query_vector.size()) == 3:
                retriever_scores = compute_colbert_scores(query_vector, passage_vector, query_mask, passage_mask)
                print(query_vector.size(), passage_vector.size(), retriever_scores)
            else:
                retriever_scores = torch.matmul(query_vector, passage_vector.transpose(0, 1))
        return retriever_scores

    def compute_loss(self, inputs, global_step=None):
        teacher_inputs, student_inputs = inputs

        teacher_retriever_scores, teacher_decoder_scores = self._forward(self.teacher_model, teacher_inputs)

        query_vector, passage_vector, rnds, query_mask, passage_mask = \
            self._forward(self.model, student_inputs, only_encoding=True, return_rnds=True)

        query_vector, cs_query_vector = torch.chunk(query_vector, chunks=2, dim=0)
        passage_vector, cs_passage_vector = torch.chunk(passage_vector, chunks=2, dim=0)
        query_mask, cs_query_mask = torch.chunk(query_mask, chunks=2, dim=0)
        passage_mask, cs_passage_mask = torch.chunk(passage_mask, chunks=2, dim=0)

        query_vector = query_vector.float().detach().requires_grad_()
        passage_vector = passage_vector.float().detach().requires_grad_()
        retriever_scores = self.compute_scores(query_vector, passage_vector, query_mask, passage_mask)

        cs_query_vector = cs_query_vector.float().detach().requires_grad_()
        cs_passage_vector = cs_passage_vector.float().detach().requires_grad_()
        cs_retriever_scores = self.compute_scores(cs_query_vector, cs_passage_vector, cs_query_mask, cs_passage_mask)

        teacher_retriever_logits = torch.softmax(teacher_retriever_scores, dim=-1)
        retriever_logits = torch.log_softmax(retriever_scores, dim=-1)
        retriever_loss = torch.nn.functional.kl_div(retriever_logits, teacher_retriever_logits,
                                                    reduction='batchmean')

        cs_retriever_logits = torch.log_softmax(cs_retriever_scores, dim=-1)
        cs_retriever_loss = torch.nn.functional.kl_div(cs_retriever_logits, teacher_retriever_logits,
                                                       reduction='batchmean') + \
                            torch.nn.functional.kl_div(cs_retriever_logits,
                                                       torch.softmax(retriever_scores.detach(), dim=-1),
                                                       reduction='batchmean')
        retriever_loss = retriever_loss + cs_retriever_loss

        retriever_loss = retriever_loss * self.training_args.retriever_weight

        retriever_loss.backward()

        _, reader_inputs, _, _ = student_inputs

        inputs_ids, attention_mask, independent_mask, query_mask, passage_mask, labels = \
            reader_inputs['input_ids'], reader_inputs['attention_mask'], reader_inputs['independent_mask'], \
            reader_inputs['query_mask'], reader_inputs['passage_mask'], reader_inputs['labels']
        self.training_args.gc_chunk_size = labels.size()[0]
        input_ids_chunks = torch.chunk(inputs_ids, chunks=self.training_args.gc_chunk_size, dim=0)
        attention_mask_chunks = torch.chunk(attention_mask, chunks=self.training_args.gc_chunk_size, dim=0)
        independent_mask_chunks = torch.chunk(independent_mask, chunks=self.training_args.gc_chunk_size, dim=0)
        labels_chunks = torch.chunk(labels, chunks=self.training_args.gc_chunk_size, dim=0)

        query_mask_chunks = torch.chunk(query_mask, chunks=self.training_args.gc_chunk_size, dim=0)
        passage_mask_chunks = torch.chunk(passage_mask, chunks=self.training_args.gc_chunk_size, dim=0)

        reader_loss, reader_decoder_attn_loss = 0, 0
        query_grads_chunk = torch.chunk(torch.cat([query_vector.grad, cs_query_vector.grad], dim=0),
                                        chunks=self.training_args.gc_chunk_size, dim=0)
        passage_grads_chunk = torch.chunk(torch.cat([passage_vector.grad, cs_passage_vector.grad], dim=0),
                                          chunks=self.training_args.gc_chunk_size, dim=0)
        teacher_decoder_scores = torch.cat([teacher_decoder_scores, teacher_decoder_scores], dim=0)
        lm_logits_list = []
        for idx, (
                input_ids, attention_mask, independent_mask, query_mask, passage_mask, labels, query_grads,
                passage_grads,
                rnd) in \
                enumerate(zip(input_ids_chunks, attention_mask_chunks, independent_mask_chunks, query_mask_chunks,
                              passage_mask_chunks, labels_chunks, query_grads_chunk, passage_grads_chunk, rnds)):
            def chunk_forward():
                with rnd:
                    with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            independent_mask=independent_mask,
                            query_mask=query_mask,
                            passage_mask=passage_mask,
                            labels=labels,
                            output_attentions=True,
                            return_dict=True,
                            use_cache=False,
                        )
                        loss = outputs.loss / self.training_args.gc_chunk_size
                        query_vector, passage_vector = outputs.query_vector, outputs.passage_vector
                        if len(query_vector.size()) == 3:
                            num_query, seq_len, _ = query_vector.size()
                            bsz = num_query // self.data_args.train_n_passages
                            query_vector = query_vector.view(bsz, self.data_args.train_n_passages, seq_len, -1).mean(1)
                        else:
                            bsz = query_vector.size()[0] // self.data_args.train_n_passages
                            query_vector = query_vector.view(bsz, self.data_args.train_n_passages, -1).mean(1)
                        surrogate = torch.dot(query_vector.flatten().float(), query_grads.flatten()) + \
                                    torch.dot(passage_vector.flatten().float(), passage_grads.flatten())

                        if idx < self.training_args.gc_chunk_size // 2:
                            lm_logits_list.append(outputs.logits.detach())
                        else:
                            mask = labels.view(-1) != -100
                            lm_logits = outputs.logits
                            lm_logits = torch.log_softmax(lm_logits.view(-1, lm_logits.size(-1))[mask], dim=-1)
                            teacher_lm_logits = lm_logits_list[idx - self.training_args.gc_chunk_size // 2]
                            teacher_lm_logits = torch.softmax(
                                teacher_lm_logits.view(-1, teacher_lm_logits.size(-1)),
                                dim=-1)
                            lm_alignment = torch.nn.functional.kl_div(lm_logits, teacher_lm_logits[mask],
                                                                      reduction='batchmean')
                            loss += lm_alignment / (self.training_args.gc_chunk_size // 2)

                        cross_attention_scores = outputs.cross_attentions[-1][:, :, 0]
                        bsz, n_heads, _ = cross_attention_scores.size()
                        scores = cross_attention_scores.view(bsz, n_heads, self.data_args.train_n_passages, -1)
                        decoder_scores = scores.sum(dim=-1).mean(dim=1)
                        decoder_attn_alignment = torch.nn.functional.kl_div(decoder_scores.log(),
                                                                            teacher_decoder_scores[idx],
                                                                            reduction='batchmean')
                        decoder_attn_alignment = decoder_attn_alignment / self.training_args.gc_chunk_size

                surrogate = surrogate * self._dist_loss_scale_factor + loss + decoder_attn_alignment
                if self.use_amp:
                    self.scaler.scale(surrogate).backward()
                else:
                    surrogate.backward()

                return loss + decoder_attn_alignment

            if idx != len(labels_chunks) - 1:
                with self.model.no_sync():
                    loss = chunk_forward()
            else:
                loss = chunk_forward()

            reader_loss = reader_loss + loss

        return reader_loss + retriever_loss

    def train_step(self, batch, global_step=None):
        return self.compute_loss(batch, global_step)
