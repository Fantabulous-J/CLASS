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

from dataloader import GenericDataLoader, EncodeDataset, CXMIDataset, EncodeCollator, ReaderDataset, ReaderCollator
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
            # if isinstance(self.train_dataloader, list):
            #     self.max_step = self.training_args.max_steps * len(self.train_dataloader)
            # else:
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
        _, query, passage, reader_inputs, _, _ = inputs

        if global_step < self.training_args.distillation_start_steps:
            outputs = self.model(**reader_inputs, return_dict=True)
            return outputs.loss

        self.model.eval()
        with torch.no_grad():
            self.model.requires_grad_(False)
            decoder_input_ids = torch.ones(reader_inputs['labels'].size(0), 1, dtype=torch.long,
                                           device=self.training_args.device) * self.config.decoder_start_token_id
            outputs = self.model(
                input_ids=reader_inputs['input_ids'],
                attention_mask=reader_inputs['attention_mask'],
                decoder_input_ids=decoder_input_ids,
                output_attentions=True,
                return_dict=True,
            )
            self.model.requires_grad_(True)
        cross_attention_scores = outputs.cross_attentions[-1][:, :, 0]
        bsz, n_heads, _ = cross_attention_scores.size()
        scores = cross_attention_scores.view(bsz, n_heads, self.data_args.train_n_passages, -1)
        decoder_scores = scores.sum(dim=-1).mean(dim=1).detach()
        self.model.train()

        outputs = self.model(**reader_inputs, query=query, passage=passage, return_dict=True)
        reader_loss = outputs.loss

        query_vector, passage_vector = outputs.query_vector, outputs.passage_vector
        if self.training_args.negatives_x_device:
            query_vector = self.dist_gather_tensor(query_vector)
            passage_vector = self.dist_gather_tensor(passage_vector)
            decoder_scores = self.dist_gather_tensor(decoder_scores)
        retriever_scores = torch.matmul(query_vector, passage_vector.transpose(0, 1))

        assert decoder_scores.size()[0] == retriever_scores.size()[0], (decoder_scores.size(), retriever_scores.size())
        teacher_scores = torch.zeros_like(retriever_scores, device=self.training_args.device)
        bsz = decoder_scores.size()[0]
        for i in range(bsz):
            start = i * self.data_args.train_n_passages
            end = (i + 1) * self.data_args.train_n_passages
            teacher_scores[i, start:end] = decoder_scores[i]

        retriever_logits = torch.log_softmax(retriever_scores, dim=-1)
        retriever_loss = torch.nn.functional.kl_div(retriever_logits, teacher_scores, reduction='batchmean')
        loss = reader_loss + retriever_loss * self._dist_loss_scale_factor

        return loss / self.training_args.gradient_accumulation_steps

    def grad_cache_compute_loss(self, inputs, global_step=None):
        _, query, passage, reader_inputs, _, _ = inputs

        inputs_ids, attention_mask, labels = reader_inputs['input_ids'], reader_inputs['attention_mask'], \
                                             reader_inputs['labels']
        self.training_args.gc_chunk_size = labels.size()[0]
        input_ids_chunks = torch.chunk(inputs_ids, chunks=self.training_args.gc_chunk_size, dim=0)
        attention_mask_chunks = torch.chunk(attention_mask, chunks=self.training_args.gc_chunk_size, dim=0)
        labels_chunks = torch.chunk(labels, chunks=self.training_args.gc_chunk_size, dim=0)

        reader_loss = 0
        for idx, (input_ids, attention_mask, labels) in enumerate(
                zip(input_ids_chunks, attention_mask_chunks, labels_chunks)):
            def chunk_forward():
                with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                    loss = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
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

        if global_step < self.training_args.distillation_start_steps:
            return reader_loss

        self.model.eval()
        with torch.no_grad():
            self.model.requires_grad_(False)
            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                decoder_scores_list = []
                for input_ids, attention_mask, labels in zip(input_ids_chunks, attention_mask_chunks, labels_chunks):
                    decoder_input_ids = torch.ones(labels.size(0), 1, dtype=torch.long,
                                                   device=self.training_args.device) * self.config.decoder_start_token_id
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        decoder_input_ids=decoder_input_ids,
                        output_attentions=True,
                        return_dict=True,
                    )
                    cross_attention_scores = outputs.cross_attentions[-1][:, :, 0]
                    bsz, n_heads, _ = cross_attention_scores.size()
                    scores = cross_attention_scores.view(bsz, n_heads, self.data_args.train_n_passages, -1)
                    decoder_scores = scores.sum(dim=-1).mean(dim=1).detach()
                    decoder_scores_list.append(decoder_scores)
                decoder_scores = torch.cat(decoder_scores_list, dim=0)
            self.model.requires_grad_(True)
        self.model.train()

        def compute_vector(inputs_ids_chunks, attention_mask_chunks, is_query=False):
            vector_list, rnds = [], []
            self.model.requires_grad_(False)
            for input_ids, attention_mask in zip(inputs_ids_chunks, attention_mask_chunks):
                rnds.append(RandContext(input_ids, attention_mask))
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                        inputs = {
                            "query": {
                                "input_ids": input_ids,
                                "attention_mask": attention_mask,
                            } if is_query else None,
                            "passage": {
                                "input_ids": input_ids,
                                "attention_mask": attention_mask,
                            } if not is_query else None,
                        }
                        outputs = self.model(
                            **inputs, only_encoding=True, return_dict=True
                        )
                        vector = outputs.query_vector if is_query else outputs.passage_vector
                vector_list.append(vector)
            self.model.requires_grad_(True)
            vector = torch.cat(vector_list, dim=0)
            return vector, rnds

        query_inputs_ids, query_attention_mask = query['input_ids'], query['attention_mask']
        query_inputs_ids_chunks = torch.chunk(query_inputs_ids, chunks=self.training_args.gc_chunk_size, dim=0)
        query_attention_mask_chunks = torch.chunk(query_attention_mask, chunks=self.training_args.gc_chunk_size, dim=0)
        query_vector, query_rnds = compute_vector(query_inputs_ids_chunks, query_attention_mask_chunks, is_query=True)

        passage_inputs_ids, passage_attention_mask = passage['input_ids'], passage['attention_mask']
        passage_inputs_ids_chunks = torch.chunk(passage_inputs_ids, chunks=self.training_args.gc_chunk_size, dim=0)
        passage_attention_mask_chunks = torch.chunk(passage_attention_mask, chunks=self.training_args.gc_chunk_size,
                                                    dim=0)
        passage_vector, passage_rnds = compute_vector(passage_inputs_ids_chunks, passage_attention_mask_chunks)

        query_vector = query_vector.float().detach().requires_grad_()
        passage_vector = passage_vector.float().detach().requires_grad_()
        if self.training_args.negatives_x_device:
            all_query_vector = self.dist_gather_tensor(query_vector)
            all_passage_vector = self.dist_gather_tensor(passage_vector)
            decoder_scores = self.dist_gather_tensor(decoder_scores)
            retriever_scores = torch.matmul(all_query_vector, all_passage_vector.transpose(0, 1))
        else:
            retriever_scores = torch.matmul(query_vector, passage_vector.transpose(0, 1))

        assert decoder_scores.size()[0] == retriever_scores.size()[0], (decoder_scores.size(), retriever_scores.size())
        teacher_scores = torch.zeros_like(retriever_scores, device=self.training_args.device)
        bsz = decoder_scores.size()[0]
        for i in range(bsz):
            start = i * self.data_args.train_n_passages
            end = (i + 1) * self.data_args.train_n_passages
            teacher_scores[i, start:end] = decoder_scores[i]

        retriever_logits = torch.log_softmax(retriever_scores, dim=-1)
        retriever_loss = torch.nn.functional.kl_div(retriever_logits, teacher_scores, reduction='batchmean')
        retriever_loss.backward()

        def vector_forward(inputs_ids_chunks, attention_mask_chunks, grads_chunk, rnds, is_query=False):
            for idx, (input_ids, attention_mask, grads, rnd) in \
                    enumerate(zip(inputs_ids_chunks, attention_mask_chunks, grads_chunk, rnds)):
                def chunk_forward():
                    with rnd:
                        with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                            inputs = {
                                "query": {
                                    "input_ids": input_ids,
                                    "attention_mask": attention_mask,
                                } if is_query else None,
                                "passage": {
                                    "input_ids": input_ids,
                                    "attention_mask": attention_mask,
                                } if not is_query else None,
                            }
                            outputs = self.model(
                                **inputs, only_encoding=True, return_dict=True
                            )
                            vector = outputs.query_vector if is_query else outputs.passage_vector
                            surrogate = torch.dot(vector.flatten().float(), grads.flatten())
                            sum_of_params = sum([p.sum() for p in self.model.parameters()])
                    surrogate = surrogate * self._dist_loss_scale_factor + 0. * sum_of_params
                    if self.use_amp:
                        self.scaler.scale(surrogate).backward()
                    else:
                        surrogate.backward()

                if idx != len(inputs_ids_chunks) - 1:
                    with self.model.no_sync():
                        chunk_forward()
                else:
                    chunk_forward()

        query_grads_chunk = torch.chunk(query_vector.grad, chunks=self.training_args.gc_chunk_size, dim=0)
        vector_forward(query_inputs_ids_chunks, query_attention_mask_chunks, query_grads_chunk, query_rnds,
                       is_query=True)

        passage_grads_chunk = torch.chunk(passage_vector.grad, chunks=self.training_args.gc_chunk_size, dim=0)
        vector_forward(passage_inputs_ids_chunks, passage_attention_mask_chunks, passage_grads_chunk, passage_rnds)

        return reader_loss + retriever_loss

    def separate_joint_encoding_compute_loss(self, inputs, global_step=None):
        # _, reader_inputs, _, _ = inputs
        #
        # inputs_ids, attention_mask, independent_mask, query_mask, passage_mask, labels = \
        #     reader_inputs['input_ids'], reader_inputs['attention_mask'], reader_inputs['independent_mask'], \
        #     reader_inputs['query_mask'], reader_inputs['passage_mask'], reader_inputs['labels']
        # self.training_args.gc_chunk_size = labels.size()[0]
        # input_ids_chunks = torch.chunk(inputs_ids, chunks=self.training_args.gc_chunk_size, dim=0)
        # attention_mask_chunks = torch.chunk(attention_mask, chunks=self.training_args.gc_chunk_size, dim=0)
        # independent_mask_chunks = torch.chunk(independent_mask, chunks=self.training_args.gc_chunk_size, dim=0)
        # labels_chunks = torch.chunk(labels, chunks=self.training_args.gc_chunk_size, dim=0)
        #
        # if global_step < self.training_args.distillation_start_steps:
        #     reader_loss = 0
        #     for idx, (input_ids, attention_mask, independent_mask, labels) in \
        #             enumerate(zip(input_ids_chunks, attention_mask_chunks, independent_mask_chunks, labels_chunks)):
        #         def chunk_forward():
        #             with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
        #                 loss = self.model(
        #                     input_ids=input_ids,
        #                     attention_mask=attention_mask,
        #                     independent_mask=independent_mask,
        #                     labels=labels,
        #                     use_cache=False,
        #                 ).loss
        #                 loss /= self.training_args.gc_chunk_size
        #             if self.use_amp:
        #                 self.scaler.scale(loss).backward()
        #             else:
        #                 loss.backward()
        #             return loss
        #
        #         if idx != len(labels_chunks) - 1:
        #             with self.model.no_sync():
        #                 loss = chunk_forward()
        #         else:
        #             loss = chunk_forward()
        #         reader_loss = reader_loss + loss
        #     return reader_loss
        #
        # query_mask_chunks = torch.chunk(query_mask, chunks=self.training_args.gc_chunk_size, dim=0)
        # passage_mask_chunks = torch.chunk(passage_mask, chunks=self.training_args.gc_chunk_size, dim=0)
        #
        # query_vector_list, passage_vector_list = [], []
        # rnds = []
        # decoder_scores_list = []
        # for idx, (input_ids, attention_mask, independent_mask, query_mask, passage_mask, labels) in \
        #         enumerate(zip(input_ids_chunks, attention_mask_chunks, independent_mask_chunks, query_mask_chunks,
        #                       passage_mask_chunks, labels_chunks)):
        #     rnds.append(RandContext(input_ids, attention_mask, independent_mask, query_mask, passage_mask, labels))
        #     with torch.no_grad():
        #         with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
        #             outputs = self.model(
        #                 input_ids=input_ids,
        #                 attention_mask=attention_mask,
        #                 independent_mask=independent_mask,
        #                 query_mask=query_mask,
        #                 passage_mask=passage_mask,
        #                 labels=labels,
        #                 output_attentions=True,
        #                 return_dict=True,
        #                 use_cache=False,
        #                 requires_grad=False,
        #             )
        #
        #     query_vector, passage_vector = outputs.query_vector, outputs.passage_vector
        #     if len(query_vector.size()) == 3:
        #         num_query, seq_len, _ = query_vector.size()
        #         bsz = num_query // self.data_args.train_n_passages
        #         query_vector = query_vector.view(bsz, self.data_args.train_n_passages, seq_len, -1).mean(1)
        #     else:
        #         bsz = query_vector.size()[0] // self.data_args.train_n_passages
        #         query_vector.view(bsz, self.data_args.train_n_passages, -1).mean(1)
        #
        #     cross_attention_scores = outputs.cross_attentions[-1][:, :, 0]
        #     bsz, n_heads, _ = cross_attention_scores.size()
        #     scores = cross_attention_scores.view(bsz, n_heads, self.data_args.train_n_passages, -1)
        #     decoder_scores = scores.sum(dim=-1).mean(dim=1).detach()
        #
        #     decoder_scores_list.append(decoder_scores)
        #     query_vector_list.append(query_vector)
        #     passage_vector_list.append(passage_vector)
        #
        # decoder_scores = torch.cat(decoder_scores_list, dim=0)
        # query_vector = torch.cat(query_vector_list, dim=0)
        # passage_vector = torch.cat(passage_vector_list, dim=0)
        #
        # query_vector = query_vector.float().detach().requires_grad_()
        # passage_vector = passage_vector.float().detach().requires_grad_()
        #
        # def compute_colbert_scores(query_vector, passage_vector, query_mask, passage_mask):
        #     # [num_query, num_passages, q_len, p_len]
        #     score_list = []
        #     chunk_query_vector = torch.chunk(query_vector, chunks=query_vector.size()[0], dim=0)
        #     for chunk in chunk_query_vector:
        #         scores = chunk.unsqueeze(1) @ passage_vector.unsqueeze(0).transpose(2, 3)
        #         scores = scores.masked_fill(~passage_mask[None, :, None].bool(), -1e9)
        #         scores = torch.max(scores, dim=-1).values
        #         score_list.append(scores)
        #     scores = torch.cat(score_list, dim=0)
        #     scores = scores.masked_fill(~query_mask[:, None].bool(), 0.0)
        #     scores = torch.sum(scores, dim=-1) / query_mask.sum(dim=1)[..., None]
        #
        #     return scores
        #
        # if self.training_args.negatives_x_device:
        #     all_query_vector = self.dist_gather_tensor(query_vector)
        #     all_passage_vector = self.dist_gather_tensor(passage_vector)
        #     decoder_scores = self.dist_gather_tensor(decoder_scores)
        #     all_query_mask = self.dist_gather_tensor(torch.cat(query_mask_chunks, dim=0))
        #     all_query_mask = all_query_mask.view(all_query_mask.size(0) // self.data_args.train_n_passages,
        #                                          self.data_args.train_n_passages, -1)[:, 0]
        #     all_passage_mask = self.dist_gather_tensor(torch.cat(passage_mask_chunks, dim=0))
        #     retriever_scores = compute_colbert_scores(all_query_vector, all_passage_vector, all_query_mask,
        #                                               all_passage_mask)
        # else:
        #     all_query_mask = torch.cat(query_mask_chunks, dim=0)
        #     all_query_mask = all_query_mask.view(all_query_mask.size(0) // self.data_args.train_n_passages,
        #                                          self.data_args.train_n_passages, -1)[:, 0]
        #     all_passage_mask = torch.cat(passage_mask_chunks, dim=0)
        #     retriever_scores = compute_colbert_scores(query_vector, passage_vector, all_query_mask, all_passage_mask)
        #
        # assert decoder_scores.size()[0] == retriever_scores.size()[0], (decoder_scores.size(), retriever_scores.size())
        # teacher_scores = torch.zeros_like(retriever_scores, device=self.training_args.device)
        # bsz = decoder_scores.size()[0]
        # for i in range(bsz):
        #     start = i * self.data_args.train_n_passages
        #     end = (i + 1) * self.data_args.train_n_passages
        #     teacher_scores[i, start:end] = decoder_scores[i]
        #
        # retriever_logits = torch.log_softmax(retriever_scores, dim=-1)
        # retriever_loss = torch.nn.functional.kl_div(retriever_logits, teacher_scores,
        #                                             reduction='batchmean') * self.training_args.retriever_weight
        # retriever_loss.backward()
        #
        # reader_loss = 0
        # query_grads_chunk = torch.chunk(query_vector.grad, chunks=self.training_args.gc_chunk_size, dim=0)
        # passage_grads_chunk = torch.chunk(passage_vector.grad, chunks=self.training_args.gc_chunk_size, dim=0)
        # for idx, (input_ids, attention_mask, independent_mask, query_mask, passage_mask, labels, query_grads, passage_grads, rnd) in \
        #         enumerate(zip(input_ids_chunks, attention_mask_chunks, independent_mask_chunks, query_mask_chunks,
        #                       passage_mask_chunks, labels_chunks, query_grads_chunk, passage_grads_chunk, rnds)):
        #     def chunk_forward():
        #         with rnd:
        #             with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
        #                 outputs = self.model(
        #                     input_ids=input_ids,
        #                     attention_mask=attention_mask,
        #                     independent_mask=independent_mask,
        #                     query_mask=query_mask,
        #                     passage_mask=passage_mask,
        #                     labels=labels,
        #                     output_attentions=True,
        #                     return_dict=True,
        #                     use_cache=False,
        #                 )
        #                 loss = outputs.loss / self.training_args.gc_chunk_size
        #                 query_vector, passage_vector = outputs.query_vector, outputs.passage_vector
        #                 if len(query_vector.size()) == 3:
        #                     num_query, seq_len, _ = query_vector.size()
        #                     bsz = num_query // self.data_args.train_n_passages
        #                     query_vector = query_vector.view(bsz, self.data_args.train_n_passages, seq_len, -1).mean(1)
        #                 else:
        #                     bsz = query_vector.size()[0] // self.data_args.train_n_passages
        #                     query_vector.view(bsz, self.data_args.train_n_passages, -1).mean(1)
        #                 surrogate = torch.dot(query_vector.flatten().float(), query_grads.flatten()) + \
        #                             torch.dot(passage_vector.flatten().float(), passage_grads.flatten())
        #         surrogate = surrogate * self._dist_loss_scale_factor + loss
        #         if self.use_amp:
        #             self.scaler.scale(surrogate).backward()
        #         else:
        #             surrogate.backward()
        #
        #         return loss
        #
        #     if idx != len(labels_chunks) - 1:
        #         with self.model.no_sync():
        #             loss = chunk_forward()
        #     else:
        #         loss = chunk_forward()
        #
        #     reader_loss = reader_loss + loss
        #
        # return reader_loss + retriever_loss

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
                            add_bias=self.training_args.add_bias,
                            bias_scores=bias_scores,
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
                    # todo: before adding below code, check if this kind of bias can be used for reranking (works)
                    #  1. add bias to every cross-attention score layer
                    #  2. a linear layer on top encoder's joint encodings and use the output to mimic the bias
                    #  3. adjust the loss
                    #  4. adjust the inference code where we use the linear layer's output as bias
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
                        add_bias=self.training_args.add_bias,
                        bias_scores=bias_scores,
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
                        # retriever_scores = torch.matmul(query_vector, passage_vector.transpose(0, 1)) / \
                        #                    math.sqrt(query_vector.size()[-1])
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

    def cross_lingual_alignment_compute_loss(self, inputs, global_step=None):
        if self.training_args.separate_joint_encoding:
            _, reader_inputs, _, _ = inputs

            inputs_ids, attention_mask, independent_mask, query_mask, passage_mask, labels = \
                reader_inputs['input_ids'], reader_inputs['attention_mask'], reader_inputs['independent_mask'], \
                reader_inputs['query_mask'], reader_inputs['passage_mask'], reader_inputs['labels']
            self.training_args.gc_chunk_size = labels.size()[0]
            input_ids_chunks = torch.chunk(inputs_ids, chunks=self.training_args.gc_chunk_size, dim=0)
            attention_mask_chunks = torch.chunk(attention_mask, chunks=self.training_args.gc_chunk_size, dim=0)
            independent_mask_chunks = torch.chunk(independent_mask, chunks=self.training_args.gc_chunk_size, dim=0)
            labels_chunks = torch.chunk(labels, chunks=self.training_args.gc_chunk_size, dim=0)

            if global_step < self.training_args.distillation_start_steps:
                reader_loss = 0
                for idx, (input_ids, attention_mask, independent_mask, labels) in \
                        enumerate(zip(input_ids_chunks, attention_mask_chunks, independent_mask_chunks, labels_chunks)):
                    def chunk_forward():
                        with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                            loss = self.model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                independent_mask=independent_mask,
                                labels=labels,
                                use_cache=False,
                            ).loss
                            loss /= (self.training_args.gc_chunk_size // 2)
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

            query_vector_list, passage_vector_list, decoder_scores_list, lm_logits_list = [], [], [], []
            rnds = []
            for idx, (input_ids, attention_mask, independent_mask, query_mask, passage_mask, labels) in \
                    enumerate(zip(input_ids_chunks, attention_mask_chunks, independent_mask_chunks, query_mask_chunks,
                                  passage_mask_chunks, labels_chunks)):
                rnds.append(RandContext(input_ids, attention_mask, independent_mask, query_mask, passage_mask, labels))
                with torch.no_grad():
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
                            requires_grad=False,
                        )
                        cross_attention_scores = outputs.cross_attentions[-1][:, :, 0]
                        bsz, n_heads, _ = cross_attention_scores.size()
                        scores = cross_attention_scores.view(bsz, n_heads, self.data_args.train_n_passages, -1)
                        decoder_scores = scores.sum(dim=-1).mean(dim=1).detach()
                        decoder_scores_list.append(decoder_scores)
                        lm_logits_list.append(outputs.logits.detach())

                query_vector, passage_vector = outputs.query_vector, outputs.passage_vector
                query_vector = query_vector.view(labels.size(0), self.data_args.train_n_passages, -1).mean(1)
                query_vector_list.append(query_vector)
                passage_vector_list.append(passage_vector)

            decoder_scores = torch.cat(decoder_scores_list, dim=0)
            query_vector = torch.cat(query_vector_list, dim=0)
            passage_vector = torch.cat(passage_vector_list, dim=0)
            lm_logits = torch.cat(lm_logits_list, dim=0)

            query_vector, cs_query_vector = torch.chunk(query_vector, chunks=2, dim=0)
            passage_vector, cs_passage_vector = torch.chunk(passage_vector, chunks=2, dim=0)
            decoder_scores, cs_decoder_scores = torch.chunk(decoder_scores, chunks=2, dim=0)

            def compute_retriever_scores(query_vector, passage_vector, decoder_scores):
                if self.training_args.negatives_x_device:
                    all_query_vector = self.dist_gather_tensor(query_vector)
                    all_passage_vector = self.dist_gather_tensor(passage_vector)
                    decoder_scores = self.dist_gather_tensor(decoder_scores)
                    retriever_scores = torch.matmul(all_query_vector, all_passage_vector.transpose(0, 1))
                else:
                    retriever_scores = torch.matmul(query_vector, passage_vector.transpose(0, 1))

                assert decoder_scores.size()[0] == retriever_scores.size()[0], \
                    (decoder_scores.size(), retriever_scores.size())
                teacher_scores = torch.zeros_like(retriever_scores, device=self.training_args.device)
                bsz = decoder_scores.size()[0]
                for i in range(bsz):
                    start = i * self.data_args.train_n_passages
                    end = (i + 1) * self.data_args.train_n_passages
                    teacher_scores[i, start:end] = decoder_scores[i]

                return retriever_scores, teacher_scores

            query_vector = query_vector.float().detach().requires_grad_()
            passage_vector = passage_vector.float().detach().requires_grad_()
            retriever_scores, teacher_scores = compute_retriever_scores(query_vector, passage_vector, decoder_scores)
            retriever_logits = torch.log_softmax(retriever_scores, dim=-1)
            retriever_loss = torch.nn.functional.kl_div(retriever_logits, teacher_scores, reduction='batchmean')

            cs_query_vector = cs_query_vector.float().detach().requires_grad_()
            cs_passage_vector = cs_passage_vector.float().detach().requires_grad_()
            cs_retriever_scores, cs_teacher_scores = compute_retriever_scores(cs_query_vector, cs_passage_vector,
                                                                              cs_decoder_scores)
            cs_retriever_logits = torch.log_softmax(cs_retriever_scores, dim=-1)
            cs_retriever_loss = torch.nn.functional.kl_div(cs_retriever_logits, cs_teacher_scores,
                                                           reduction='batchmean') \
                                + torch.nn.functional.kl_div(cs_retriever_logits,
                                                             torch.softmax(retriever_scores.detach(), dim=-1),
                                                             reduction='batchmean')
            retriever_loss += cs_retriever_loss

            if self.training_args.query_alignment:
                if self.training_args.negatives_x_device:
                    all_query_vector = self.dist_gather_tensor(query_vector)
                    all_cs_query_vector = self.dist_gather_tensor(cs_query_vector)
                    query_scores = torch.matmul(all_query_vector, all_cs_query_vector.transpose(0, 1))
                else:
                    query_scores = torch.matmul(query_vector, cs_query_vector.transpose(0, 1))

                target = torch.arange(query_scores.size()[0], device=self.training_args.device, dtype=torch.long)
                query_loss = torch.nn.functional.cross_entropy(query_scores, target)
                retriever_loss += query_loss

            retriever_loss.backward()

            reader_loss = 0
            query_grads_chunk = torch.chunk(torch.cat([query_vector.grad, cs_query_vector.grad], dim=0),
                                            chunks=self.training_args.gc_chunk_size, dim=0)
            passage_grads_chunk = torch.chunk(torch.cat([passage_vector.grad, cs_passage_vector.grad], dim=0),
                                              chunks=self.training_args.gc_chunk_size, dim=0)
            lm_logits_chunk = torch.chunk(torch.chunk(lm_logits, chunks=2, dim=0)[0],
                                          chunks=self.training_args.gc_chunk_size // 2, dim=0)
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
                            loss = outputs.loss / (self.training_args.gc_chunk_size // 2)
                            query_vector, passage_vector = outputs.query_vector, outputs.passage_vector
                            query_vector = query_vector.view(labels.size(0), self.data_args.train_n_passages, -1).mean(
                                1)
                            surrogate = torch.dot(query_vector.flatten().float(), query_grads.flatten()) + \
                                        torch.dot(passage_vector.flatten().float(), passage_grads.flatten())

                    surrogate = surrogate * self._dist_loss_scale_factor + loss
                    if idx >= self.training_args.gc_chunk_size // 2:
                        mask = labels.view(-1) != -100
                        lm_logits = outputs.logits
                        lm_logits = torch.log_softmax(lm_logits.view(-1, lm_logits.size(-1))[mask], dim=-1)
                        teacher_lm_logits = lm_logits_chunk[idx - self.training_args.gc_chunk_size // 2]
                        teacher_lm_logits = torch.softmax(teacher_lm_logits.view(-1, teacher_lm_logits.size(-1)),
                                                          dim=-1)
                        lm_alignment = torch.nn.functional.kl_div(lm_logits, teacher_lm_logits[mask],
                                                                  reduction='batchmean')
                        surrogate += lm_alignment / (self.training_args.gc_chunk_size // 2)
                    if self.use_amp:
                        self.scaler.scale(surrogate).backward()
                    else:
                        surrogate.backward()
                    return loss

                if idx != len(labels_chunks) - 1:
                    with self.model.no_sync():
                        loss = chunk_forward()
                else:
                    loss = chunk_forward()
                reader_loss = reader_loss + loss

            return reader_loss + retriever_loss

        _, query, passage, reader_inputs, _, _ = inputs

        inputs_ids, attention_mask, labels = reader_inputs['input_ids'], reader_inputs['attention_mask'], \
                                             reader_inputs['labels']
        self.training_args.gc_chunk_size = labels.size()[0]
        input_ids_chunks = torch.chunk(inputs_ids, chunks=self.training_args.gc_chunk_size, dim=0)
        attention_mask_chunks = torch.chunk(attention_mask, chunks=self.training_args.gc_chunk_size, dim=0)
        labels_chunks = torch.chunk(labels, chunks=self.training_args.gc_chunk_size, dim=0)

        if global_step < self.training_args.distillation_start_steps:
            reader_loss = 0
            for idx, (input_ids, attention_mask, labels) in enumerate(
                    zip(input_ids_chunks, attention_mask_chunks, labels_chunks)):
                def chunk_forward():
                    with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                        loss = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            return_dict=True,
                        ).loss
                        loss /= (self.training_args.gc_chunk_size // 2)
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

        self.model.eval()
        with torch.no_grad():
            self.model.requires_grad_(False)
            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                decoder_scores_list = []
                lm_logits_list = []
                for input_ids, attention_mask, labels in zip(input_ids_chunks, attention_mask_chunks, labels_chunks):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        output_attentions=True,
                        return_dict=True,
                    )
                    cross_attention_scores = outputs.cross_attentions[-1][:, :, 0]
                    bsz, n_heads, _ = cross_attention_scores.size()
                    scores = cross_attention_scores.view(bsz, n_heads, self.data_args.train_n_passages, -1)
                    decoder_scores = scores.sum(dim=-1).mean(dim=1).detach()
                    decoder_scores_list.append(decoder_scores)
                    lm_logits_list.append(outputs.logits.detach())
                decoder_scores = torch.cat(decoder_scores_list, dim=0)
                lm_logits = torch.cat(lm_logits_list, dim=0)
            self.model.requires_grad_(True)
        self.model.train()

        lm_logits_chunk = torch.chunk(torch.chunk(lm_logits, chunks=2, dim=0)[0],
                                      chunks=self.training_args.gc_chunk_size // 2, dim=0)
        reader_loss = 0
        for idx, (input_ids, attention_mask, labels) in enumerate(
                zip(input_ids_chunks, attention_mask_chunks, labels_chunks)):
            def chunk_forward():
                with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        return_dict=True,
                    )
                    loss = outputs.loss / (self.training_args.gc_chunk_size // 2)
                    if idx >= self.training_args.gc_chunk_size // 2:
                        mask = labels.view(-1) != -100
                        lm_logits = outputs.logits
                        lm_logits = torch.log_softmax(lm_logits.view(-1, lm_logits.size(-1))[mask], dim=-1)
                        teacher_lm_logits = lm_logits_chunk[idx - self.training_args.gc_chunk_size // 2]
                        teacher_lm_logits = torch.softmax(teacher_lm_logits.view(-1, teacher_lm_logits.size(-1)),
                                                          dim=-1)
                        lm_alignment = torch.nn.functional.kl_div(lm_logits, teacher_lm_logits[mask],
                                                                  reduction='batchmean')
                        loss += lm_alignment / (self.training_args.gc_chunk_size // 2)
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

        def compute_vector(inputs_ids_chunks, attention_mask_chunks, is_query=False):
            self.model.requires_grad_(False)
            vector_list = []
            normalised_vector_list = []
            rnds = []
            for input_ids, attention_mask in zip(inputs_ids_chunks, attention_mask_chunks):
                rnds.append(RandContext(input_ids, attention_mask))
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                        inputs = {
                            "query": {
                                "input_ids": input_ids,
                                "attention_mask": attention_mask,
                            } if is_query else None,
                            "passage": {
                                "input_ids": input_ids,
                                "attention_mask": attention_mask,
                            } if not is_query else None,
                        }
                        outputs = self.model(
                            **inputs, only_encoding=True, return_dict=True
                        )
                        vector = outputs.query_vector if is_query else outputs.passage_vector
                vector_list.append(vector)
                normalised_vector_list.append(outputs.normalised_query_vector)
            self.model.requires_grad_(True)
            vector = torch.cat(vector_list, dim=0)
            return vector, rnds, normalised_vector_list

        query_inputs_ids, query_attention_mask = query['input_ids'], query['attention_mask']
        query_inputs_ids_chunks = torch.chunk(query_inputs_ids, chunks=self.training_args.gc_chunk_size, dim=0)
        query_attention_mask_chunks = torch.chunk(query_attention_mask, chunks=self.training_args.gc_chunk_size, dim=0)
        query_vector, query_rnds, normalised_vector = compute_vector(query_inputs_ids_chunks,
                                                                     query_attention_mask_chunks, is_query=True)

        passage_inputs_ids, passage_attention_mask = passage['input_ids'], passage['attention_mask']
        passage_inputs_ids_chunks = torch.chunk(passage_inputs_ids, chunks=self.training_args.gc_chunk_size, dim=0)
        passage_attention_mask_chunks = torch.chunk(passage_attention_mask, chunks=self.training_args.gc_chunk_size,
                                                    dim=0)
        passage_vector, passage_rnds, _ = compute_vector(passage_inputs_ids_chunks, passage_attention_mask_chunks)

        query_vector, cs_query_vector = torch.chunk(query_vector, chunks=2, dim=0)
        passage_vector, cs_passage_vector = torch.chunk(passage_vector, chunks=2, dim=0)
        decoder_scores, cs_decoder_scores = torch.chunk(decoder_scores, chunks=2, dim=0)

        def compute_retriever_scores(query_vector, passage_vector, decoder_scores):
            if self.training_args.negatives_x_device:
                all_query_vector = self.dist_gather_tensor(query_vector)
                all_passage_vector = self.dist_gather_tensor(passage_vector)
                decoder_scores = self.dist_gather_tensor(decoder_scores)
                retriever_scores = torch.matmul(all_query_vector, all_passage_vector.transpose(0, 1))
            else:
                retriever_scores = torch.matmul(query_vector, passage_vector.transpose(0, 1))

            assert decoder_scores.size()[0] == retriever_scores.size()[0], (
                decoder_scores.size(), retriever_scores.size())
            teacher_scores = torch.zeros_like(retriever_scores, device=self.training_args.device)
            bsz = decoder_scores.size()[0]
            for i in range(bsz):
                start = i * self.data_args.train_n_passages
                end = (i + 1) * self.data_args.train_n_passages
                teacher_scores[i, start:end] = decoder_scores[i]
            return retriever_scores, teacher_scores

        query_vector = query_vector.float().detach().requires_grad_()
        passage_vector = passage_vector.float().detach().requires_grad_()
        retriever_scores, teacher_scores = compute_retriever_scores(query_vector, passage_vector, decoder_scores)
        retriever_logits = torch.log_softmax(retriever_scores, dim=-1)
        retriever_loss = torch.nn.functional.kl_div(retriever_logits, teacher_scores, reduction='batchmean')

        cs_query_vector = cs_query_vector.float().detach().requires_grad_()
        cs_passage_vector = cs_passage_vector.float().detach().requires_grad_()
        cs_retriever_scores, cs_teacher_scores = compute_retriever_scores(cs_query_vector, cs_passage_vector,
                                                                          cs_decoder_scores)
        cs_retriever_logits = torch.log_softmax(cs_retriever_scores, dim=-1)
        cs_retriever_loss = torch.nn.functional.kl_div(cs_retriever_logits, cs_teacher_scores, reduction='batchmean') \
                            + torch.nn.functional.kl_div(cs_retriever_logits,
                                                         torch.softmax(retriever_scores.detach(), dim=-1),
                                                         reduction='batchmean')
        retriever_loss += cs_retriever_loss

        if self.training_args.query_alignment:
            # # works worse with cross-entropy loss
            # if self.training_args.negatives_x_device:
            #     all_query_vector = self.dist_gather_tensor(query_vector)
            #     all_cs_query_vector = self.dist_gather_tensor(cs_query_vector)
            #     query_scores = torch.matmul(all_query_vector, all_cs_query_vector.transpose(0, 1))
            # else:
            #     query_scores = torch.matmul(query_vector, cs_query_vector.transpose(0, 1))
            # target = torch.arange(query_scores.size()[0], device=self.training_args.device, dtype=torch.long)
            # query_loss = torch.nn.functional.cross_entropy(query_scores, target)

            # todo: mse loss and apply gradient stop on english query vectors (add this is better)
            if self.training_args.negatives_x_device:
                all_query_vector = self.dist_gather_tensor(query_vector)
                all_cs_query_vector = self.dist_gather_tensor(cs_query_vector)
                query_loss = torch.nn.functional.mse_loss(all_cs_query_vector, all_query_vector.detach())
            else:
                query_loss = torch.nn.functional.mse_loss(cs_query_vector, query_vector.detach())

            retriever_loss += query_loss

        if self.training_args.barlow_twins:
            def off_diagonal(x):
                # return a flattened view of the off-diagonal elements of a square matrix
                n, m = x.shape
                assert n == m
                return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

            z1 = torch.cat([x for x, _ in normalised_vector], dim=0)
            z2 = torch.cat([x for _, x in normalised_vector], dim=0)
            z1 = z1.float().detach().requires_grad_()
            z2 = z2.float().detach().requires_grad_()
            if self.training_args.negatives_x_device:
                all_z1 = self.dist_gather_tensor(z1)
                all_z2 = self.dist_gather_tensor(z2)
                c = all_z1.T @ all_z2
            else:
                c = z1 @ z2

            # sum the cross-correlation matrix between all gpus
            c.div_(z1.size()[0])

            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(c).pow_(2).sum()
            bt_loss = on_diag + 0.0051 * off_diag
            retriever_loss += bt_loss

        retriever_loss.backward()

        def vector_forward(input_ids_chunks, attention_mask_chunks, grads_chunk, rnds, is_query=False):
            for idx, (input_ids, attention_mask, grads, rnd) in \
                    enumerate(zip(input_ids_chunks, attention_mask_chunks, grads_chunk, rnds)):
                def chunk_forward():
                    with rnd:
                        with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                            inputs = {
                                "query": {
                                    "input_ids": input_ids,
                                    "attention_mask": attention_mask,
                                } if is_query else None,
                                "passage": {
                                    "input_ids": input_ids,
                                    "attention_mask": attention_mask,
                                } if not is_query else None,
                            }
                            outputs = self.model(
                                **inputs, only_encoding=True, return_dict=True
                            )
                            vector = outputs.query_vector if is_query else outputs.passage_vector
                            surrogate = torch.dot(vector.flatten().float(), grads.flatten())
                            if self.training_args.barlow_twins and is_query:
                                normalised_vector = outputs.normalised_query_vector
                                z1 = torch.cat([x for x, _ in normalised_vector], dim=0)
                                z2 = torch.cat([x for _, x in normalised_vector], dim=0)
                                surrogate += torch.dot(z1.flatten().float(), z1_grads_chunk[idx].flatten()) + \
                                             torch.dot(z2.flatten().float(), z2_grads_chunk[idx].flatten())
                            sum_of_params = sum([p.sum() for p in self.model.parameters()])
                    surrogate = surrogate * self._dist_loss_scale_factor + 0. * sum_of_params
                    if self.use_amp:
                        self.scaler.scale(surrogate).backward()
                    else:
                        surrogate.backward()

                if idx != len(input_ids_chunks) - 1:
                    with self.model.no_sync():
                        chunk_forward()
                else:
                    chunk_forward()

        query_grads_chunk = torch.chunk(torch.cat([query_vector.grad, cs_query_vector.grad], dim=0),
                                        chunks=self.training_args.gc_chunk_size, dim=0)
        if self.training_args.barlow_twins:
            z1_grads_chunk = torch.chunk(z1.grad, chunks=self.training_args.gc_chunk_size, dim=0)
            z2_grads_chunk = torch.chunk(z2.grad, chunks=self.training_args.gc_chunk_size, dim=0)
            assert len(z1_grads_chunk) == len(z2_grads_chunk) == len(query_grads_chunk), \
                (len(z1_grads_chunk), len(z2_grads_chunk), len(query_grads_chunk))

        vector_forward(query_inputs_ids_chunks, query_attention_mask_chunks, query_grads_chunk, query_rnds,
                       is_query=True)

        passage_grads_chunk = torch.chunk(torch.cat([passage_vector.grad, cs_passage_vector.grad], dim=0),
                                          chunks=self.training_args.gc_chunk_size, dim=0)
        vector_forward(passage_inputs_ids_chunks, passage_attention_mask_chunks, passage_grads_chunk, passage_rnds)

        return reader_loss + retriever_loss

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
        if self.training_args.cross_lingual_alignment:
            return self.cross_lingual_alignment_compute_loss(batch, global_step)

        if self.training_args.separate_joint_encoding:
            return self.separate_joint_encoding_compute_loss(batch, global_step)

        if self.training_args.grad_cache:
            return self.grad_cache_compute_loss(batch, global_step)

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
            eval_result = self.separate_joint_refresh_passages(do_eval=True, global_step=global_step)
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
            if self.training_args.separate_joint_encoding:
                self.separate_joint_refresh_passages(do_eval=False)
            else:
                self.refresh_passages(do_eval=True, epoch=0)
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

            # todo: one dataloader for one languages (i.e., queries within a batch should be in the same language)
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
                    # kl_loss = kl_loss / self._dist_loss_scale_factor
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
                    # if self.is_world_process_zero():
                    #     checkpoint_folder = f"checkpoint-{global_step}"
                    #     output_dir = os.path.join(self.training_args.output_dir, checkpoint_folder)
                    #     self._save(self.model, output_dir)

                    if self.training_args.separate_joint_encoding:
                        eval_result = self.separate_joint_refresh_passages(do_eval=True, global_step=global_step)
                    else:
                        eval_result = self.refresh_passages(epoch=self.epoch + 1)
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
                    if self.training_args.separate_joint_encoding:
                        self.separate_joint_refresh_passages(do_eval=False, global_step=global_step)
                    else:
                        self.refresh_passages(epoch=self.epoch + 1)

                if global_step >= self.max_step:
                    break
            if global_step >= self.max_step:
                break

    def test(self):
        torch.distributed.barrier()
        global_step = self.training_args.max_steps
        if self.training_args.eval_on_test:
            # logger.info(f'Evaluating on {self.data_args.task} test')
            # self.model = RRForConditionalGeneration.from_pretrained(
            #     os.path.join(self.training_args.output_dir, f"checkpoint-best"),
            #     config=self.config,
            # )
            # self.model.to(self.training_args.device)
            # if self.data_args.task == "XOR-Retrieve":
            #     self.data_args.eval_query_file = "xor_test_retrieve_eng_span_q_only_v1_1.jsonl"
            # elif self.data_args.task == "XOR-Full":
            #     self.data_args.eval_query_file = "xor_test_full_q_only_v1_1.jsonl"
            # else:
            #     raise NotImplementedError
            # _ = self.separate_joint_refresh_passages(do_eval=True, eval_set="test", global_step=global_step)

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
            _ = self.separate_joint_refresh_passages(do_eval=True, eval_set="mkqa", global_step=global_step)

        # if self.training_args.eval_on_mkqa:

    def refresh_passages(self, do_eval=True, epoch=0):
        self.model.eval()

        torch.cuda.empty_cache()
        output_dir = os.path.join(self.training_args.output_dir, "encoding")
        os.makedirs(output_dir, exist_ok=True)

        if self.is_world_process_zero():
            self.encode(
                is_query=True, query_file=self.data_args.query_file,
                encoded_save_path=os.path.join(output_dir, "train_query_embedding.pt"),
                text_max_length=self.data_args.max_query_length,
            )
        torch.distributed.barrier()

        output_path = os.path.join(output_dir, "embedding_split%.2d.pt" % self.training_args.process_index)
        self.encode(is_query=False, encoded_save_path=output_path, text_max_length=self.data_args.max_passage_length)
        torch.distributed.barrier()
        logger.info(f"Process {self.training_args.process_index} Done encoding")

        eval_result = 0.0
        if self.is_world_process_zero():
            index_files = glob.glob(os.path.join(output_dir, "embedding_split*.pt"))

            p_reps_0, p_lookup_0 = torch.load(index_files[0])
            retriever = FaissIPRetriever(p_reps_0.float().numpy(), True)

            shards = chain([(p_reps_0, p_lookup_0)], map(torch.load, index_files[1:]))
            if len(index_files) > 1:
                shards = tqdm(shards, desc='Loading shards into index', total=len(index_files))
            look_up = []

            all_p_reps = []
            for p_reps, p_lookup in shards:
                all_p_reps.append(p_reps.numpy())
                look_up += p_lookup
            retriever.add(np.concatenate(all_p_reps, axis=0))

            # load train query embeddings and refresh top-100 passages
            q_reps, q_lookup = torch.load(os.path.join(output_dir, "train_query_embedding.pt"))
            q_reps = q_reps.float().numpy()

            def search_queries(retriever, q_reps, p_lookup):
                all_scores, all_indices = retriever.batch_search(q_reps, 100, 5000)

                psg_indices = [[p_lookup[x] for x in q_dd] for q_dd in all_indices]
                return all_scores, psg_indices

            all_scores, psg_indices = search_queries(retriever, q_reps, look_up)

            with open(os.path.join(output_dir, "train.jsonl"), 'w') as f:
                for qid, q_doc_scores, q_doc_indices in zip(q_lookup, all_scores, psg_indices):
                    score_list = [(s, idx) for s, idx in zip(q_doc_scores, q_doc_indices)]
                    score_list = sorted(score_list, key=lambda x: x[0], reverse=True)
                    example = {
                        'qid': qid,
                        'pids': [idx for _, idx in score_list],
                    }
                    f.write(json.dumps(example) + '\n')

            if do_eval:
                # load dev query embeddings and do evaluation
                self.encode(is_query=True, query_file="xor_dev_retrieve_eng_span_v1_1.jsonl",
                            encoded_save_path=os.path.join(output_dir, "dev_query_embedding.pt"),
                            text_max_length=self.data_args.max_query_length)
                q_reps, q_lookup = torch.load(os.path.join(output_dir, "dev_query_embedding.pt"))
                q_reps = q_reps.float().numpy()

                all_scores, psg_indices = search_queries(retriever, q_reps, look_up)
                write_ranking(psg_indices, all_scores, q_lookup,
                              os.path.join(output_dir, "dev_xor_retrieve_results.jsonl"),
                              os.path.join(self.data_args.train_dir, "xor_dev_retrieve_eng_span_v1_1.jsonl"),
                              self.train_dataset[0].corpus, False)

                predictions = json.load(open(os.path.join(output_dir, "dev_xor_retrieve_results.jsonl")))
                input_data = read_jsonlines(
                    os.path.join(self.data_args.train_dir, "xor_dev_retrieve_eng_span_v1_1.jsonl"))
                qid2answers = {item["id"]: item["answers"] for item in input_data}
                eval_results = {}
                for topk in [2, 5]:
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

        torch.distributed.barrier()
        with open(os.path.join(output_dir, "train.jsonl"), 'r') as f:
            examples = [json.loads(jsonline) for jsonline in f]
        self.train_dataloader[0].dataset.examples = examples
        self.model.train()

        return eval_result

    def encode(self, is_query=False, query_file=None, encoded_save_path=None, text_max_length=200):
        if is_query:
            queries = GenericDataLoader(self.data_args.train_dir, corpus_file=self.data_args.corpus_file,
                                        query_file=query_file).load_queries()
            start, end = 0, len(queries)
        else:
            corpus = self.train_dataset[0].corpus
            shard_size = len(corpus) // self.world_size
            start = self.training_args.process_index * shard_size
            end = (self.training_args.process_index + 1) * shard_size \
                if self.training_args.process_index + 1 != self.world_size else len(corpus)
            logger.info(
                f'Process {self.training_args.process_index} => Generate passage embeddings from {start} to {end}')

        encode_dataset = EncodeDataset(queries if is_query else corpus, self.tokenizer,
                                       max_length=text_max_length, is_query=is_query,
                                       start=start, end=end, normalize_text=self.data_args.normalize_text,
                                       lower_case=self.data_args.lower_case,
                                       add_lang_token=self.data_args.add_lang_token,
                                       eval_mode=True, )
        encode_loader = DataLoader(
            encode_dataset,
            batch_size=self.training_args.per_device_eval_batch_size * self.training_args.n_gpu,
            collate_fn=EncodeCollator(
                self.tokenizer,
                max_length=text_max_length
            ),
            shuffle=False,
            drop_last=False,
            num_workers=self.training_args.dataloader_num_workers,
        )
        encoded = []
        lookup_indices = []

        for (batch_ids, batch) in tqdm(encode_loader):
            lookup_indices.extend(batch_ids)
            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                with torch.no_grad():
                    for k, v in batch.items():
                        batch[k] = v.to(self.training_args.device)
                    if self.data_args.encode_is_qry:
                        query_vector = self.model(query=batch.data, only_encoding=True).query_vector
                        encoded.append(query_vector.cpu())
                    else:
                        passage_vector = self.model(passage=batch.data, only_encoding=True).passage_vector
                        encoded.append(passage_vector.cpu())

        encoded = torch.cat(encoded)
        torch.save((encoded, lookup_indices), encoded_save_path)

    def separate_joint_refresh_passages(self, do_eval=True, eval_set="dev", global_step=0):
        if not do_eval and self.training_args.use_mcontriever and global_step < self.training_args.self_retrieve_steps:
            logger.info(f"Process {self.training_args.process_index} Loading updated training passages")
            with open(os.path.join(self.data_args.train_dir, "mss.ICL.train.jsonl"), 'r') as f:
                examples = [json.loads(jsonline) for jsonline in f]
            idx = global_step // self.training_args.refresh_intervals
            num_queries = 64 * self.training_args.refresh_intervals
            examples = examples[idx * num_queries: (idx + 1) * num_queries]
            self.train_dataloader[0].dataset.examples = examples

            return

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

        if not do_eval and self.training_args.retrieve_from_each_lang:
            corpus_dict = group_corpus_by_langs(corpus)
            results = {}
            for lang, lang_corpus in corpus_dict.items():
                logger.info(f"Process {self.training_args.process_index} => Refreshing {lang} passages")
                results[lang] = self.separate_joint_encode(do_eval=do_eval, corpus=lang_corpus, global_step=global_step)
        else:
            results = self.separate_joint_encode(do_eval=do_eval, corpus=corpus, global_step=global_step)
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
                    if self.training_args.retrieve_from_each_lang:
                        for lang in corpus_dict.keys():
                            results = load_results(os.path.join(self.training_args.output_dir,
                                                                f"train_{lang}.split*.jsonl"))
                            output_path = os.path.join(self.training_args.output_dir, f"train_{lang}.jsonl")
                            save_results(results, output_path, add_score=False)
                            for file in glob.glob(os.path.join(self.training_args.output_dir,
                                                               f"train_{lang}.split*.jsonl")):
                                if os.path.exists(file):
                                    os.remove(file)
                                    assert not os.path.exists(file)
                    else:
                        results = load_results(os.path.join(self.training_args.output_dir, "train.split*.jsonl"))
                        output_path = os.path.join(self.training_args.output_dir, "train.jsonl")
                        save_results(results, output_path, add_score=False)

        torch.distributed.barrier()
        if not do_eval:
            logger.info(f"Process {self.training_args.process_index} Loading updated training passages")
            if self.training_args.retrieve_from_each_lang:
                examples = {}
                for lang in corpus_dict.keys():
                    with open(os.path.join(self.training_args.output_dir, f"train_{lang}.jsonl"), 'r') as f:
                        for jsonline in f:
                            example = json.loads(jsonline)
                            qid = example['qid']
                            if qid not in examples:
                                examples[qid] = []
                            examples[qid].extend(example['pids'])
                examples = [{"qid": qid, "pids": pids} for qid, pids in examples.items()]
            else:
                with open(os.path.join(self.training_args.output_dir, "train.jsonl"), 'r') as f:
                    examples = [json.loads(jsonline) for jsonline in f]
            # if self.training_args.add_bias:
            if self.training_args.retrieve_from_each_lang:
                self.compute_bias(corpus, examples)
                torch.distributed.barrier()
                data_path = os.path.join(self.training_args.output_dir, "reader_cxmi_predictions.split*.json")
                prediction_files = sorted(glob.glob(data_path))
                predictions = {}
                for path in prediction_files:
                    with open(path) as f:
                        predictions.update(json.load(f))
                results = {qid: {} for qid in predictions.keys()}
                for qid, prediction in predictions.items():
                    for answer, pid, score in prediction:
                        if answer not in results[qid]:
                            results[qid][answer] = []
                        results[qid][answer].append((pid, score))
                new_examples = {}
                for qid, result in results.items():
                    new_result = {}
                    for answer in result.keys():
                        no_context_score = [score for pid, score in result[answer] if pid is None][0]
                        for pid, score in result[answer]:
                            if pid is None:
                                continue
                            if pid not in new_result:
                                new_result[pid] = []
                            new_result[pid].append(score / no_context_score)
                    pids = {}
                    for pid, scores in new_result.items():
                        score = mean(scores)
                        pids[pid] = -score
                    new_examples[qid] = pids
                for example in examples:
                    qid, pids = example['qid'], example['pids']
                    pid_scores = new_examples[qid]
                    example['pids'] = sorted([(pid, pid_scores[pid]) for pid in pids], key=lambda x: x[1], reverse=True)
                    example['pos_pids'] = [(pid, score) for pid, score in pid_scores.items() if pid not in pids]
                save_results({example['qid']: {pid: score for pid, score in example['pids']} for example in examples},
                             os.path.join(self.training_args.output_dir, "train.jsonl"), add_score=True)
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
            eval_mode=True,
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
                    "add_bias": self.training_args.add_bias,
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

    def compute_bias(self, corpus, examples):
        queries = self.train_dataloader[0].dataset.queries

        shard_size = len(examples) // self.training_args.world_size
        start = self.training_args.process_index * shard_size
        end = (self.training_args.process_index + 1) * shard_size \
            if self.training_args.process_index + 1 != self.training_args.world_size else len(examples)
        examples = examples[start:end]

        eval_dataset = CXMIDataset(
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
            batch_size=150,
            sampler=SequentialSampler(eval_dataset),
            collate_fn=data_collator,
            drop_last=False,
            num_workers=self.training_args.dataloader_num_workers,
        )
        model = self.model
        while hasattr(model, 'module'):
            model = model.module

        train_n_passages = 1
        model.n_passages = train_n_passages
        predictions = {}
        with torch.no_grad():
            for batch in tqdm(eval_dataloader):
                qids, reader_inputs, batch_answer, batch_pids = batch

                reader_inputs = {k: v.to(self.training_args.device) for k, v in reader_inputs.items()}
                outputs = model(**reader_inputs, output_attentions=True, return_dict=True)
                # [bsz * n_passages, seq_len, vocab_size]
                lm_logits = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
                labels = reader_inputs.pop("labels")
                indexs = torch.masked_fill(labels, labels == -100, 0)
                logits = torch.gather(lm_logits, dim=-1, index=indexs[..., None]).squeeze(-1)
                logits = torch.sum(logits * (labels != -100).float(), dim=-1)
                decoder_scores = logits.view(-1, train_n_passages)
                batch_decoder_scores = decoder_scores.cpu().tolist()

                for k, scores in enumerate(batch_decoder_scores):
                    qid = qids[k]
                    answer = batch_answer[k]
                    pid = batch_pids[k][0]
                    if qid not in predictions:
                        predictions[qid] = []
                    predictions[qid].append((answer, pid, scores[0]))

                torch.cuda.empty_cache()

        output_path = os.path.join(self.training_args.output_dir, "reader_cxmi_predictions.split{}.json".
                                   format(self.training_args.process_index))
        with open(output_path, "w") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        model.n_passages = self.data_args.train_n_passages

    def encode_query(self, queries, start, end, do_eval=False):
        encode_dataset = EncodeDataset(queries, self.tokenizer, max_length=self.data_args.max_query_length,
                                       is_query=True, start=start, end=end,
                                       normalize_text=self.data_args.normalize_text,
                                       lower_case=self.data_args.lower_case, separate_joint_encoding=True,
                                       add_lang_token=self.data_args.add_lang_token,
                                       eval_mode=do_eval,
                                       task=self.data_args.task)
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

    def separate_joint_encode(self, corpus, do_eval=True, global_step=0):
        if global_step == 0 and os.path.exists(os.path.join(self.training_args.output_dir, "train.jsonl")):
            if self.data_args.query_file == "xor_nq_train_full.jsonl":
                queries = GenericDataLoader(self.training_args.output_dir, corpus_file=self.data_args.corpus_file,
                                            query_file="train.queries.jsonl").load_queries()
                assert len(queries) == 104486, len(queries)
                self.train_dataloader[0].dataset.queries = queries
            return

        if do_eval:
            if 'nq-dpr' in self.data_args.train_dir or self.data_args.task not in self.data_args.train_dir:
                train_dir = 'data/XOR-Retrieve' if self.data_args.task == "XOR-Retrieve" else 'data/XOR-Full'
            else:
                train_dir = self.data_args.train_dir
            queries = GenericDataLoader(train_dir, corpus_file=self.data_args.corpus_file,
                                        query_file=self.data_args.eval_query_file).load_queries()

            if self.data_args.query_file == "xor_train_full_il.jsonl":
                new_queries = {}
                for qid in queries.keys():
                    query, answers, langs = queries[qid]
                    new_queries[qid] = (query[5:].strip(), answers, langs)
                queries = new_queries
        else:
            queries = GenericDataLoader(self.data_args.train_dir, corpus_file=self.data_args.corpus_file,
                                        query_file=self.data_args.query_file).load_queries()
            if ('mss' in self.data_args.query_file or 'wikidata' in self.data_args.query_file
                    or 'fs-qa' in self.data_args.query_file) and len(queries) >= 200000:
                idx = global_step // self.training_args.refresh_intervals
                num_queries = 64 * self.training_args.refresh_intervals
                queries = dict(list(queries.items())[idx * num_queries: (idx + 1) * num_queries])
            # new_queries = {}
            # for qid in queries.keys():
            #     query, answers, lang = queries[qid]
            #     new_queries[qid] = (query + " ".join(answers), answers, lang)
            # queries = new_queries
            if self.data_args.query_file == "xor_nq_train_full.jsonl":
                if self.is_world_process_zero():
                    new_queries = {}
                    for qid in queries.keys():
                        query, answers, langs = queries[qid]
                        if 'nq' in qid:
                            idx = random.choice(range(len(answers)))
                            answer = answers[idx]
                            lang = langs[idx]
                            new_queries[qid] = (f"[{lang}] " + query, [answer], lang)
                        else:
                            new_queries[qid] = (query, answers, langs)
                    with open(os.path.join(self.training_args.output_dir, "train.queries.jsonl"), 'w') as f:
                        for qid, (query, answer, lang) in new_queries.items():
                            f.write(json.dumps({"id": qid, "question": query, "answers": answer, "lang": lang}) + '\n')
                torch.distributed.barrier()
                queries = GenericDataLoader(self.training_args.output_dir, corpus_file=self.data_args.corpus_file,
                                            query_file="train.queries.jsonl").load_queries()
                assert len(queries) == 104486, len(queries)
                self.train_dataloader[0].dataset.queries = queries

            # if global_step == 0 and self.data_args.query_file == "xor_train_full.jsonl" and \
            #         any(["[en]" in query for query, _, _ in list(queries.values())]):
            #     new_queries = {}
            #     for qid in queries.keys():
            #         query, answers, langs = queries[qid]
            #         new_queries[qid] = (query[5:].strip(), answers, langs)
            #     queries = new_queries

        if self.training_args.debug:
            queries = dict(list(queries.items())[:100])
        start, end = 0, len(queries)
        query_vector, q_mask, q_lookup_indices = self.encode_query(queries, start, end, do_eval=do_eval)

        results = {qid: {} for qid in q_lookup_indices}

        # if 'nq-dpr' in self.data_args.train_dir and do_eval:
        #     train_dir = 'data/XOR-Retrieve' if self.data_args.task == "XOR-Retrieve" else 'data/XOR-Full'
        #     corpus_file = 'psgs_w100.tsv' if self.data_args.task == "XOR-Retrieve" else 'all_w100.tsv'
        #     corpus = GenericDataLoader(train_dir, corpus_file=corpus_file,
        #                                query_file=self.data_args.query_file).load_corpus()
        # else:
        #     corpus = self.train_dataset[0].corpus

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
                                       add_lang_token=self.data_args.add_lang_token,
                                       eval_mode=do_eval,
                                       task=self.data_args.task)
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
                    # if batch_scores.size()[0] > 60000 and len(results[qid]) > 100:
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
            # scores = torch.sum(scores, dim=-1) / query_mask.sum(dim=1)[..., None]
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
                # # todo: YONO style
                # retriever_scores = torch.matmul(all_query_vector, all_passage_vector.transpose(0, 1)) / \
                #                    math.sqrt(all_query_vector.size()[-1])
        else:
            if len(query_vector.size()) == 3:
                retriever_scores = compute_colbert_scores(query_vector, passage_vector, query_mask, passage_mask)
                print(query_vector.size(), passage_vector.size(), retriever_scores)
            else:
                retriever_scores = torch.matmul(query_vector, passage_vector.transpose(0, 1))
                # # todo: YONO style
                # retriever_scores = torch.sigmoid(torch.matmul(query_vector, passage_vector.transpose(0, 1)) /
                #                                  math.sqrt(query_vector.size()[-1]))
                # print(query_vector.size(), passage_vector.size(), math.sqrt(query_vector.size()[-1]), retriever_scores,
                #       torch.matmul(query_vector, passage_vector.transpose(0, 1)))
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

        if self.training_args.query_alignment:
            if len(query_vector.size()) == 3:
                pooled_query_vector = \
                    torch.masked_fill(query_vector, ~query_mask[..., None].bool(), 0.0).sum(dim=1) / \
                    query_mask.sum(dim=1)[..., None]
                pooled_cs_query_vector = \
                    torch.masked_fill(cs_query_vector, ~cs_query_mask[..., None].bool(), 0.0).sum(dim=1) / \
                    cs_query_mask.sum(dim=1)[..., None]
            else:
                pooled_query_vector = query_vector
                pooled_cs_query_vector = cs_query_vector
            if self.training_args.negatives_x_device:
                all_query_vector = self.dist_gather_tensor(pooled_query_vector)
                all_cs_query_vector = self.dist_gather_tensor(pooled_cs_query_vector)
                query_loss = torch.nn.functional.mse_loss(all_cs_query_vector, all_query_vector.detach())
            else:
                query_loss = torch.nn.functional.mse_loss(pooled_cs_query_vector, pooled_query_vector.detach())

            retriever_loss += query_loss

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

    def target_language_compute_loss(self, inputs, global_step=None):
        teacher_inputs, student_inputs = inputs

        teacher_retriever_scores, teacher_decoder_scores = self._forward(self.teacher_model, teacher_inputs)

        query_vector, passage_vector, rnds, query_mask, passage_mask = \
            self._forward(self.model, student_inputs, only_encoding=True, return_rnds=True)

        query_vector = query_vector.float().detach().requires_grad_()
        passage_vector = passage_vector.float().detach().requires_grad_()
        retriever_scores = self.compute_scores(query_vector, passage_vector, query_mask, passage_mask)

        teacher_retriever_logits = torch.softmax(teacher_retriever_scores, dim=-1)
        retriever_logits = torch.log_softmax(retriever_scores, dim=-1)
        retriever_loss = torch.nn.functional.kl_div(retriever_logits, teacher_retriever_logits,
                                                    reduction='batchmean')

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
        query_grads_chunk = torch.chunk(query_vector.grad, chunks=self.training_args.gc_chunk_size, dim=0)
        passage_grads_chunk = torch.chunk(passage_vector.grad,
                                          chunks=self.training_args.gc_chunk_size, dim=0)
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

    def no_cross_lingual_alignment_compute_loss(self, inputs, global_step=None):
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
                            query_vector.view(bsz, self.data_args.train_n_passages, -1).mean(1)
                        surrogate = torch.dot(query_vector.flatten().float(), query_grads.flatten()) + \
                                    torch.dot(passage_vector.flatten().float(), passage_grads.flatten())

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
        if self.training_args.cross_lingual_alignment:
            return self.compute_loss(batch, global_step)
        else:
            if self.training_args.parallel_queries:
                return self.no_cross_lingual_alignment_compute_loss(batch, global_step)
            else:
                # assert not self.training_args.parallel_queries, \
                #     "parallel queries are only supported for cross lingual alignment"
                return self.target_language_compute_loss(batch, global_step)


class ReaderWikidataTrainer(ReaderTrainer):
    def __init__(self, model, train_dataset, data_collator, training_args, data_args, tokenizer):
        super(ReaderWikidataTrainer, self).__init__(model, train_dataset, data_collator, training_args, data_args,
                                                    tokenizer)

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    def compute_loss(self, inputs, global_step=None):
        _, reader_inputs, _, _ = inputs

        input_ids, attention_mask, independent_mask, query_mask, passage_mask, labels = \
            reader_inputs['input_ids'], reader_inputs['attention_mask'], reader_inputs['independent_mask'], \
            reader_inputs['query_mask'], reader_inputs['passage_mask'], reader_inputs['labels']

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
        reader_loss = outputs.loss
        query_vector, passage_vector = outputs.query_vector, outputs.passage_vector

        bsz = query_vector.size()[0] // self.data_args.train_n_passages
        query_vector = query_vector.view(bsz, self.data_args.train_n_passages, -1).mean(1)

        if self.training_args.negatives_x_device:
            all_query_vector = self.dist_gather_tensor(query_vector)
            all_passage_vector = self.dist_gather_tensor(passage_vector)
            retriever_scores = torch.matmul(all_query_vector, all_passage_vector.transpose(0, 1))
        else:
            retriever_scores = torch.matmul(query_vector, passage_vector.transpose(0, 1))

        target = torch.arange(
            retriever_scores.size(0),
            device=retriever_scores.device,
            dtype=torch.long
        )

        retriever_loss = self.cross_entropy(retriever_scores, target)

        if self.training_args.negatives_x_device:
            retriever_loss = retriever_loss * self.world_size
        retriever_loss = retriever_loss * self.training_args.retriever_weight

        loss = reader_loss + retriever_loss
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return reader_loss + retriever_loss / self._dist_loss_scale_factor

    def e2e_compute_loss(self, inputs, global_step=None):
        _, reader_inputs, _, _ = inputs

        inputs_ids, attention_mask, independent_mask, query_mask, passage_mask, labels = \
            reader_inputs['input_ids'], reader_inputs['attention_mask'], reader_inputs['independent_mask'], \
            reader_inputs['query_mask'], reader_inputs['passage_mask'], reader_inputs['labels']
        self.training_args.gc_chunk_size = labels.size()[0]
        input_ids_chunks = torch.chunk(inputs_ids, chunks=self.training_args.gc_chunk_size, dim=0)
        attention_mask_chunks = torch.chunk(attention_mask, chunks=self.training_args.gc_chunk_size, dim=0)
        independent_mask_chunks = torch.chunk(independent_mask, chunks=self.training_args.gc_chunk_size, dim=0)
        labels_chunks = torch.chunk(labels, chunks=self.training_args.gc_chunk_size, dim=0)

        if global_step < self.training_args.distillation_start_steps or self.training_args.only_reader:
            reader_loss = 0
            for idx, (input_ids, attention_mask, independent_mask, labels) in \
                    enumerate(zip(input_ids_chunks, attention_mask_chunks, independent_mask_chunks, labels_chunks)):
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

        reader_loss = 0
        for idx, (input_ids, attention_mask, independent_mask, query_mask, passage_mask, labels) in \
                enumerate(zip(input_ids_chunks, attention_mask_chunks, independent_mask_chunks, query_mask_chunks,
                              passage_mask_chunks, labels_chunks)):
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
                    bsz = query_vector.size()[0] // self.data_args.train_n_passages
                    query_vector = query_vector.view(bsz, self.data_args.train_n_passages, -1).mean(1)

                    cross_attention_scores = outputs.cross_attentions[-1][:, :, 0]
                    bsz, n_heads, _ = cross_attention_scores.size()
                    scores = cross_attention_scores.view(bsz, n_heads, self.data_args.train_n_passages, -1)
                    teacher_scores = scores.sum(dim=-1).mean(dim=1).detach()

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

    def train_step(self, batch, global_step=None):
        if self.training_args.e2e_training:
            return self.e2e_compute_loss(batch, global_step)
        return self.compute_loss(batch, global_step)
