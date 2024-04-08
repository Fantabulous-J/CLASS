import json
import logging
import os
import sys
from statistics import mean

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

from torch.utils.data import DataLoader, SequentialSampler

from arguments import DataArguments, DistilModelArguments, BiEncoderTrainingArguments
from dataloader import ReaderDataset, CXMIDataset, ReaderCollator, GenericDataLoader
from model import T5ForConditionalGeneration

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((DistilModelArguments, DataArguments, BiEncoderTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: DistilModelArguments
        data_args: DataArguments
        training_args: BiEncoderTrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)
    logger.info("DATA parameters %s", data_args)

    set_seed(training_args.seed)

    if training_args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False
    )
    config.n_passages = data_args.train_n_passages
    if not training_args.de_avg_pooling:
        config.retriever_head = 6
    if training_args.separate_joint_encoding:
        from model import RRForConditionalGeneration
        model = RRForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir
        )
        # model.encoder.copy_relative_attention_bias()
        # model.decoder.copy_relative_attention_bias()
    else:
        model = T5ForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir
        )

    model.to(training_args.device)
    model.eval()

    data_dir = data_args.train_dir
    if os.path.exists(data_args.train_path):
        train_path = data_args.train_path
    else:
        train_path = os.path.join(data_dir, data_args.train_path)

    corpus = GenericDataLoader(data_dir, corpus_file=data_args.corpus_file).load_corpus()
    queries = GenericDataLoader(data_dir, query_file=data_args.query_file).load_queries()

    if os.path.isdir(data_args.output_path):
        if data_args.train_path != data_args.query_file:
            if ".cl" in data_args.train_path:
                output_path = os.path.join(data_args.output_path, "retrieval.filtered.cl_predictions.json")
            elif ".il" in data_args.train_path:
                output_path = os.path.join(data_args.output_path, "retrieval.filtered.il_predictions.json")
            elif ".yes_no" in data_args.train_path:
                output_path = os.path.join(data_args.output_path, "retrieval.filtered.yes_no_predictions.json")
            else:
                raise ValueError("Invalid train_path")
            if data_args.encode_num_shard > 1:
                output_path = output_path.replace(".json", f".split{data_args.encode_shard_index}.json")
        else:
            if "yes_no" in data_args.train_path:
                output_path = os.path.join(data_args.output_path, "yes_no_predictions.json")
            else:
                output_path = os.path.join(data_args.output_path, "predictions.json")
    else:
        output_path = os.path.join(model_args.model_name_or_path, data_args.output_path)

    eval_dataset = ReaderDataset(
        queries=queries,
        corpus=corpus,
        tokenizer=tokenizer,
        train_path=train_path,
        data_args=data_args,
        eval_mode=False,
        # answer_in_en=".cl" in data_args.train_path,
        answer_in_en=True
    )

    if data_args.encode_num_shard > 1:
        examples = eval_dataset.examples
        shard_size = len(examples) // data_args.encode_num_shard
        start = data_args.encode_shard_index * shard_size
        end = start + shard_size if data_args.encode_shard_index < data_args.encode_num_shard - 1 else len(examples)
        eval_dataset.examples = examples[start:end]

    predictions = {}
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            predictions = json.load(f)
        new_examples = []
        for example in eval_dataset.examples:
            if example["qid"] not in predictions:
                new_examples.append(example)
        eval_dataset.examples = new_examples

    data_collator = ReaderCollator(
        tokenizer,
        max_query_length=data_args.max_query_length,
        max_passage_length=data_args.max_passage_length,
        max_query_passage_length=data_args.max_query_passage_length,
        max_answer_length=data_args.max_answer_length,
        separate_joint_encoding=training_args.separate_joint_encoding,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=training_args.eval_batch_size,
        sampler=SequentialSampler(eval_dataset),
        collate_fn=data_collator,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )

    autocast_dtype = torch.bfloat16 if training_args.bf16 else torch.float16 if training_args.fp16 else torch.float32

    if "yes_no" in data_args.train_path and "fs_qa.retrieval.yes_no.filtered.pids.jsonl" not in data_args.train_path:
        logger.info("Choosing from Yes/No predictions")
        with tokenizer.as_target_tokenizer():
            encoded_answer = tokenizer.batch_encode_plus(
                ["yes"] * training_args.eval_batch_size,
                max_length=data_args.max_answer_length,
                truncation=True,
                padding=True,
                return_tensors='pt'
            )
            labels = encoded_answer['input_ids']
            answer_mask = encoded_answer["attention_mask"].bool()
            yes_labels = labels.masked_fill(~answer_mask, -100)

            encoded_answer = tokenizer.batch_encode_plus(
                ["no"] * training_args.eval_batch_size,
                max_length=data_args.max_answer_length,
                truncation=True,
                padding=True,
                return_tensors='pt'
            )
            labels = encoded_answer['input_ids']
            answer_mask = encoded_answer["attention_mask"].bool()
            no_labels = labels.masked_fill(~answer_mask, -100)

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            if training_args.separate_joint_encoding:
                qids, reader_inputs, batch_answer, batch_pids = batch
            else:
                qids, encoded_query, encoded_passage, reader_inputs, batch_answer, batch_pids = batch

            if "yes_no" in data_args.train_path and "fs_qa.retrieval.yes_no.filtered.pids.jsonl" not in data_args.train_path:
                reader_inputs['labels'] = yes_labels[:reader_inputs["input_ids"].size(0)]
                reader_inputs = {k: v.to(training_args.device) for k, v in reader_inputs.items()}
                with torch.cuda.amp.autocast(enabled=training_args.fp16 or training_args.bf16, dtype=autocast_dtype):
                    with torch.no_grad():
                        outputs = model(**reader_inputs, return_dict=True)
                    lm_logits = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
                    labels = reader_inputs.pop("labels")
                    indexs = torch.masked_fill(labels, labels == -100, 0)
                    logits = torch.gather(lm_logits, dim=-1, index=indexs[..., None]).squeeze(-1)
                    yes_logits = torch.sum(logits * (labels != -100).float(), dim=-1).cpu().tolist()

                reader_inputs['labels'] = no_labels[:reader_inputs["input_ids"].size(0)]
                reader_inputs = {k: v.to(training_args.device) for k, v in reader_inputs.items()}
                with torch.cuda.amp.autocast(enabled=training_args.fp16 or training_args.bf16, dtype=autocast_dtype):
                    with torch.no_grad():
                        outputs = model(**reader_inputs, return_dict=True)
                    lm_logits = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
                    labels = reader_inputs.pop("labels")
                    indexs = torch.masked_fill(labels, labels == -100, 0)
                    logits = torch.gather(lm_logits, dim=-1, index=indexs[..., None]).squeeze(-1)
                    no_logits = torch.sum(logits * (labels != -100).float(), dim=-1).cpu().tolist()

                for k, (yes_logit, no_logit) in enumerate(zip(yes_logits, no_logits)):
                    if yes_logit > no_logit:
                        predictions[qids[k]] = "yes"
                    else:
                        predictions[qids[k]] = "no"
            else:
                bsz, seq_len = reader_inputs["input_ids"].size()
                input_ids = reader_inputs["input_ids"].view(bsz // data_args.train_n_passages,
                                                            data_args.train_n_passages, seq_len)
                if training_args.separate_joint_encoding:
                    model_inputs = {
                        "input_ids": input_ids.to(training_args.device),
                        "attention_mask": reader_inputs["attention_mask"].to(training_args.device),
                        "independent_mask": reader_inputs["independent_mask"].to(training_args.device),
                    }
                else:
                    model_inputs = {
                        "input_ids": input_ids.to(training_args.device),
                        "attention_mask": reader_inputs["attention_mask"].to(training_args.device)
                    }

                with torch.cuda.amp.autocast(enabled=training_args.fp16 or training_args.bf16, dtype=autocast_dtype):
                    outputs = model.generate(
                        **model_inputs,
                        max_length=data_args.max_answer_length,
                        num_beams=1,
                    )
                for k, o in enumerate(outputs):
                    ans = tokenizer.decode(o, skip_special_tokens=True)
                    predictions[qids[k]] = ans

                if len(predictions) % 1000 == 0:
                    with open(output_path, "w") as f:
                        json.dump(predictions, f, ensure_ascii=False, indent=2)

            # with torch.cuda.amp.autocast(enabled=training_args.fp16 or training_args.bf16, dtype=autocast_dtype):
            #     labels = reader_inputs.pop("labels")
            #     decoder_input_ids = torch.ones(labels.size(0), 1, dtype=torch.long) * config.decoder_start_token_id
            #     reader_inputs["decoder_input_ids"] = decoder_input_ids
            #     reader_inputs = {k: v.to(training_args.device) for k, v in reader_inputs.items()}
            #
            #     # first output token
            #     outputs = model(**reader_inputs, output_attentions=True, return_dict=True)
            #     cross_attention_scores = outputs.cross_attentions[-1][:, :, 0]
            #     bsz, n_heads, _ = cross_attention_scores.size()
            #     decoder_scores = cross_attention_scores.view(bsz, n_heads, data_args.train_n_passages, -1)
            #     decoder_scores = decoder_scores.sum(dim=-1).mean(dim=1).cpu().tolist()
            #     print(list(zip(batch_pids[0], decoder_scores[0])))
            #
            #     # # average all output tokens
            #     # reader_inputs = {k: v.to(training_args.device) for k, v in reader_inputs.items()}
            #     # outputs = model(**reader_inputs, output_attentions=True, return_dict=True)
            #     # cross_attention_scores = outputs.cross_attentions[-1]
            #     # bsz, n_heads, n_tokens, _ = cross_attention_scores.size()
            #     # decoder_scores = cross_attention_scores.view(bsz, n_heads, n_tokens, data_args.train_n_passages, -1)
            #     # label_mask = (reader_inputs['labels'] != -100).float()
            #     # decoder_scores = decoder_scores.sum(dim=-1).mean(dim=1) * label_mask[..., None]
            #     # decoder_scores = decoder_scores.sum(dim=1) / label_mask.sum(dim=-1)[..., None]
            #
            #     # # likelihood of generating the gold answer as reranking scores
            #     # labels = reader_inputs.pop("labels")
            #     # # labels = labels.unsqueeze(1).expand([-1, data_args.train_n_passages, -1]).reshape(-1, labels.size(-1))
            #     # labels = torch.repeat_interleave(labels, data_args.train_n_passages, dim=0)
            #     # reader_inputs["labels"] = labels
            #     # reader_inputs = {k: v.to(training_args.device) for k, v in reader_inputs.items()}
            #     # outputs = model(**reader_inputs, output_attentions=True, return_dict=True)
            #     # # [bsz * n_passages, seq_len, vocab_size]
            #     # lm_logits = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
            #     # labels = reader_inputs.pop("labels")
            #     # indexs = torch.masked_fill(labels, labels == -100, 0)
            #     # logits = torch.gather(lm_logits, dim=-1, index=indexs[..., None]).squeeze(-1)
            #     # logits = torch.sum(logits * (labels != -100).float(), dim=-1)
            #     # decoder_scores = logits.view(-1, data_args.train_n_passages)
            #     # # print(torch.softmax(decoder_scores, dim=-1))
            #     # decoder_scores = decoder_scores.cpu().tolist()
            #
            #     # outputs = model(**reader_inputs, output_attentions=True, return_dict=True, only_encoding=True)
            #     # query_vector, passage_vector = outputs.query_vector, outputs.passage_vector
            #     # decoder_scores = torch.sum(query_vector * passage_vector, dim=-1)
            #     # decoder_scores = decoder_scores.view(-1, data_args.train_n_passages).cpu().tolist()
            #
            #     # outputs = model(**reader_inputs, output_attentions=True, return_dict=True, only_encoding=True)
            #     # # (batch_size, seq_length, key_length)
            #     # query_vector, passage_vector = outputs.query_vector, outputs.passage_vector
            #     # query_mask = reader_inputs["query_mask"]
            #     # passage_mask = reader_inputs["passage_mask"]
            #     # attn_weights = torch.matmul(query_vector, passage_vector.transpose(-1, -2))
            #     # attn_weights = attn_weights.masked_fill(~passage_mask[:, None].bool(), -1e9)
            #     # attn_weights = torch.max(attn_weights, dim=-1).values
            #     # attn_weights = attn_weights.masked_fill(~query_mask.bool(), 0.0)
            #     # attn_weights = torch.sum(attn_weights, dim=-1) / query_mask.sum(dim=1)
            #     # # attn_weights = torch.sum(attn_weights, dim=-1)
            #     # decoder_scores = attn_weights.view(-1, data_args.train_n_passages).cpu().tolist()
            #
            # for qid, pids, scores in zip(qids, batch_pids, decoder_scores):
            #     pid_scores = zip(pids, scores)
            #     pid_scores = sorted(pid_scores, key=lambda x: x[1], reverse=True)
            #     ctxs = []
            #     for k, (pid, score) in enumerate(pid_scores):
            #         ctxs.append(corpus[pid]["text"])
            #     reader_ranks.append({"id": qid, "lang": queries[qid][-1], "ctxs": ctxs})
            #
            # # for qid, pids, scores in zip(qids, batch_pids, decoder_scores):
            # #     pid_scores = zip(pids, scores)
            # #     pid_scores = sorted(pid_scores, key=lambda x: x[1], reverse=True)
            # #     for k, (pid, score) in enumerate(pid_scores):
            # #         reader_ranks.append((qid, pid, k + 1, score))

            torch.cuda.empty_cache()

    # with open(os.path.join(model_args.model_name_or_path, "dev_xor_reader_rerank_results.json"), "w") as f:
    #     json.dump(reader_ranks, f)

    # with open(os.path.join(model_args.model_name_or_path, "reader.rank.txt.trec"), "w") as f:
    #     for qid, pid, rank, score in reader_ranks:
    #         f.write(f'{qid} Q0 {pid} {rank} {score} dense\n')

    with open(output_path, "w") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    # train_n_passages = 1
    # model.n_passages = train_n_passages
    #
    # predictions = {}
    # with torch.no_grad():
    #     for batch in tqdm(eval_dataloader):
    #         qids, reader_inputs, batch_answer, batch_pids = batch
    #
    #         reader_inputs = {k: v.to(training_args.device) for k, v in reader_inputs.items()}
    #         outputs = model(**reader_inputs, output_attentions=True, return_dict=True)
    #         # [bsz * n_passages, seq_len, vocab_size]
    #         lm_logits = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
    #         labels = reader_inputs.pop("labels")
    #         indexs = torch.masked_fill(labels, labels == -100, 0)
    #         logits = torch.gather(lm_logits, dim=-1, index=indexs[..., None]).squeeze(-1)
    #         logits = torch.sum(logits * (labels != -100).float(), dim=-1)
    #         decoder_scores = logits.view(-1, train_n_passages)
    #         batch_decoder_scores = decoder_scores.cpu().tolist()
    #
    #         for k, scores in enumerate(batch_decoder_scores):
    #             qid = qids[k]
    #             answer = batch_answer[k]
    #             pid = batch_pids[k][0]
    #             if qid not in predictions:
    #                 predictions[qid] = []
    #             predictions[qid].append((answer, pid, scores[0]))
    #
    #         torch.cuda.empty_cache()
    #
    # results = {qid: {} for qid in predictions.keys()}
    # for qid, prediction in predictions.items():
    #     for answer, pid, score in prediction:
    #         if answer not in results[qid]:
    #             results[qid][answer] = []
    #         results[qid][answer].append((pid, score))
    # new_examples = {}
    # for qid, result in results.items():
    #     new_result = {}
    #     for answer in result.keys():
    #         no_context_score = [score for pid, score in result[answer] if pid is None][0]
    #         for pid, score in result[answer]:
    #             if pid is None:
    #                 continue
    #             if pid not in new_result:
    #                 new_result[pid] = []
    #             new_result[pid].append(score / no_context_score)
    #     pids = {}
    #     for pid, scores in new_result.items():
    #         score = mean(scores)
    #         pids[pid] = score
    #     new_examples[qid] = pids
    # reader_ranks = []
    # for qid, pids in new_examples.items():
    #     pid_scores = sorted(pids.items(), key=lambda x: x[1], reverse=False)
    #     assert len(pid_scores) == 100, len(pid_scores)
    #     ctxs = []
    #     for k, (pid, score) in enumerate(pid_scores):
    #         ctxs.append(corpus[pid]["text"])
    #     reader_ranks.append({"id": qid, "lang": queries[qid][-1], "ctxs": ctxs})
    #
    # with open(os.path.join(model_args.model_name_or_path, "dev_xor_reader_rerank_results.json"), "w") as f:
    #     json.dump(reader_ranks, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
