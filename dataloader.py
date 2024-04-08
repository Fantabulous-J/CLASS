import csv
import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Dict, Union, List

import torch
from torch.utils.data import Dataset
from tqdm.autonotebook import tqdm
from transformers import PreTrainedTokenizer, DataCollatorWithPadding, MT5Tokenizer, MT5TokenizerFast, T5Tokenizer, \
    T5TokenizerFast

import normalize_text
from arguments import DataArguments

logging.getLogger('transformers.tokenization_utils').setLevel(logging.ERROR)
logging.getLogger('transformers.tokenization_utils_base').setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


class GenericDataLoader:

    def __init__(self, data_folder: str = None, corpus_file: str = "corpus.tsv", query_file: str = "train.query.txt",
                 qrel_file: str = "train-hard-negatives.jsonl"):
        self.corpus = {}
        self.queries = {}
        self.qrels = []

        self.corpus_file = os.path.join(data_folder, corpus_file) if data_folder else corpus_file
        self.query_file = os.path.join(data_folder, query_file) if data_folder else query_file
        self.qrel_file = os.path.join(data_folder, qrel_file) if data_folder else qrel_file

    @staticmethod
    def check(fIn: str, ext: str):
        if not os.path.exists(fIn):
            raise ValueError("File {} not present! Please provide accurate file.".format(fIn))

        if not fIn.endswith(ext):
            raise ValueError("File {} must be present with extension {}".format(fIn, ext))

    def load(self, split="test"):

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d %s Documents.", len(self.corpus), split.upper())
            logger.info("Doc Example: %s",
                        list(self.corpus.values())[0] if isinstance(self.corpus, dict) else self.corpus['0'])

        if not len(self.queries):
            self.load_queries()

        if not len(self.qrels):
            self.load_qrels()

        return self.corpus, self.queries, self.qrels

    def load_corpus(self) -> Dict[str, Dict[str, str]]:

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s",
                        list(self.corpus.values())[0] if isinstance(self.corpus, dict) else self.corpus['0'])

        return self.corpus

    def load_queries(self) -> Dict[str, str]:
        logger.info("Loading Queries...")
        self.queries = self._load_queries(self.query_file)
        logger.info("Loaded %d Queries.", len(self.queries))
        logger.info("Query Example: %s", list(self.queries.values())[0])

        return self.queries

    def load_qrels(self):
        def _load_qrels(qrel_file):
            qrels = []
            with open(qrel_file, 'r') as f:
                for jsonline in tqdm(f.readlines()):
                    example = json.loads(jsonline)
                    qrels.append(example)
            return qrels

        self.qrels = _load_qrels(self.qrel_file)
        logger.info("Loaded %d Queries from %s.", len(self.qrels), self.qrel_file)

    def _load_corpus(self):
        if self.corpus_file.endswith('jsonl'):
            with open(self.corpus_file, encoding='utf-8') as fIn:
                for jsonline in tqdm(fIn):
                    example = json.loads(jsonline)
                    self.corpus[example['pid']] = jsonline
        else:
            self.check(fIn=self.corpus_file, ext="tsv")
            normalize = "all_w100.tsv" in self.corpus_file
            if normalize:
                import unicodedata
            with open(self.corpus_file, encoding='utf-8') as fIn:
                reader = csv.reader(fIn, delimiter="\t")
                for row in tqdm(reader):
                    if not row[0] == "id":
                        self.corpus[row[0]] = {
                            "title": row[2] if not normalize else unicodedata.normalize('NFC', row[2]),
                            "text": row[1]
                        }

    def _load_queries(self, query_file):
        queries = {}
        if query_file.endswith('jsonl'):
            with open(query_file, encoding='utf-8') as fIn:
                for jsonline in fIn.readlines():
                    example = json.loads(jsonline)
                    lang = example['lang'].strip() if isinstance(example['lang'], str) else example['lang']
                    if 'answers' in example:
                        queries[example['id']] = (example['question'], example['answers'], lang)
                    else:
                        queries[example['id']] = (example['question'], ['placeholder'], lang)
        elif query_file.endswith('csv'):
            with open(query_file, 'r') as fIn:
                reader = csv.reader(fIn, delimiter='\t')
                for idx, row in enumerate(reader):
                    query = row[0]
                    answers = eval(row[1])
                    if len(row) == 3:
                        queries[str(idx)] = (query, answers, row[2])
                    else:
                        queries[str(idx)] = (query, answers, 'en')
        else:
            with open(query_file, encoding='utf-8') as fIn:
                for line in fIn:
                    try:
                        query_id, text, trans_text, answer = line.strip().split('\t')
                        lang = 'en' if 'parallel' in query_file else query_id.split("-")[0]
                        queries[query_id] = ([text, trans_text], [answer], lang)
                    except ValueError:
                        try:
                            query_id, text, answer = line.strip().split('\t')
                            lang = 'en'
                            if query_id.split("-")[0] in ['ar', 'bn', 'de', 'es', 'fi', 'fr', 'it', 'ja', 'ko',
                                                          'ru', 'te', 'ta', 'ml', 'kn', 'zh']:
                                lang = query_id.split("-")[0]
                            queries[query_id] = (text, [answer], lang)
                        except ValueError:
                            query_id, text = line.strip().split('\t')
                            queries[query_id] = text

        return queries


class ReaderDataset(Dataset):
    def __init__(self,
                 queries,
                 corpus,
                 tokenizer: PreTrainedTokenizer,
                 train_path: Union[str, List],
                 data_args: DataArguments,):
        super(ReaderDataset, self).__init__()

        self.queries = queries
        self.corpus = corpus

        self.tokenizer = tokenizer
        self.data_args = data_args

        if isinstance(train_path, str):
            if train_path.endswith(".jsonl"):
                with open(train_path) as f:
                    self.examples = [json.loads(jsonline) for jsonline in f.readlines()]

            elif train_path.endswith('.json'):
                with open(train_path) as f:
                    self.examples = json.load(f)
            elif train_path.endswith('.csv'):
                with open(train_path, 'r') as f:
                    reader = csv.reader(f, delimiter='\t')
                    self.examples = [row for row in reader]
            else:
                logger.info("generate placeholder examples")
                self.examples = [{} for _ in range(64000)]
        else:
            assert isinstance(train_path, List), type(train_path)
            self.examples = train_path

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        if 'ctxs' in example:
            query, answers, ctxs = example['question'], example['answers'], example['ctxs']
            query = f"question: {query}"
            answer = random.choice(answers)
            passages = []
            for ctx in ctxs[:self.data_args.train_n_passages]:
                title, text = ctx['title'], ctx['text']
                passages.append("title: " + title + " context: " + text)

            return idx, query, passages, answer, [ctx['id'] for ctx in ctxs]
        else:
            qid, pids = example['qid'], example['pids']
            assert len(pids) >= self.data_args.train_n_passages, len(pids)

            query, answers, lang = self.queries[qid]

            if len(self.examples) == 64000 and isinstance(query, list):
                assert len(query) == 2, query
                query = query[1]

            if isinstance(query, list):
                if len(query) == 2:
                    query, trans_query = query
                    if answers[0] in trans_query:
                        trans_query = trans_query.replace(answers[0], '_X_')
                else:
                    query, *trans_query = query
                    trans_query = random.choice(trans_query)
                trans_query = normalize_text.normalize(trans_query)
                query = f"question: {query}"
                trans_query = f"question: {trans_query}"
                query = (query, trans_query)
            else:
                if lang != 'en':
                    query = normalize_text.normalize(query)
                query = f"question: {query}"

            answer = random.choice(answers)

            passages = []
            for pid in pids[:self.data_args.train_n_passages]:
                title, text = self.corpus[pid]['title'], self.corpus[pid]['text']
                passages.append("title: " + title + " context: " + text)

            return qid, query, passages, answer, pids[:self.data_args.train_n_passages]


class EncodeDataset(Dataset):
    def __init__(self,
                 data: Dict,
                 tokenizer: PreTrainedTokenizer,
                 max_length: int,
                 is_query: bool,
                 normalize_text: bool = False,
                 lower_case: bool = False,
                 separate_joint_encoding: bool = False,
                 add_lang_token: bool = False,
                 start: int = 0,
                 end: int = 0,
                 sep: str = " ",):
        super(EncodeDataset, self).__init__()

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_query = is_query

        self.data = []
        for idx, text in data.items():
            self.data.append((idx, text))
        if end > 0:
            self.data = self.data[start:end]

        self.normalize_text = normalize_text
        self.lower_case = lower_case
        self.separate_joint_encoding = separate_joint_encoding
        self.add_lang_token = add_lang_token

        self.start = start
        self.end = end
        self.sep = sep

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text_id, text = self.data[idx]

        if isinstance(self.tokenizer, (MT5Tokenizer, MT5TokenizerFast, T5Tokenizer, T5TokenizerFast)):
            if self.is_query:
                if isinstance(text, tuple):
                    text = text[0]
                    if isinstance(text, list):
                        assert len(text) == 2, text
                        text = text[1]
                    text = normalize_text.normalize(text)
                text = "question: " + text
            else:
                text = "title: " + text['title'] + " context: " + text['text']
        else:
            if not self.is_query:
                text = (text['title'] + self.sep + text['text']).strip()

            if isinstance(text, tuple):
                text = text[0]
                if isinstance(text, list):
                    assert len(text) == 2, text
                    text = text[1]

            if self.normalize_text:
                text = normalize_text.normalize(text)
            if self.lower_case:
                text = text.lower()

        if self.separate_joint_encoding:
            if self.is_query:
                encoded_text = self.tokenizer.encode(text, add_special_tokens=False)[:self.max_length]
            else:
                encoded_text = self.tokenizer.encode(text, add_special_tokens=True, max_length=self.max_length)
        else:
            encoded_text = self.tokenizer.encode_plus(
                text,
                max_length=self.max_length,
                truncation='only_first',
                padding=False,
                return_token_type_ids=False,
            )

        return text_id, encoded_text


@dataclass
class ReaderCollator(DataCollatorWithPadding):
    max_query_length: int = 50
    max_passage_length: int = 200
    max_query_passage_length: int = 250
    max_answer_length: int = 50
    separate_joint_encoding: bool = False
    parallel_queries: bool = False
    multilingual_distill: bool = False
    dictionaries: dict = None
    teacher_tokenizer: PreTrainedTokenizer = None

    def separate_joint_encode(self, batch_qids, batch_query, batch_passage, batch_answer, batch_pids, tokenizer):
        pad_token_id = tokenizer.pad_token_id

        batch_input_ids = []
        batch_attention_mask, batch_independent_mask, batch_query_mask, batch_passage_mask = [], [], [], []
        for query, passages in zip(batch_query, batch_passage):
            query_ids = tokenizer.encode(query, add_special_tokens=False)[:self.max_query_length]
            for passage in passages:
                passage_ids = tokenizer.encode(passage, add_special_tokens=True, max_length=self.max_passage_length)
                query_passage_ids = query_ids + passage_ids

                padded_length = self.max_query_passage_length - len(query_passage_ids)
                attention_mask = [1] * len(query_passage_ids) + [0] * padded_length
                query_mask = [1] * len(query_ids) + [0] * (len(passage_ids) + padded_length)
                passage_mask = [0] * len(query_ids) + [1] * len(passage_ids) + [0] * padded_length
                independent_mask = torch.zeros(self.max_query_passage_length, self.max_query_passage_length,
                                               dtype=torch.long)
                independent_mask[:len(query_ids), :len(query_ids)] = 1
                independent_mask[len(query_ids):, len(query_ids): len(query_passage_ids)] = 1

                batch_input_ids.append(query_passage_ids + [pad_token_id] * padded_length)
                batch_attention_mask.append(attention_mask)
                batch_independent_mask.append(independent_mask)
                batch_query_mask.append(query_mask)
                batch_passage_mask.append(passage_mask)

        batch_input_ids = torch.LongTensor(batch_input_ids)
        batch_attention_mask = torch.LongTensor(batch_attention_mask)
        batch_independent_mask = torch.stack(batch_independent_mask, dim=0)
        batch_query_mask = torch.LongTensor(batch_query_mask)
        batch_passage_mask = torch.LongTensor(batch_passage_mask)

        with tokenizer.as_target_tokenizer():
            encoded_answer = tokenizer.batch_encode_plus(
                batch_answer,
                max_length=self.max_answer_length,
                truncation=True,
                padding=True,
                return_tensors='pt'
            )
            labels = encoded_answer['input_ids']
            answer_mask = encoded_answer["attention_mask"].bool()
            labels = labels.masked_fill(~answer_mask, -100)

        reader_inputs = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "independent_mask": batch_independent_mask,
            "query_mask": batch_query_mask,
            "passage_mask": batch_passage_mask,
            'labels': labels,
        }

        if isinstance(batch_pids[0][0], tuple):
            batch_pids = ([[pid[0] for pid in pids] for pids in batch_pids],
                          torch.tensor([pid[1] for pids in batch_pids for pid in pids], dtype=torch.float64))

        return batch_qids, reader_inputs, batch_answer, batch_pids

    def encode(self, batch_query, batch_passage, batch_answer):
        # retriever inputs
        encoded_query = self.tokenizer.batch_encode_plus(
            batch_query,
            max_length=self.max_query_length,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        encoded_passage = self.tokenizer.batch_encode_plus(
            sum(batch_passage, []),
            max_length=self.max_passage_length,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )

        # reader inputs
        encoded_query_passage = self.tokenizer.batch_encode_plus(
            [query + " " + passage for query, passages in zip(batch_query, batch_passage) for passage in passages],
            max_length=self.max_query_passage_length,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )

        with self.tokenizer.as_target_tokenizer():
            encoded_answer = self.tokenizer.batch_encode_plus(
                batch_answer,
                max_length=self.max_answer_length,
                truncation=True,
                padding=True,
                return_tensors='pt'
            )
            labels = encoded_answer['input_ids']
            answer_mask = encoded_answer["attention_mask"].bool()
            labels = labels.masked_fill(~answer_mask, -100)

        reader_inputs = {
            "input_ids": encoded_query_passage['input_ids'],
            "attention_mask": encoded_query_passage['attention_mask'],
            'labels': labels,
        }

        return encoded_query, encoded_passage, reader_inputs

    def code_switch_augmentation(self, batch_query):
        batch_code_switching_query = []
        for query in batch_query:
            lang = random.choice(list(self.dictionaries.keys()))
            dictionary = self.dictionaries[lang]
            code_switching_query = []
            # for each query, the token is only in one target language, mixed with possible original english tokens
            for token in query.split(" "):
                if token == "question:":
                    code_switching_query.append(token)
                    continue
                if token.lower() in dictionary:
                    code_switching_query.append(random.choice(dictionary[token.lower()]))
                elif token.title() in dictionary:
                    code_switching_query.append(random.choice(dictionary[token.title()]))
                elif token.upper() in dictionary:
                    code_switching_query.append(random.choice(dictionary[token.upper()]))
                else:
                    code_switching_query.append(token)
            batch_code_switching_query.append(" ".join(code_switching_query))

        return batch_code_switching_query

    def __call__(self, features):
        batch_qids = [x[0] for x in features]
        batch_query = [x[1] for x in features]
        batch_passage = [x[2] for x in features]
        batch_answer = [x[3] for x in features]
        batch_pids = [x[4] for x in features]

        if self.teacher_tokenizer is not None:
            if isinstance(batch_query[0], tuple):
                if self.multilingual_distill:
                    teacher_query = [query[1] for query in batch_query]
                else:
                    teacher_query = [query[0] for query in batch_query]
                # print("teacher", teacher_query)
            else:
                teacher_query = batch_query
            teacher_inputs = self.separate_joint_encode(batch_qids, teacher_query, batch_passage, batch_answer,
                                                        batch_pids, self.teacher_tokenizer)
        if self.dictionaries is not None:
            batch_code_switching_query = self.code_switch_augmentation(batch_query)

            batch_qids = batch_qids + batch_qids
            batch_query = batch_query + batch_code_switching_query
            batch_passage = batch_passage + batch_passage
            batch_answer = batch_answer + batch_answer
            batch_pids = batch_pids + batch_pids
        elif self.parallel_queries:
            assert isinstance(batch_query[0], tuple), "Parallel queries must be tuple"
            # print("student", batch_query)
            batch_qids = batch_qids + batch_qids
            if self.multilingual_distill:
                batch_query = [query[1] for query in batch_query] + [query[0] for query in batch_query]
            else:
                batch_query = [query for sublist in zip(*batch_query) for query in sublist]
            batch_passage = batch_passage + batch_passage
            batch_answer = batch_answer + batch_answer
            batch_pids = batch_pids + batch_pids
        elif isinstance(batch_query[0], tuple):
            batch_query = [query[1] for query in batch_query]

        if self.separate_joint_encoding:
            inputs = self.separate_joint_encode(batch_qids, batch_query, batch_passage, batch_answer, batch_pids,
                                                self.tokenizer)
            if self.teacher_tokenizer is not None:
                return teacher_inputs, inputs
            else:
                return inputs

        encoded_query, encoded_passage, reader_inputs = self.encode(batch_query, batch_passage, batch_answer)

        return batch_qids, encoded_query, encoded_passage, reader_inputs, batch_answer, batch_pids


@dataclass
class EncodeCollator(DataCollatorWithPadding):
    separate_joint_encoding: bool = False
    padding_to_max_length: bool = True

    def __call__(self, features):
        if self.separate_joint_encoding:
            batch_text_id = [x[0] for x in features]
            batch_text = [x[1] for x in features]

            input_ids, attention_mask = [], []
            max_length = self.max_length if self.padding_to_max_length else \
                min(max([len(x) for x in batch_text]), self.max_length)
            for text in batch_text:
                padded_length = max_length - len(text)
                mask = [1] * len(text) + [0] * padded_length
                input_ids.append(text + [self.tokenizer.pad_token_id] * padded_length)
                attention_mask.append(mask)

            input_ids = torch.LongTensor(input_ids)
            attention_mask = torch.LongTensor(attention_mask)

            batch_input = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

            return batch_text_id, batch_input

        batch_text_id = [x[0] for x in features]
        batch_text = [x[1] for x in features]
        batch_text = super().__call__(batch_text)

        return batch_text_id, batch_text
