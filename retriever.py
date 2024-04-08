import glob
import json
import logging
from argparse import ArgumentParser
from itertools import chain

import faiss
import numpy as np
import torch
from tqdm import tqdm

from dataloader import GenericDataLoader

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


class FaissIPRetriever:
    def __init__(self, init_reps: np.ndarray, use_gpu=False):
        index = faiss.IndexFlatIP(init_reps.shape[1])
        if use_gpu:
            ngpus = faiss.get_num_gpus()
            assert ngpus > 0, ngpus
            config = faiss.GpuMultipleClonerOptions()
            config.shard = True
            config.useFloat16 = True
            index = faiss.index_cpu_to_all_gpus(index, co=config)
        self.index = index
        self.use_gpu = use_gpu

    def search(self, query_vector: np.ndarray, k: int):
        return self.index.search(query_vector, k)

    def add(self, passage_vector: np.ndarray):
        self.index.add(passage_vector)

    def batch_search(self, query_vector: np.ndarray, k: int, batch_size: int):
        num_query = query_vector.shape[0]
        all_scores = []
        all_indices = []
        for start_idx in tqdm(range(0, num_query, batch_size)):
            nn_scores, nn_indices = self.search(query_vector[start_idx: start_idx + batch_size], k)
            all_scores.append(nn_scores)
            all_indices.append(nn_indices)
        all_scores = np.concatenate(all_scores, axis=0)
        all_indices = np.concatenate(all_indices, axis=0)

        return all_scores, all_indices


def write_ranking(corpus_indices, corpus_scores, q_lookup, ranking_save_file, qas_file=None, corpus=None,
                  save_text=False, save_jsonl=False, depth=100):
    assert not (save_text and save_jsonl), "Only one of save_text or save_jsonl can be True"
    if save_text:
        with open(ranking_save_file, 'w') as f:
            for qid, q_doc_scores, q_doc_indices in zip(q_lookup, corpus_scores, corpus_indices):
                score_list = [(s, idx) for s, idx in zip(q_doc_scores, q_doc_indices)]
                score_list = sorted(score_list, key=lambda x: x[0], reverse=True)
                for k, (s, idx) in enumerate(score_list[:depth]):
                    f.write(f'{qid} Q0 {idx} {k + 1} {s} dense\n')
    elif save_jsonl:
        with open(ranking_save_file, 'w') as f:
            for qid, q_doc_scores, q_doc_indices in tqdm(zip(q_lookup, corpus_scores, corpus_indices)):
                score_list = [(s, idx) for s, idx in zip(q_doc_scores, q_doc_indices)]
                score_list = sorted(score_list, key=lambda x: x[0], reverse=True)
                f.write(json.dumps({"qid": qid, "pids": [idx for s, idx in score_list[:depth]]}) + '\n')
    else:
        qas = {}
        for question, qid, answers, lang in parse_qa_jsonlines_file(qas_file):
            qas[qid] = (question, answers, lang)

        xor_output_prediction_format = []
        with open(ranking_save_file, 'w') as f:
            for qid, q_doc_scores, q_doc_indices in zip(q_lookup, corpus_scores, corpus_indices):
                score_list = [(s, idx) for s, idx in zip(q_doc_scores, q_doc_indices)]
                score_list = sorted(score_list, key=lambda x: x[0], reverse=True)
                ctxs = []
                for s, idx in score_list[:depth]:
                    ctxs.append(corpus[idx]["text"])
                question, answers, lang = qas[qid]
                xor_output_prediction_format.append({"id": qid, "lang": lang, "ctxs": ctxs})
            json.dump(xor_output_prediction_format, f)


def search_queries(retriever, q_reps, p_lookup, args):
    if args.batch_size > 0:
        all_scores, all_indices = retriever.batch_search(q_reps, args.depth, args.batch_size)
    else:
        all_scores, all_indices = retriever.search(q_reps, args.depth)

    psg_indices = [[p_lookup[x] for x in q_dd] for q_dd in all_indices]
    return all_scores, psg_indices


def read_jsonlines(eval_file_name):
    examples = []
    print("loading examples from {0}".format(eval_file_name))
    with open(eval_file_name) as f:
        for jsonline in f.readlines():
            examples.append(json.loads(jsonline))
    return examples


def parse_qa_jsonlines_file(location):
    data = read_jsonlines(location)
    for row in data:
        question = row["question"]
        answers = row["answers"]
        qid = row["id"]
        lang = row["lang"]
        yield question, qid, answers, lang


def main():
    parser = ArgumentParser()
    parser.add_argument('--query_embeddings', required=True)
    parser.add_argument('--passage_embeddings', required=True)
    parser.add_argument('--train_dir', type=str, default=None)
    parser.add_argument('--qas_file', type=str, default=None)
    parser.add_argument('--corpus_file', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--depth', type=int, default=1000)
    parser.add_argument('--save_ranking_to', required=True)
    parser.add_argument('--save_text', action='store_true')
    parser.add_argument('--save_jsonl', action='store_true')
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--search_then_merge', action='store_true')

    args = parser.parse_args()

    faiss.omp_set_num_threads(72)

    if args.search_then_merge:
        q_reps, q_lookup = torch.load(args.query_embeddings)
        q_reps = q_reps.float().numpy()

        if '*' in args.passage_embeddings:
            index_files = glob.glob(args.passage_embeddings)
            index_files = sorted(index_files)
            logger.info(f'Pattern match found {len(index_files)} files; Search then merge.')
        else:
            index_files = [args.passage_embeddings]
        logger.info('Index Search Start')
        all_scores, psg_indices = None, None
        for file in index_files:
            logger.info(f'Search over index file: {file}')
            p_reps, p_lookup = torch.load(file)
            shards = [(p_reps, p_lookup)]
            retriever = FaissIPRetriever(p_reps.float().numpy(), args.use_gpu)

            look_up = []
            if args.use_gpu:
                all_p_reps = []
                for p_reps, p_lookup in shards:
                    all_p_reps.append(p_reps.numpy())
                    look_up += p_lookup
                retriever.add(np.concatenate(all_p_reps, axis=0))
            else:
                for p_reps, p_lookup in shards:
                    retriever.add(p_reps.float().numpy())
                    look_up += p_lookup

            scores, indices = search_queries(retriever, q_reps, look_up, args)
            if all_scores is None and psg_indices is None:
                all_scores = scores
                psg_indices = indices
            else:
                def topk_by_sort(input, k, axis=None, ascending=True):
                    if not ascending:
                        input *= -1
                    ind = np.argsort(input, axis=axis)
                    ind = np.take(ind, np.arange(k), axis=axis)
                    if not ascending:
                        input *= -1
                    val = np.take_along_axis(input, ind, axis=axis)
                    return ind, val

                all_scores = np.concatenate((all_scores, scores), axis=1)
                psg_indices = np.concatenate((psg_indices, indices), axis=1)

                ind, all_scores = topk_by_sort(all_scores, args.depth, axis=1, ascending=False)
                psg_indices = np.take_along_axis(psg_indices, ind, axis=1)

        logger.info('Index Search Finished')
    else:
        if '*' in args.passage_embeddings:
            index_files = glob.glob(args.passage_embeddings)
            index_files = sorted(index_files)
            logger.info(f'Pattern match found {len(index_files)} files; loading them into index.')

            p_reps_0, p_lookup_0 = torch.load(index_files[0])
            retriever = FaissIPRetriever(p_reps_0.float().numpy(), args.use_gpu)

            shards = chain([(p_reps_0, p_lookup_0)], map(torch.load, index_files[1:]))
            if len(index_files) > 1:
                shards = tqdm(shards, desc='Loading shards into index', total=len(index_files))
        else:
            p_reps, p_lookup = torch.load(args.passage_embeddings)
            shards = [(p_reps, p_lookup)]
            retriever = FaissIPRetriever(p_reps.float().numpy(), args.use_gpu)
        look_up = []

        if args.use_gpu:
            all_p_reps = []
            for p_reps, p_lookup in shards:
                all_p_reps.append(p_reps.numpy())
                look_up += p_lookup
            retriever.add(np.concatenate(all_p_reps, axis=0))
        else:
            for p_reps, p_lookup in shards:
                retriever.add(p_reps.float().numpy())
                look_up += p_lookup

        if '*' in args.query_embeddings:
            index_files = glob.glob(args.query_embeddings)
            q_reps_0, q_lookup_0 = torch.load(index_files[0])
            shards = chain([(q_reps_0, q_lookup_0)], map(torch.load, index_files[1:]))
            if len(index_files) > 1:
                shards = tqdm(shards, desc='Loading shards into index', total=len(index_files))
            all_q_reps, all_q_lookup = [], []
            for q_reps, q_lookup in shards:
                all_q_reps.append(q_reps.float().numpy())
                all_q_lookup += q_lookup
            q_reps = np.concatenate(all_q_reps, axis=0)
            q_lookup = all_q_lookup
        else:
            q_reps, q_lookup = torch.load(args.query_embeddings)
            q_reps = q_reps.float().numpy()

        logger.info('Index Search Start')
        all_scores, psg_indices = search_queries(retriever, q_reps, look_up, args)
        logger.info('Index Search Finished')

    corpus = None
    if not args.save_text and not args.save_jsonl:
        corpus = GenericDataLoader(args.train_dir, corpus_file=args.corpus_file).load_corpus()

    write_ranking(psg_indices, all_scores, q_lookup, args.save_ranking_to, args.qas_file, corpus, args.save_text,
                  args.save_jsonl, args.depth)


if __name__ == '__main__':
    main()
