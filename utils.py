import json
import logging
import os
import pickle
import random
import re
import subprocess
import unicodedata
from typing import Tuple

import numpy as np
import regex
import torch
import torch.distributed as dist
from torch.utils.checkpoint import get_device_states, set_device_states

logger = logging.getLogger(__name__)

langid_to_lang = {
    "ar": "Arabic",
    "bn": "Bengali",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "he": "Hebrew",
    "id": "Indonesian",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
    "sv": "Swedish",
    "te": "Telugu",
    "th": "Thai",
}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def get_rank():
    return dist.get_rank()


def get_world_size():
    return dist.get_world_size()


def get_default_group():
    return dist.group.WORLD


def all_reduce(tensor, group=None):
    if group is None:
        group = get_default_group()
    return dist.all_reduce(tensor, group=group)


def all_gather_list(data, group=None, max_size=40000):
    SIZE_STORAGE_BYTES = 4  # int32 to encode the payload size

    enc = pickle.dumps(data)
    enc_size = len(enc)

    if enc_size + SIZE_STORAGE_BYTES > max_size:
        raise ValueError(
            'encoded data exceeds max_size, this can be fixed by increasing buffer size: {}'.format(enc_size))

    rank = get_rank()
    world_size = get_world_size()
    buffer_size = max_size * world_size

    if not hasattr(all_gather_list, '_buffer') or \
            all_gather_list._buffer.numel() < buffer_size:
        all_gather_list._buffer = torch.cuda.ByteTensor(buffer_size)
        all_gather_list._cpu_buffer = torch.ByteTensor(max_size).pin_memory()

    buffer = all_gather_list._buffer
    buffer.zero_()
    cpu_buffer = all_gather_list._cpu_buffer

    assert enc_size < 256 ** SIZE_STORAGE_BYTES, 'Encoded object size should be less than {} bytes'.format(
        256 ** SIZE_STORAGE_BYTES)

    size_bytes = enc_size.to_bytes(SIZE_STORAGE_BYTES, byteorder='big')

    cpu_buffer[0:SIZE_STORAGE_BYTES] = torch.ByteTensor(list(size_bytes))
    cpu_buffer[SIZE_STORAGE_BYTES: enc_size + SIZE_STORAGE_BYTES] = torch.ByteTensor(list(enc))

    start = rank * max_size
    size = enc_size + SIZE_STORAGE_BYTES
    buffer[start: start + size].copy_(cpu_buffer[:size])

    all_reduce(buffer, group=group)

    try:
        result = []
        for i in range(world_size):
            out_buffer = buffer[i * max_size: (i + 1) * max_size]
            size = int.from_bytes(out_buffer[0:SIZE_STORAGE_BYTES], byteorder='big')
            if size > 0:
                result.append(pickle.loads(bytes(out_buffer[SIZE_STORAGE_BYTES: size + SIZE_STORAGE_BYTES].tolist())))
        return result
    except pickle.UnpicklingError:
        raise Exception(
            'Unable to unpickle data from other workers. all_gather_list requires all '
            'workers to enter the function together, so this error usually indicates '
            'that the workers have fallen out of sync somehow. Workers can fall out of '
            'sync if one of them runs out of memory, or if there are other conditions '
            'in your training script that can cause one worker to finish an epoch '
            'while other workers are still iterating over their portions of the data.'
        )


def _infer_slurm_init(distributed_port) -> Tuple[str, int, int, int]:
    node_list = os.environ.get("SLURM_STEP_NODELIST")
    if node_list is None:
        node_list = os.environ.get("SLURM_JOB_NODELIST")
    logger.info("SLURM_JOB_NODELIST: %s", node_list)

    if node_list is None:
        raise RuntimeError("Can't find SLURM node_list from env parameters")

    local_rank = None
    world_size = None
    distributed_init_method = None
    device_id = None
    try:
        hostnames = subprocess.check_output(["scontrol", "show", "hostnames", node_list])
        distributed_init_method = "tcp://{host}:{port}".format(
            host=hostnames.split()[0].decode("utf-8"),
            port=distributed_port,
        )
        nnodes = int(os.environ.get("SLURM_NNODES"))
        logger.info("SLURM_NNODES: %s", nnodes)
        ntasks_per_node = os.environ.get("SLURM_NTASKS_PER_NODE")
        if ntasks_per_node is not None:
            ntasks_per_node = int(ntasks_per_node)
            logger.info("SLURM_NTASKS_PER_NODE: %s", ntasks_per_node)
            if ntasks_per_node == 1:
                gpus_per_node = torch.cuda.device_count()
                node_id = int(os.environ.get("SLURM_NODEID"))
                local_rank = node_id * gpus_per_node
                device_id = 0
                world_size = nnodes * gpus_per_node
                logger.info("node_id: %s", node_id)
            else:
                world_size = ntasks_per_node * nnodes
                proc_id = os.environ.get("SLURM_PROCID")
                local_id = os.environ.get("SLURM_LOCALID")
                logger.info("SLURM_PROCID %s", proc_id)
                logger.info("SLURM_LOCALID %s", local_id)
                local_rank = int(proc_id)
                device_id = int(local_id)
        else:
            ntasks = int(os.environ.get("SLURM_NTASKS"))
            logger.info("SLURM_NTASKS: %s", ntasks)
            # assert ntasks % nnodes == 0
            # ntasks_per_node = int(ntasks / nnodes)
            world_size = ntasks
            proc_id = os.environ.get("SLURM_PROCID")
            local_id = os.environ.get("SLURM_LOCALID")
            logger.info("SLURM_PROCID %s, SLURM_LOCALID %s", proc_id, local_id)
            local_rank = int(proc_id)

    except subprocess.CalledProcessError as e:  # scontrol failed
        raise e
    except FileNotFoundError:  # Slurm is not installed
        pass
    return distributed_init_method, local_rank, world_size, device_id


class RandContext:
    def __init__(self, *tensors):
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)

    def __enter__(self):
        self._fork = torch.random.fork_rng(
            devices=self.fwd_gpu_devices,
            enabled=True
        )
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None


# copy from https://github.com/facebookresearch/DPR/blob/main/dpr/utils/tokenizers.py
class Tokens(object):
    """A class to represent a list of tokenized text."""

    TEXT = 0
    TEXT_WS = 1
    SPAN = 2
    POS = 3
    LEMMA = 4
    NER = 5

    def __init__(self, data, annotators, opts=None):
        self.data = data
        self.annotators = annotators
        self.opts = opts or {}

    def __len__(self):
        """The number of tokens."""
        return len(self.data)

    def slice(self, i=None, j=None):
        """Return a view of the list of tokens from [i, j)."""
        new_tokens = copy.copy(self)
        new_tokens.data = self.data[i:j]
        return new_tokens

    def untokenize(self):
        """Returns the original text (with whitespace reinserted)."""
        return "".join([t[self.TEXT_WS] for t in self.data]).strip()

    def words(self, uncased=False):
        """Returns a list of the text of each token
        Args:
            uncased: lower cases text
        """
        if uncased:
            return [t[self.TEXT].lower() for t in self.data]
        else:
            return [t[self.TEXT] for t in self.data]

    def offsets(self):
        """Returns a list of [start, end) character offsets of each token."""
        return [t[self.SPAN] for t in self.data]

    def pos(self):
        """Returns a list of part-of-speech tags of each token.
        Returns None if this annotation was not included.
        """
        if "pos" not in self.annotators:
            return None
        return [t[self.POS] for t in self.data]

    def lemmas(self):
        """Returns a list of the lemmatized text of each token.
        Returns None if this annotation was not included.
        """
        if "lemma" not in self.annotators:
            return None
        return [t[self.LEMMA] for t in self.data]

    def entities(self):
        """Returns a list of named-entity-recognition tags of each token.
        Returns None if this annotation was not included.
        """
        if "ner" not in self.annotators:
            return None
        return [t[self.NER] for t in self.data]

    def ngrams(self, n=1, uncased=False, filter_fn=None, as_strings=True):
        """Returns a list of all ngrams from length 1 to n.
        Args:
            n: upper limit of ngram length
            uncased: lower cases text
            filter_fn: user function that takes in an ngram list and returns
              True or False to keep or not keep the ngram
            as_string: return the ngram as a string vs list
        """

        def _skip(gram):
            if not filter_fn:
                return False
            return filter_fn(gram)

        words = self.words(uncased)
        ngrams = [
            (s, e + 1)
            for s in range(len(words))
            for e in range(s, min(s + n, len(words)))
            if not _skip(words[s: e + 1])
        ]

        # Concatenate into strings
        if as_strings:
            ngrams = ["{}".format(" ".join(words[s:e])) for (s, e) in ngrams]

        return ngrams

    def entity_groups(self):
        """Group consecutive entity tokens with the same NER tag."""
        entities = self.entities()
        if not entities:
            return None
        non_ent = self.opts.get("non_ent", "O")
        groups = []
        idx = 0
        while idx < len(entities):
            ner_tag = entities[idx]
            # Check for entity tag
            if ner_tag != non_ent:
                # Chomp the sequence
                start = idx
                while idx < len(entities) and entities[idx] == ner_tag:
                    idx += 1
                groups.append((self.slice(start, idx).untokenize(), ner_tag))
            else:
                idx += 1
        return groups


class Tokenizer(object):
    """Base tokenizer class.
    Tokenizers implement tokenize, which should return a Tokens class.
    """

    def tokenize(self, text):
        raise NotImplementedError

    def shutdown(self):
        pass

    def __del__(self):
        self.shutdown()


class SimpleTokenizer(Tokenizer):
    ALPHA_NUM = r"[\p{L}\p{N}\p{M}]+"
    NON_WS = r"[^\p{Z}\p{C}]"

    def __init__(self, **kwargs):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            "(%s)|(%s)" % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE,
        )
        if len(kwargs.get("annotators", {})) > 0:
            logger.warning(
                "%s only tokenizes! Skipping annotators: %s" % (type(self).__name__, kwargs.get("annotators"))
            )
        self.annotators = set()

    def tokenize(self, text):
        data = []
        matches = [m for m in self._regexp.finditer(text)]
        for i in range(len(matches)):
            # Get text
            token = matches[i].group()

            # Get whitespace
            span = matches[i].span()
            start_ws = span[0]
            if i + 1 < len(matches):
                end_ws = matches[i + 1].span()[0]
            else:
                end_ws = span[1]

            # Format data
            data.append(
                (
                    token,
                    text[start_ws:end_ws],
                    span,
                )
            )
        return Tokens(data, self.annotators)


def has_answer(answers, text, tokenizer, match_type="string") -> bool:
    text = _normalize(text)

    if match_type == "string":
        # Answer is a list of possible strings
        text = tokenizer.tokenize(text).words(uncased=True)

        for single_answer in answers:
            single_answer = _normalize(single_answer)
            single_answer = tokenizer.tokenize(single_answer)
            single_answer = single_answer.words(uncased=True)

            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i: i + len(single_answer)]:
                    return True

    elif match_type == "regex":
        # Answer is a regex
        for single_answer in answers:
            single_answer = _normalize(single_answer)
            if regex_match(text, single_answer):
                return True
    return False


def regex_match(text, pattern):
    try:
        pattern = re.compile(pattern, flags=re.IGNORECASE + re.UNICODE + re.MULTILINE)
    except BaseException:
        return False
    return pattern.search(text) is not None


def _normalize(text):
    return unicodedata.normalize("NFD", text)


def load_dictionaries(reversed=False):
    total_dictionaries = {}
    for lang in ['ar', 'bn', 'fi', 'ja', 'ko', 'ru', 'te']:
        dictionary = {}
        with open(f"MUSE/en-{lang}.txt") as f:
            for line in f.readlines():
                try:
                    source, target = line.strip().split("\t")
                except:
                    source, target = line.strip().split()
                if source.strip() == target.strip():
                    continue
                if reversed:
                    source, target = target, source
                if source not in dictionary:
                    dictionary[source] = [target]
                else:
                    dictionary[source].append(target)
        total_dictionaries[lang] = dictionary

    return total_dictionaries


def translate_with_dictionary_ja(sentence, dictionary):
    from sudachipy import dictionary as sudachi_dictionary
    from sudachipy import tokenizer

    # Load the default dictionary
    tokenizer_obj = sudachi_dictionary.Dictionary().create()
    mode = tokenizer.Tokenizer.SplitMode.C
    tokens = tokenizer_obj.tokenize(sentence, mode=mode)
    words = [token.surface() for token in tokens]
    translated_sentence = []
    for word in words:
        if word in dictionary:
            translated_sentence.append(random.choice(dictionary[word]))
        else:
            translated_sentence.append(word)
    return " ".join(translated_sentence)


def translate_with_dictionary_ko(sentence, dictionary):
    from konlpy.tag import Okt
    tokenizer = Okt()
    words = tokenizer.morphs(sentence)
    translated_sentence = []
    for word in words:
        if word in dictionary:
            translated_sentence.append(random.choice(dictionary[word]))
        else:
            translated_sentence.append(word)
    return " ".join(translated_sentence)


def compute_colbert_scores(query_vector, passage_vector, query_mask, passage_mask):
    # [num_query, num_passages, q_len, p_len]
    score_list = []
    # query_vector.size()[0] // 256
    chunk_query_vector = torch.chunk(query_vector, chunks=max(query_vector.size()[0] // 256, 1), dim=0)
    for chunk in chunk_query_vector:
        scores = chunk.unsqueeze(1) @ passage_vector.unsqueeze(0).transpose(2, 3)
        scores = scores.masked_fill(~passage_mask[None, :, None].bool(), -1e9)
        scores = torch.max(scores, dim=-1).values
        score_list.append(scores)
    scores = torch.cat(score_list, dim=0)
    scores = scores.masked_fill(~query_mask[:, None].bool(), 0.0)
    scores = torch.sum(scores, dim=-1) / query_mask.sum(dim=1)[..., None]

    return scores


def group_corpus_by_langs(corpus):
    corpus_dict = {
        "ar": {},
        "bn": {},
        "en": {},
        "es": {},
        "fi": {},
        "he": {},
        "id": {},
        "ko": {},
        "ru": {},
        "sv": {},
        "te": {},
        "ja": {},
        "th": {}
    }
    for idx, (pid, text) in enumerate(corpus.items()):
        if 1 <= idx + 1 <= 18003200:
            corpus_dict["en"][pid] = text
        elif 18003201 <= idx + 1 <= 19308028:
            corpus_dict["ar"][pid] = text
        elif 19308029 <= idx + 1 <= 19487964:
            corpus_dict["bn"][pid] = text
        elif 19487965 <= idx + 1 <= 25226448:
            corpus_dict["es"][pid] = text
        elif 25226449 <= idx + 1 <= 26113043:
            corpus_dict["fi"][pid] = text
        elif 26113044 <= idx + 1 <= 27158298:
            corpus_dict["he"][pid] = text
        elif 27158299 <= idx + 1 <= 27978870:
            corpus_dict["id"][pid] = text
        elif 27978871 <= idx + 1 <= 28617734:
            corpus_dict["ko"][pid] = text
        elif 28617735 <= idx + 1 <= 33163369:
            corpus_dict["ru"][pid] = text
        elif 33163370 <= idx + 1 <= 37689064:
            corpus_dict["sv"][pid] = text
        elif 37689065 <= idx + 1 <= 37963294:
            corpus_dict["te"][pid] = text
        elif 37963295 <= idx + 1 <= 43080199:
            corpus_dict["ja"][pid] = text
        elif 43080200 <= idx + 1 <= 43600337:
            corpus_dict["th"][pid] = text

    return corpus_dict
