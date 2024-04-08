from dataclasses import dataclass, field
from typing import Optional, List, Union

import torch
from transformers import TrainingArguments
from transformers.training_args import ParallelMode
from transformers.utils import cached_property, logging

from utils import _infer_slurm_init

logger = logging.get_logger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    target_model_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained reranker target model"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    # modeling
    shared_encoder: bool = field(
        default=True,
        metadata={"help": "weight sharing between qry passage encoders"}
    )

    teacher_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    teacher_config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as teacher_model_name"}
    )
    teacher_tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as teacher_model_name"}
    )


@dataclass
class DataArguments:
    task: str = field(
        default="XOR-Retrieve", metadata={"help": "task name"}
    )
    train_dir: str = field(
        default=None, metadata={"help": "Path to train directory"}
    )
    train_path: str = field(
        default=None, metadata={"help": "Path to train data"}
    )
    dataset_name: str = field(
        default=None, metadata={"help": "huggingface dataset name"}
    )
    dataset_proc_num: int = field(
        default=12, metadata={"help": "number of proc used in dataset preprocess"}
    )
    corpus_file: str = field(default="corpus.tsv", metadata={"help": "corpus text path"})
    query_file: str = field(default="train.query.txt", metadata={"help": "query text path"})
    eval_query_file: str = field(default="xor_dev_retrieve_eng_span_v1_1.jsonl",
                                 metadata={"help": "query text path for evaluation"})
    teacher_corpus_file: str = field(default="corpus.tsv", metadata={"help": "corpus text path for teacher model"})
    teacher_query_file: str = field(default="train.query.txt", metadata={"help": "query text path for teacher model"})
    qrels_file: str = field(default="qrels.train.tsv", metadata={"help": "query passage relation path"})
    output_path: str = field(default=None, metadata={"help": "output path"})

    labelled_train_path: str = field(
        default=None, metadata={"help": "Path to train data"}
    )
    labelled_query_file: str = field(default="train.query.txt", metadata={"help": "query text path"})

    train_n_passages: int = field(default=8)
    no_shuffle_positive: bool = field(default=False)
    sample_hard_negative_prob: float = field(default=1.0)

    encode_in_path: List[str] = field(default=None, metadata={"help": "Path to data to encode"})
    encoded_save_path: str = field(default=None, metadata={"help": "where to save the encode"})
    encode_is_qry: bool = field(default=False)
    encode_num_shard: int = field(default=1)
    encode_shard_index: int = field(default=0)
    split: str = field(default='train', metadata={"help": "dataset splits"})
    normalize_text: bool = field(default=False, metadata={"help": "normalize text"})
    lower_case: bool = field(default=False, metadata={"help": "lower case text"})
    add_lang_token: bool = field(default=False, metadata={"help": "add language token"})
    load_partial: bool = field(default=False, metadata={"help": "load partial data"})
    add_positive_passage: bool = field(default=False, metadata={"help": "add positive passage to the training data"})
    lang: str = field(default="en", metadata={"help": "language"})

    max_query_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_passage_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    max_query_passage_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage & query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        }
    )

    max_answer_length: int = field(default=50, metadata={"help": "max answer length"})

    retrieval_data_path: str = field(default=None, metadata={"help": "retrieval results data path"})

    max_teacher_query_length: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_teacher_passage_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )


@dataclass
class BiEncoderTrainingArguments(TrainingArguments):
    warmup_ratio: float = field(default=0.1)
    negatives_x_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    do_encode: bool = field(default=False, metadata={"help": "run the encoding loop"})
    qgen_pretrain: bool = field(default=False, metadata={"help": "do pretraining on cleaned synthetic data"})
    retriever_score_scaling: bool = field(default=False, metadata={"help": "scale retriever score or not when "
                                                                           "computing the distribution"})

    grad_cache: bool = field(default=False, metadata={"help": "Use gradient cache update"})
    gc_chunk_size: int = field(default=8)

    print_steps: int = field(default=100, metadata={"help": "step for displaying"})
    teacher_temp: float = field(default=1)
    student_temp: float = field(default=1)

    distributed_port: int = field(default=None, metadata={"help": "port for multi-node multi-gpu distributed training "
                                                                  "using slurm"})
    multi_task: bool = field(default=False, metadata={"help": "use multi-task training"})

    distillation_start_steps: int = field(default=3000, metadata={"help": "start distillation after certain steps"})
    separate_joint_encoding: bool = field(default=False, metadata={"help": "separate joint encoding"})
    cross_lingual_alignment: bool = field(default=False, metadata={"help": "cross lingual alignment"})
    query_alignment: bool = field(default=False, metadata={"help": "query alignment"})

    refresh_passages: bool = field(default=False, metadata={"help": "refresh passages"})
    refresh_intervals: int = field(default=3000, metadata={"help": "refresh intervals"})

    retriever_weight: float = field(default=1.0, metadata={"help": "retriever weight"})

    barlow_twins: bool = field(default=False, metadata={"help": "use barlow twins loss"})

    english_distill: bool = field(default=False, metadata={"help": "distillation from english teacher"})
    multilingual_distill: bool = field(default=False, metadata={"help": "distillation from multilingual teacher"})
    parallel_queries: bool = field(default=False, metadata={"help": "use parallel queries"})
    wikidata: bool = field(default=False, metadata={"help": "use wikidata"})
    e2e_training: bool = field(default=False, metadata={"help": "end-to-end training"})
    homogeneous_batch: bool = field(default=False, metadata={"help": "use homogeneous data (i.e. same languages) "
                                                                     "within a batch"})

    only_reader: bool = field(default=False, metadata={"help": "only train the reader component and "
                                                               "keep the retriever component frozen"})

    eval_on_test: bool = field(default=False, metadata={"help": "evaluate on test set when training is done"})
    eval_on_mkqa: bool = field(default=False, metadata={"help": "evaluate on mkqa set when training is done"})

    load_corpus: bool = field(default=True, metadata={"help": "load corpus from file"})

    debug: bool = field(default=False, metadata={"help": "debug mode"})

    eval_at_start: bool = field(default=False, metadata={"help": "evaluate at start of training"})

    use_mcontriever: bool = field(default=False, metadata={"help": "use mcontriever model to retrieve relevant passages"
                                                                   " in first k training steps"})
    self_retrieve_steps: int = field(default=3000, metadata={"help": "number of steps to use self retriever"})

    de_avg_pooling: bool = field(default=False, metadata={"help": "use average pooling for dual-encoder"})

    scheduler_type: str = field(default="linear_warmup", metadata={"help": "sheduler type"})

    add_bias: bool = field(default=False, metadata={"help": "add bias to the cross-attention"})

    retrieve_from_each_lang: bool = field(default=False, metadata={"help": "retrieve from each language"})

    @cached_property
    def _setup_devices(self) -> "torch.device":
        if self.distributed_port:
            logger.info("PyTorch: setting up devices")
            distributed_init_method, local_rank, world_size, device_id = _infer_slurm_init(self.distributed_port)
            self.local_rank = local_rank
            torch.distributed.init_process_group(
                backend="nccl", init_method=distributed_init_method, world_size=world_size, rank=local_rank
            )

            logger.info("local rank {}, device id {}".format(local_rank, device_id))
            self._n_gpu = 1
            if device_id is None:
                device = torch.device("cuda")
            else:
                device = torch.device("cuda", device_id)
                if device.type == "cuda":
                    torch.cuda.set_device(device)

            return device
        else:
            return super(BiEncoderTrainingArguments, self)._setup_devices

    @property
    def parallel_mode(self):
        if self.local_rank != -1:
            return ParallelMode.DISTRIBUTED
        elif self.n_gpu > 1:
            return ParallelMode.NOT_DISTRIBUTED
        else:
            return ParallelMode.NOT_PARALLEL


@dataclass
class CrossEncoderTrainingArguments(BiEncoderTrainingArguments):
    pass


@dataclass
class DynamicNegativeDistillTrainingArguments(BiEncoderTrainingArguments):
    update_negative_steps: int = field(default=None, metadata={"help": "steps for updating negatives using the "
                                                                       "latest retriever"})
    update_negative: bool = field(default=True, metadata={"help": "whether to update hard negatives"})
    passage_embeddings: str = field(default=None, metadata={"help": "path for storing passage embeddings"})
    query_embeddings: str = field(default=None, metadata={"help": "path for storing query embeddings"})
    retrieve_batch_size: int = field(default=5000, metadata={"help": "batch size for faiss search"})
    depth: int = field(default=32, metadata={"help": "top-k negatives"})


@dataclass
class OnTheFlyNegativeDistillTrainingArguments(DynamicNegativeDistillTrainingArguments):
    pass


@dataclass
class DistilModelArguments(ModelArguments):
    teacher_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    teacher_config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as teacher_model_name"}
    )
    teacher_tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as teacher_model_name"}
    )
    self_distill: bool = field(default=False, metadata={"help": "use self-distillation on crop sentences or not"})
    distill: bool = field(default=False, metadata={"help": "take the crop-sent trained dual-encoder as teacher, and "
                                                           "distill the knowledge to a cross-encoder"})
    enc_dec: bool = field(default=False, metadata={"help": "use full encoder-decoder structure of T5 for reranking"})