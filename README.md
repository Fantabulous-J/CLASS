## Pre-training Cross-lingual Open Domain Question Answering with Large-scale Synthetic Supervision
The source code for our EMNLP 2024 Paper [Pre-training Cross-lingual Open Domain Question Answering with Large-scale Synthetic Supervision](https://arxiv.org/abs/2402.16508).

## Install environment
```shell
pip install -r requirements.txt
```

## Evaluation
### Models
- [fanjiang98/CLASS-XOR-Retrieve](https://huggingface.co/fanjiang98/CLASS-XOR-Retrieve): model fine-tuned on XOR-Retrieve.
- [fanjiang98/CLASS-XOR-Full](https://huggingface.co/fanjiang98/CLASS-XOR-Full): model fine-tuned on XOR-Full.
### XOR-TYDI-QA
#### Download Dataset
```shell
mkdir -p data/XOR-Retrieve
cd data/XOR-Retrieve
wget https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_train_retrieve_eng_span.jsonl
wget https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_dev_retrieve_eng_span_v1_1.jsonl
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-train.qa.csv
wget https://nlp.cs.washington.edu/xorqa/XORQA_site/data/models/enwiki_20190201_w100.tsv -O psgs_w100.tsv
cd ../../

mkdir -p data/XOR-Full
cd data/XOR-Full
wget https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_train_full.jsonl
wget https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_dev_full_v1_1.jsonl
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-train.qa.csv
wget https://nlp.cs.washington.edu/xorqa/cora/models/all_w100.tsv
cd ../../
```

#### XOR-Retrieve
#### Generate Embeddings
Encode Query
```shell
bash scripts/XOR-Retrieve/encode_query.sh
```
Encode Corpus
```shell
bash scripts/XOR-Retrieve/encode_corpus.sh
```
Note that ```MODEL_PATH``` should be ```fanjiang98/CLASS-XOR-Retrieve```.
#### Retrieve
```shell
bash scripts/XOR-Retrieve/retrieve_hn.sh
```
Note that ```MODEL_PATH``` should be ```fanjiang98/CLASS-XOR-Retrieve```
We use the official scripts provided by XOR-TYDI-QA for evaluation:
```shell
python3 evals/eval_xor_retrieve.py \
    --data_file <path_to_input_data> \
    --pred_file <path_to_predictions>
```
This leads to the following results:

<table style="text-align: center;">
  <tr>
    <th>Model</th>
    <th colspan="8">R@2k</th>
    <th colspan="8">R@5k</th>
  </tr>
  <tr>
      <td></td>
      <th>Ar</th>
      <th>Bn</th>
      <th>Fi</th>
      <th>Ja</th>
      <th>Ko</th>
      <th>Ru</th>
      <th>te</th>
      <th>Avg</th>
      <th>Ar</th>
      <th>Bn</th>
      <th>Fi</th>
      <th>Ja</th>
      <th>Ko</th>
      <th>Ru</th>
      <th>te</th>
      <th>Avg</th>
  </tr>
  <tr>
      <td>CLASS-US</td>
      <td>54.5</td>
      <td>67.4</td>
      <td>58.6</td>
      <td>47.7</td>
      <td>51.6</td>
      <td>59.9</td>
      <td>65.6</td>
      <td>57.9</td>
      <td>64.8</td>
      <td>73.0</td>
      <td>64.7</td>
      <td>57.3</td>
      <td>58.6</td>
      <td>67.9</td>
      <td>70.6</td>
      <td>65.3</td>
  </tr>
  <tr>
      <td>CLASS-ZS</td>
      <td>59.2</td>
      <td>70.1</td>
      <td>59.9</td>
      <td>51.5</td>
      <td>57.2</td>
      <td>51.5</td>
      <td>72.3</td>
      <td>60.2</td>
      <td>66.7</td>
      <td>78.6</td>
      <td>66.6</td>
      <td>60.2</td>
      <td>63.2</td>
      <td>58.2</td>
      <td>78.2</td>
      <td>67.4</td>
  </tr>
  <tr>
      <td>CLASS</td>
      <td>66.7</td>
      <td>79.6</td>
      <td>64.3</td>
      <td>58.1</td>
      <td>66.0</td>
      <td>64.1</td>
      <td>77.7</td>
      <td>68.1</td>
      <td>70.6</td>
      <td>84.9</td>
      <td>71.0</td>
      <td>66.0</td>
      <td>72.6</td>
      <td>70.0</td>
      <td>81.9</td>
      <td>73.9</td>
  </tr>
</table>

#### XOR-Full
#### Retrieve
It is the same as in XOR-Retrieve. Please find corresponding scripts under ```scripts/XOR-Full``` and replace ```MODEL_PATH``` with ```fanjiang98/CLASS-XOR-Full```.

#### Answer Generation
```shell
bash scripts/XOR-Full/eval_reader.sh
```
```MODEL_PATH``` should be ```fanjiang98/CLASS-XOR-Full```. We use the official scripts provided by XOR-TYDI-QA for evaluation:
```shell
python3 evals/eval_xor_full.py \
    --data_file <path_to_input_data> \
    --pred_file <path_to_predictions>
```

This leads to the following results:

<table style="margin-left: auto; margin-right: auto; text-align: center;">
  <tr>
    <td>Model</td>
    <th colspan="7">F1</th>
    <th colspan="3">Macro Average</th>
  </tr>
  <tr>
      <td></td>
      <th>Ar</th>
      <th>Bn</th>
      <th>Fi</th>
      <th>Ja</th>
      <th>Ko</th>
      <th>Ru</th>
      <th>te</th>
      <th>F1</th>
      <th>EM</th>
      <th>BLEU</th>
  </tr>
  <tr>
      <td>CORA</td>
      <td>42.9</td>
      <td>26.9</td>
      <td>41.4</td>
      <td>36.8</td>
      <td>30.4</td>
      <td>33.9</td>
      <td>30.9</td>
      <td>34.7</td>
      <td>25.8</td>
      <td>23.3</td>
  </tr>
  <tr>
      <td>CLASS</td>
      <td>49.1</td>
      <td>32.0</td>
      <td>46.7</td>
      <td>44.1</td>
      <td>38.4</td>
      <td>39.9</td>
      <td>41.1</td>
      <td>41.6</td>
      <td>32.5</td>
      <td>28.2</td>
  </tr>
</table>

### Training
Please download the training data from [OneDrive](https://unimelbcloud-my.sharepoint.com/:f:/g/personal/jifj_student_unimelb_edu_au/EkkBMU65NG1LvGkBHKpMEvMB3QAlGT599dgL9wDNPCgUWw?e=eMQwHK) and put them on corresponding directories under `data`.

1. Stage-1 Pre-training:
```shell
bash scripts/train_mss_distill_reader.sh
```
2. Stage-2 Pre-training:
```shell
bash scripts/XOR-Retrieve/train_mss_iterative_reader.sh
```
3. Fine-tuning on Natural Questions (zero-shot model):
```shell
bash scripts/XOR-Retrieve/train_nq_iterative_reader.sh
```
4. Fine-tuning on XOR-Retrieve training data (i.e., our released CLASS-XOR-Retrieve model):
```shell
bash scripts/XOR-Retrieve/train_iterative_reader.sh
```
The training pipeline for XOR-Full is the same, please find corresponding scripts under ```scripts/XOR-Full``` for steps 2, 3 and 4.

We use slurm for training on 32 80G A100 for stage-1 and 16 A100 for the rest.

### Acknowledgement
Some of the code was adapted from https://github.com/jzbjyb/ReAtt.
