# BMOC-KEPLM

Sorce code and datasets for ["BMOC-KEPLM: Bias Mitigation and Optimized Classification in Knowledge-Enhanced Pre-trained Language Models"], which is implemented based on the [UER](https://github.com/dbiir/UER-py) framework.

**News**

- [EasyNLP](https://github.com/alibaba/EasyNLP) integrated the K-BERT. For details, see [EasyNLP集成K-BERT算法，借助知识图谱实现更优Finetune](https://zhuanlan.zhihu.com/p/553816104).


## Requirements

Software:
```
Python3
Pytorch >= 1.0
argparse == 1.1
```


## Prepare

* Download the ``google_model.bin`` and ``text_ent_encoder.bin`` from [here](https://www.weiyun.com/disk), and save them to the ``models/`` directory.
* Download the ``CnDbpedia.spo`` from [here](https://www.weiyun.com/disk), and save it to the ``brain/kgs/`` directory.
* Optional - Download the datasets for evaluation from [here](https://www.weiyun.com/disk), unzip and place them in the ``datasets/`` directory.

The directory tree of BMOC-KEPLM:
```
BMOC-KEPLM
├── brain
│   ├── config.py
│   ├── __init__.py
│   ├── kgs
│   │   ├── CnDbpedia.spo
│   │   ├── HowNet.spo
│   │   └── Medical.spo
│   └── knowgraph.py
├── datasets
│   ├── book_review
│   │   ├── dev.tsv
│   │   ├── test.tsv
│   │   └── train.tsv
│   ├── chnsenticorp
│   │   ├── dev.tsv
│   │   ├── test.tsv
│   │   └── train.tsv
│    ...
│
├── models
│   ├── google_config.json
│   ├── google_model.bin
│   ├── text_ent_encoder.bin
│   └── google_vocab.txt
├── outputs
├── uer
├── README.md
├── requirements.txt
└── run_bmoc_keplm_cls.py
```


## BMOC-KEPLM for text classification

### Classification example

Run example on different datasets (for example, on Book review dataset) with CnDbpedia:
```sh
CUDA_VISIBLE_DEVICES='0' nohup python3 -u run_bmoc_keplm_cls.py \
    --pretrained_model_path ./models/google_model.bin \
    --relevance_model_path ./models/text_ent_encoder.bin \
    --config_path ./models/google_config.json \
    --vocab_path ./models/google_vocab.txt \
    --train_path ./datasets/book_review/train.tsv \
    --dev_path ./datasets/book_review/dev.tsv \
    --test_path ./datasets/book_review/test.tsv \
    --epochs_num 25 --batch_size 32 --kg_name CnDbpedia \
    --output_model_path ./outputs/kbert_bookreview_CnDbpedia.bin \
    > ./outputs/kbert_bookreview_CnDbpedia.log &
```


