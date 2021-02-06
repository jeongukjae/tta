# T-TA

This repository contains codes for Transformer-base Text Auto-encoder (T-TA, [paper: Fast and Accurate Deep Bidirectional Language Representations for Unsupervised Learning](https://www.aclweb.org/anthology/2020.acl-main.76/)) using TensorFlow 2.

## How to train T-TA using custom dataset

1. Prepare datasets. You need text line files.

    Example:

    ```text
    Sentence 1.
    Sentence 2.
    Sentence 3.
    ```

2. Train the sentencepiece tokenizer. You can use the `train_sentencepiece.py` or train sentencepiece model by yourself.
3. Train T-TA model. Run `train.py` with customizable arguments. Here's the usage.

    ```sh
    $ python train.py --help
    usage: train.py [-h] [--train-data TRAIN_DATA] [--dev-data DEV_DATA] [--model-config MODEL_CONFIG] [--batch-size BATCH_SIZE] [--spm-model SPM_MODEL]
                    [--learning-rate LEARNING_RATE] [--target-epoch TARGET_EPOCH] [--steps-per-epoch STEPS_PER_EPOCH] [--warmup-ratio WARMUP_RATIO]

    optional arguments:
        -h, --help            show this help message and exit
        --train-data TRAIN_DATA
        --dev-data DEV_DATA
        --model-config MODEL_CONFIG
        --batch-size BATCH_SIZE
        --spm-model SPM_MODEL
        --learning-rate LEARNING_RATE
        --target-epoch TARGET_EPOCH
        --steps-per-epoch STEPS_PER_EPOCH
        --warmup-ratio WARMUP_RATIO
    ```

    I want to train models until the designated steps, so I added the `steps_per_epoch` and `target_epoch` arguments. The total steps will be the  `steps_per_epoch` * `target_epoch`.

4. (Optional) Test your model using [KorSTS data](https://github.com/kakaobrain/KorNLUDatasets). I trained my model with the Korean corpus, so I tested it using [KorSTS data](https://github.com/kakaobrain/KorNLUDatasets). You can evaluate KorSTS score (Spearman correlation) using `evaluate_unsupervised_korsts.py`. Here's the usage.

    ```sh
    $ python evaluate_unsupervised_korsts.py --help
    usage: evaluate_unsupervised_korsts.py [-h] --model-weight MODEL_WEIGHT --dataset DATASET

    optional arguments:
        -h, --help            show this help message and exit
        --model-weight MODEL_WEIGHT
        --dataset DATASET
    $ # To evaluate on dev set
    $ # python evaluate_unsupervised_korsts.py --model-weight ./path/to/checkpoint --dataset ./path/to/dataset/sts-dev.tsv
    ```

## Training details

* Training data: [lovit/namuwikitext](https://github.com/lovit/namuwikitext)
* Peak learning rate: `1e-4`
* learning rate scheduler: Linear Warmup and Linear Decay.
* Warmup ratio: 0.05 (warmup steps: 1M * 0.05 = 50k)
* Vocab size: `15000`
* num layers: `3`
* intermediate size: `2048`
* hidden size: `512`
* attention heads: `8`
* activation function: `gelu`
* max sequence length: `128`
* tokenizer: sentencepiece
* Total steps: 1M
* Final validation accuracy of auto encoding task (ignores padding): `0.5513`
* Final validation loss: `2.1691`

### Unsupervised KorSTS

|Model|Params|development|test|
|---|---|---|---|
|My Implementation|17M|65.98|56.75|
|Korean SRoBERTa (base)|111M|63.34|48.96|
|Korean SRoBERTa (large)|338M|60.15|51.35|
|SXLM-R (base)|270M|64.27|45.05|
|SXLM-R (large)|550M|55.00|39.92|
|Korean fastText| - | - |47.96|

KorSTS development and test set scores (100 * Spearman Correlation). You can check the details of all model except Korean T-TA on [this paper (KorNLI and KorSTS: New Benchmark Datasets for Korean Natural Language Understanding)](https://arxiv.org/abs/2004.03289).

## References

* Official implementaion: https://github.com/joongbo/tta
* [Fast and Accurate Deep Bidirectional Language Representations for Unsupervised Learning](https://www.aclweb.org/anthology/2020.acl-main.76/)
* [KorSTS data](https://github.com/kakaobrain/KorNLUDatasets)
* [KorNLI and KorSTS: New Benchmark Datasets for Korean Natural Language Understanding](https://arxiv.org/abs/2004.03289)
* https://github.com/lovit/namuwikitext
