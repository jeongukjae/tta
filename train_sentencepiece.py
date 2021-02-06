import argparse

import sentencepiece as spm

parser = argparse.ArgumentParser()
parser.add_argument("--input")
parser.add_argument("--model-prefix")
parser.add_argument("--vocab-size", default=15000)


if __name__ == "__main__":
    args = parser.parse_args()

    spm.SentencePieceTrainer.train(
        input=args.input,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        shuffle_input_sentence=True,
        input_sentence_size=1000000,
        pad_id=0,
        user_defined_symbols=["<mask>"],
        bos_id=1,
        eos_id=2,
        unk_id=3,
    )
