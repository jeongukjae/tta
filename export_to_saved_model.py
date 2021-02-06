import argparse
import os

import tensorflow as tf
import tensorflow_text as text

from model import TTAConfig, TTAForPretraining

parser = argparse.ArgumentParser()
parser.add_argument("--model-config", required=True, type=str)
parser.add_argument("--model-weight", required=True, type=str)
parser.add_argument("--tokenizer-path", required=True, type=str)
parser.add_argument("--output-path", required=True, type=str)
args = parser.parse_args()

tta_config = TTAConfig.from_json(args.model_config)
pretraining_model = TTAForPretraining(tta_config)
pretraining_model.load_weights(args.model_weight)

tta = pretraining_model.tta
tta(
    {
        "input_word_ids": tf.keras.Input(shape=[None], dtype=tf.int32),
        "input_mask": tf.keras.Input(shape=[None], dtype=tf.int32),
    }
)
tta.summary()

with open(args.tokenizer_path, "rb") as f:
    tokenizer = text.SentencepieceTokenizer(f.read(), add_bos=True, add_eos=True)


@tf.function(input_signature=[tf.TensorSpec([None], dtype=tf.string)])
def preprocess_input(sentences):
    tokenized = tokenizer.tokenize(sentences)
    input_mask = tf.ragged.map_flat_values(tf.ones_like, tokenized)

    input_word_ids = tokenized.to_tensor()
    input_mask = input_mask.to_tensor()

    return {
        "input_word_ids": input_word_ids,
        "input_mask": input_mask,
    }


tokenizer.__call__ = preprocess_input

tf.saved_model.save(tta, os.path.join(args.output_path, "model", "0"))
tf.saved_model.save(tokenizer, os.path.join(args.output_path, "preprocess", "0"))
