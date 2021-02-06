import argparse

import tensorflow as tf
import tensorflow_text as text
from scipy import stats

from model import TTAConfig, TTAForPretraining

parser = argparse.ArgumentParser()
parser.add_argument("--model-weight", required=True, type=str)
parser.add_argument("--dataset", required=True, type=str)
args = parser.parse_args()

tta_config = TTAConfig.from_json("./configs/base.json")
model = TTAForPretraining(tta_config)
model.load_weights(args.model_weight)
model.tta(
    {
        "input_word_ids": tf.keras.Input(shape=[None], dtype=tf.int64),
        "input_mask": tf.keras.Input(shape=[None], dtype=tf.int64),
    }
)
model.tta.summary()

with open("./tokenizer/tokenizer.model", "rb") as f:
    tokenizer = text.SentencepieceTokenizer(f.read(), add_bos=True, add_eos=True)

with open(args.dataset) as f:
    lines = [line.split("\t") for line in f][1:]
refs = [line[-2] for line in lines]
hyps = [line[-1] for line in lines]
labels = [float(line[-3]) for line in lines]


def get_repr(sentences):
    sentences = tf.constant(sentences)
    tokenized = tokenizer.tokenize(sentences)
    input_mask = tf.ragged.map_flat_values(tf.ones_like, tokenized)

    input_word_ids = tokenized.to_tensor()
    input_mask = input_mask.to_tensor()

    input_tensor = {
        "input_word_ids": input_word_ids,
        "input_mask": input_mask,
    }

    input_mask = tf.cast(input_mask, tf.float32)
    sentence_embedding = tf.reduce_sum(model.tta(input_tensor) * input_mask[:, :, tf.newaxis], axis=1)
    return sentence_embedding


refs_repr = get_repr(refs)
hyps_repr = get_repr(hyps)

similarities = (-tf.keras.losses.cosine_similarity(refs_repr, hyps_repr)).numpy().tolist()
print(stats.spearmanr(labels, similarities))
