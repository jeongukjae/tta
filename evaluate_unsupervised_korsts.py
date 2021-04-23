import argparse

import tensorflow as tf
import tensorflow_text as text
import tensorflow_datasets as tfds
import tfds_korean.korsts
from scipy import stats

from model import TTAConfig, TTAForPretraining

parser = argparse.ArgumentParser()
parser.add_argument("--model-weight", required=True, type=str)
parser.add_argument("--model-config", required=True, type=str)
parser.add_argument("--split-name", required=True, type=str)
parser.add_argument("--tokenizer", required=True, type=str)
args = parser.parse_args()

tta_config = TTAConfig.from_json(args.model_config)
model = TTAForPretraining(tta_config)
model.load_weights(args.model_weight)
model.tta(
    {
        "input_word_ids": tf.keras.Input(shape=[None], dtype=tf.int64),
        "input_mask": tf.keras.Input(shape=[None], dtype=tf.int64),
    }
)
model.tta.summary()

with open(args.tokenizer, "rb") as f:
    tokenizer = text.SentencepieceTokenizer(f.read(), add_bos=True, add_eos=True)


def _map_model_input(ds_item):
    sentences = tf.stack([ds_item['sentence1'], ds_item['sentence2']], axis=0)
    tokenized = tokenizer.tokenize(sentences)
    input_mask = tf.ragged.map_flat_values(tf.ones_like, tokenized)

    return {
        "input_word_ids": tokenized.to_tensor(),
        "input_mask": input_mask.to_tensor(),
    }, ds_item['score']


target_ds = tfds.load('korsts', split=args.split_name, batch_size=128).map(_map_model_input).prefetch(10000)


@tf.function
def inference(model_input):
    input_shapes = tf.shape(model_input['input_word_ids'])
    model_input = {
        "input_word_ids": tf.reshape(model_input['input_word_ids'], [-1, input_shapes[-1]]),
        "input_mask": tf.reshape(model_input['input_mask'], [-1, input_shapes[-1]])
    }
    n_sentences = input_shapes[1]

    model_output = model.tta(model_input)
    output_mask = tf.cast(model_input['input_mask'], tf.float32)[:, :, tf.newaxis]
    sentence_embedding = tf.reduce_sum(model_output * output_mask, axis=1)
    sentence_embedding = tf.nn.l2_normalize(sentence_embedding, axis=-1)

    refs_repr = sentence_embedding[:n_sentences]
    hyps_repr = sentence_embedding[n_sentences:]

    cosine_similarities = tf.reduce_sum(refs_repr * hyps_repr, axis=-1)

    return cosine_similarities


inference = inference.get_concrete_function(target_ds.element_spec[0])

sims = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
scores = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

for model_input, score in target_ds:
    sims = sims.scatter(sims.size() + tf.range(tf.size(score)), inference(model_input))
    scores = scores.scatter(scores.size() + tf.range(tf.size(score)), score)

sims = sims.stack()
scores = scores.stack()

print(stats.spearmanr(scores, sims))
