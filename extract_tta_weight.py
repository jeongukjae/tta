import argparse

import tensorflow as tf

from model import TTAConfig, TTAForPretraining

parser = argparse.ArgumentParser()
parser.add_argument("--model-config", required=True, type=str)
parser.add_argument("--model-weight", required=True, type=str)
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
tta.save_weights(args.output_path)
