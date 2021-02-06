import argparse

import tensorflow as tf
import tensorflow_text as text
from tensorflow.keras import mixed_precision

from model import TTAConfig, TTAForPretraining

parser = argparse.ArgumentParser()
parser.add_argument("--train-data", action="append")
parser.add_argument("--dev-data", type=str)
parser.add_argument("--model-config", default="configs/base.json")
parser.add_argument("--batch-size", default=64)
parser.add_argument("--spm-model", default="tokenizer/tokenizer.model")
parser.add_argument("--learning-rate", default=1e-4)
parser.add_argument("--target-epoch", default=1000)
parser.add_argument("--steps-per-epoch", default=1000)
parser.add_argument("--warmup-ratio", default=0.05)


def main():
    policy = mixed_precision.Policy("mixed_float16")
    mixed_precision.set_global_policy(policy)

    print("Compute dtype: %s" % policy.compute_dtype)
    print("Variable dtype: %s" % policy.variable_dtype)

    args = parser.parse_args()
    tta_config = TTAConfig.from_json(args.model_config)
    model = TTAForPretraining(tta_config)
    model(
        {
            "input_word_ids": tf.keras.Input(shape=[None], dtype=tf.int64),
            "input_mask": tf.keras.Input(shape=[None], dtype=tf.int64),
        }
    )
    model.summary()

    with open(args.spm_model, "rb") as f:
        tokenizer = text.SentencepieceTokenizer(f.read(), add_bos=True, add_eos=True)

    def preprocess_and_make_label(strings: tf.Tensor):
        tokenized = tokenizer.tokenize(strings)
        input_mask = tf.ragged.map_flat_values(tf.ones_like, tokenized)

        input_word_ids = tokenized.to_tensor(shape=[tokenized.shape[0], tta_config.max_position_ids])
        labels = tokenized.to_tensor(shape=[tokenized.shape[0], tta_config.max_position_ids], default_value=-1)
        input_mask = input_mask.to_tensor(shape=[tokenized.shape[0], tta_config.max_position_ids])

        return {
            "input_word_ids": input_word_ids,
            "input_mask": input_mask,
        }, labels

    trainset = (
        tf.data.TextLineDataset(args.train_data, num_parallel_reads=tf.data.AUTOTUNE)
        .shuffle(100000)
        .repeat()
        .batch(args.batch_size)
        .map(preprocess_and_make_label, num_parallel_calls=tf.data.AUTOTUNE)
    )
    devset = (
        tf.data.TextLineDataset(args.dev_data.split(","), num_parallel_reads=tf.data.AUTOTUNE)
        .shuffle(50000)
        .take(10000)
        .batch(args.batch_size)
        .map(preprocess_and_make_label, num_parallel_calls=tf.data.AUTOTUNE)
    )
    print(f"Total step: {args.steps_per_epoch * args.target_epoch}")
    print(f"learning rate: {args.learning_rate}, warmup ratio: {args.warmup_ratio}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LinearWarmupAndDecayScheduler(
                args.learning_rate, warmup_ratio=args.warmup_ratio, total_steps=args.steps_per_epoch * args.target_epoch
            )
        ),
        loss=sparse_categorical_crossentropy_with_ignore,
        metrics=[sparse_categorical_accuracy_with_ignore],
    )
    model.fit(
        trainset,
        validation_data=devset,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.target_epoch,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint("./models/model-{epoch}", save_best_only=True, verbose=True, save_weights_only=True),
        ],
    )


def sparse_categorical_crossentropy_with_ignore(y_true, y_pred, ignore_id=-1):
    positions = tf.where(y_true != ignore_id)

    y_true = tf.gather_nd(y_true, positions)
    y_pred = tf.gather_nd(y_pred, positions)

    return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)


def sparse_categorical_accuracy_with_ignore(y_true, y_pred, ignore_id=-1):
    positions = tf.where(y_true != ignore_id)

    y_true = tf.gather_nd(y_true, positions)
    y_pred = tf.gather_nd(y_pred, positions)

    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)


class LinearWarmupAndDecayScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, rate, warmup_ratio, total_steps, name=None):
        super().__init__()

        self.rate = rate
        self.warmup_ratio = warmup_ratio
        self.total_steps = float(total_steps)
        self.warmup_steps = warmup_ratio * total_steps
        self.name = name

    def __call__(self, step):
        with tf.name_scope("LinearWarmupAndDecayScheduler"):
            total_steps = tf.convert_to_tensor(self.total_steps, name="total_steps")
            warmup_steps = tf.convert_to_tensor(self.warmup_steps, name="warmup_steps")

            current_step = step + 1.0

            return self.rate * tf.cond(
                current_step < warmup_steps,
                lambda: self.warmup(current_step, warmup_steps),
                lambda: self.decay(current_step, total_steps, warmup_steps),
            )

    @tf.function
    def warmup(self, step, warmup_steps):
        return step / tf.math.maximum(tf.constant(1.0), warmup_steps)

    @tf.function
    def decay(self, step, total_steps, warmup_steps):
        return tf.math.maximum(tf.constant(0.0), (total_steps - step) / tf.math.maximum(tf.constant(1.0), total_steps - warmup_steps))

    def get_config(self):
        return {
            "warmup_ratio": self.warmup_ratio,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "name": self.name,
        }


if __name__ == "__main__":
    main()
