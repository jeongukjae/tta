import json
import math
import warnings
from typing import Callable, Union

import tensorflow as tf


def get_activation_fn(activation: str):
    if activation == "gelu":
        return tf.nn.gelu
    if activation == "relu":
        return tf.nn.relu
    if activation == "tanh":
        return tf.nn.tanh

    return activation


class TTAConfig:
    def __init__(
        self,
        vocab_size: int,
        num_hidden_layers: int = 12,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        intermediate_activation: str = "gelu",
        num_attention_heads: int = 12,
        max_position_ids: int = 512,
        dropout_rate: float = 0.1,
        attention_probs_dropout_rate: float = 0.1,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        if len(kwargs) != 0:
            warnings.warn("Unused parameters found: " + kwargs)

        self.vocab_size = vocab_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.intermediate_activation = get_activation_fn(intermediate_activation)
        self.num_attention_heads = num_attention_heads
        self.max_position_ids = max_position_ids
        self.dropout_rate = dropout_rate
        self.attention_probs_dropout_rate = attention_probs_dropout_rate
        self.initializer_range = initializer_range

    @staticmethod
    def from_json(json_filename: str, **kwargs) -> "TTAConfig":
        with open(json_filename, encoding="utf8") as f:
            jsondict = json.load(f)
            jsondict.update(kwargs)

        return TTAConfig(**jsondict)


class TTAForPretraining(tf.keras.Model):
    def __init__(self, tta_config: TTAConfig, **kwargs):
        super().__init__(**kwargs)

        self.tta = TTAModel(tta_config, name="tta")
        self.transform = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    tta_config.hidden_size,
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=tta_config.initializer_range),
                ),
                tf.keras.layers.Activation(tta_config.intermediate_activation, dtype="float32"),
                tf.keras.layers.LayerNormalization(dtype="float32"),
            ]
        )
        self.output_bias = self.add_weight(name="output_bias", shape=[tta_config.vocab_size], initializer="zeros", trainable=True)
        self.hidden_size = tta_config.hidden_size

    def call(self, input_tensor):
        encoded = self.tta(input_tensor)
        transformed = self.transform(encoded)
        transformed = tf.cast(transformed, encoded.dtype)

        embedding_table = self.tta.tta_embedding.input_word_embeddings.embeddings

        transformed_shape = tf.shape(transformed)
        batch_size = transformed_shape[0]
        sequence_length = transformed_shape[1]

        transformed = tf.reshape(transformed, [-1, self.hidden_size])
        logits = tf.matmul(transformed, tf.cast(embedding_table, transformed.dtype), transpose_b=True)
        logits = tf.nn.bias_add(logits, self.output_bias)
        logits = tf.keras.layers.Softmax(axis=-1, dtype="float32")(logits)

        logits = tf.reshape(logits, [batch_size, sequence_length, -1])
        return logits


class TTAModel(tf.keras.Model):
    def __init__(self, tta_config: TTAConfig, unk_id: int = 4, **kwargs):
        super().__init__(**kwargs)

        self.tta_embedding = TTAEmbedding(
            vocab_size=tta_config.vocab_size,
            hidden_size=tta_config.hidden_size,
            max_position_ids=tta_config.max_position_ids,
            dropout_rate=tta_config.dropout_rate,
            initializer_range=tta_config.initializer_range,
            name="tta_embedding",
        )

        self.transformer_layers = [
            TransformerEncoder(
                hidden_size=tta_config.hidden_size,
                num_attention_heads=tta_config.num_attention_heads,
                intermediate_size=tta_config.intermediate_size,
                intermediate_activation=tta_config.intermediate_activation,
                dropout_rate=tta_config.dropout_rate,
                attention_probs_dropout_rate=tta_config.attention_probs_dropout_rate,
                initializer_range=tta_config.initializer_range,
            )
            for i in range(tta_config.num_hidden_layers)
        ]
        self.unk_id = unk_id

    def call(self, input_tensor):
        input_word_ids = input_tensor["input_word_ids"]
        input_mask = input_tensor["input_mask"]

        embedding = self.tta_embedding(input_word_ids)
        position_embedding = self.tta_embedding(tf.ones_like(input_word_ids) * self.unk_id)

        with tf.name_scope("input_mask"):
            input_mask = tf.cast(input_mask, embedding.dtype)[:, tf.newaxis, tf.newaxis, :]
            input_mask *= 1.0 - tf.cast(tf.eye(tf.shape(input_mask)[-1])[tf.newaxis, tf.newaxis, :, :], embedding.dtype)
            input_mask = 1.0 - input_mask
            input_mask *= -10000.0

        hidden_state = position_embedding
        for layer in self.transformer_layers:
            hidden_state = layer([hidden_state, embedding], input_mask=input_mask)

        return hidden_state


class TTAEmbedding(tf.keras.layers.Layer):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_ids: int,
        dropout_rate: float,
        initializer_range: float,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.input_word_embeddings = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=hidden_size,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range),
            name="input_word_embeddings",
        )
        self.position_embeddings = tf.keras.layers.Embedding(
            input_dim=max_position_ids,
            output_dim=hidden_size,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range),
            name="position_embeddings",
        )

        self.layer_normalization = tf.keras.layers.LayerNormalization(name="layer_norm", dtype="float32")
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, input_word_ids):
        position_ids = tf.expand_dims(tf.range(tf.shape(input_word_ids)[-1]), 0)

        input_word_embedding = self.input_word_embeddings(input_word_ids)
        position_embedding = self.position_embeddings(position_ids)

        embeddings = tf.add(input_word_embedding, position_embedding)
        embeddings = self.layer_normalization(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        intermediate_activation: Union[Callable, str],
        dropout_rate: float,
        attention_probs_dropout_rate: float,
        initializer_range: float,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.multihead_attention = MultiheadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            dropout_rate=dropout_rate,
            attention_probs_dropout_rate=attention_probs_dropout_rate,
            initializer_range=initializer_range,
            name="multihead_attention",
        )
        self.mha_layer_normalization = tf.keras.layers.LayerNormalization(name="mha_layer_normalization", dtype="float32")

        self.intermediate_layer = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    intermediate_size,
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range),
                ),
                tf.keras.layers.Activation(intermediate_activation, dtype="float32"),
                tf.keras.layers.Dense(
                    hidden_size,
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range),
                ),
                tf.keras.layers.Dropout(dropout_rate),
            ],
            name="intermediate_layer",
        )
        self.intermediate_layer_normalization = tf.keras.layers.LayerNormalization(name="intermediate_layer_normalization", dtype="float32")

    def call(self, hidden_states, input_mask=None):
        # hidden_states: [query, key_and_value]
        query = hidden_states[0]
        key_and_value = hidden_states[1]

        attention_output = self.multihead_attention([query, key_and_value], input_mask=input_mask)
        hidden_state = self.mha_layer_normalization(attention_output + query)
        hidden_state = tf.cast(hidden_state, attention_output.dtype)

        intermediate_output = self.intermediate_layer(hidden_state)
        hidden_state = self.intermediate_layer_normalization(intermediate_output + hidden_state)
        hidden_state = tf.cast(hidden_state, intermediate_output.dtype)
        return hidden_state


class MultiheadAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        dropout_rate: float,
        attention_probs_dropout_rate: float,
        initializer_range: float,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert hidden_size % num_attention_heads == 0
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // self.num_attention_heads
        self.scaling_factor = 1.0 / math.sqrt(float(self.attention_head_size))

        self.query_proj = tf.keras.layers.Dense(
            hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range),
            name="query_proj",
        )
        self.key_proj = tf.keras.layers.Dense(
            hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range),
            name="key_proj",
        )
        self.value_proj = tf.keras.layers.Dense(
            hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range),
            name="value_proj",
        )
        self.attention_dropout = tf.keras.layers.Dropout(attention_probs_dropout_rate)
        self.output_projection = tf.keras.layers.Dense(
            hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range),
            name="output_projection",
        )
        self.output_dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, input_tensor, input_mask=None):
        # input_tensor: [query, key_and_value]
        query = self.query_proj(input_tensor[0])
        key = self.key_proj(input_tensor[1])
        value = self.value_proj(input_tensor[1])

        batch_size = tf.shape(query)[0]

        query = self.transpose_for_scores(query, batch_size)
        key = self.transpose_for_scores(key, batch_size)
        value = self.transpose_for_scores(value, batch_size)

        attention_scores = tf.matmul(query, key, transpose_b=True)
        attention_scores *= tf.cast(self.scaling_factor, attention_scores.dtype)
        if input_mask is not None:
            attention_scores += input_mask

        attention_probs = tf.keras.layers.Softmax(axis=-1, dtype="float32")(attention_scores)
        attention_probs = tf.cast(attention_probs, dtype=attention_scores.dtype)
        attention_probs = self.attention_dropout(attention_probs)

        context_layer = tf.matmul(attention_probs, value)
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
        context_layer = tf.reshape(context_layer, [batch_size, -1, self.num_attention_heads * self.attention_head_size])

        output = self.output_projection(context_layer)
        output = self.output_dropout(output)
        return output

    def transpose_for_scores(self, x, batch_size):
        x = tf.reshape(x, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
        return tf.transpose(x, [0, 2, 1, 3])
