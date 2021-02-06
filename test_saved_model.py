import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  # noqa

model = hub.KerasLayer("./saved-model/model/0")
model(
    {
        "input_word_ids": tf.keras.Input(shape=[None], dtype=tf.int32),
        "input_mask": tf.keras.Input(shape=[None], dtype=tf.int32),
    }
)

preprocess = hub.KerasLayer("./saved-model/preprocess/0")
preprocess(tf.keras.Input(shape=[], dtype=tf.string))

input_tensors = preprocess(["나는 강아지를 좋아해", "나는 고양이를 좋아해", "한 여성이 기타를 연주하고 있다.", "한 남자가 기타를 치고 있다."])
representations = model(input_tensors)
representations = tf.reduce_sum(representations * tf.cast(input_tensors["input_mask"], representations.dtype)[:, :, tf.newaxis], axis=1)
representations = tf.nn.l2_normalize(representations, axis=-1)
similarities = tf.tensordot(representations, representations, axes=[[1], [1]])
print(similarities)
