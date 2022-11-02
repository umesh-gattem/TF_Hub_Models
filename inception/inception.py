import tensorflow as tf
import tensorflow_hub as hub

m = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5",
                   trainable=False),  # Can be True, see below.
    tf.keras.layers.Dense(10, activation='softmax')
])
m.build([None, 299, 299, 3])  # Batch input shape.