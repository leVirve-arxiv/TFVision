import tensorflow as tf


class Feature:
    def __init__(self, value):
        self._value = value


class FeatureInt64(Feature):
    def instance(self):
        return tf.train.Feature(
            int64_list=tf.train.Int64List(value=[self._value])
        )


class FeatureBytes(Feature):
    def instance(self):
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[self._value])
        )


class FeatureFloat(Feature):
    def instance(self):
        return tf.train.Feature(
            float_list=tf.train.FloatList(value=[self._value])
        )
