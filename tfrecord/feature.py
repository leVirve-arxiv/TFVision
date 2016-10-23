import tensorflow as tf


class Feature:

    def __init__(self, **kwargs):
        self.instance = self.parse_feature(**kwargs)

    def parse_feature(self, **kwargs):
        int64_feature = kwargs.get('int64_list')
        bytes_feature = kwargs.get('bytes_list')
        float_feature = kwargs.get('float_list')

        if int64_feature:
            return tf.train.Feature(
                int64_list=tf.train.Int64List(value=[int64_feature])
            )
        if bytes_feature:
            return tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[bytes_feature])
            )
        if float_feature:
            return tf.train.Feature(
                float_list=tf.train.FloatList(value=[float_feature])
            )
