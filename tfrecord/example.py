import tensorflow as tf


class Example:

    def __init__(self, feature=None):
        self.features = feature

    def serialize(self):
        fs = {k: v.instance for k, v in self.features.items()}
        return tf.train.Example(
                   features=tf.train.Features(feature=fs)
               ).SerializeToString()

    def __repr__(self):
        return '<Example> Contains %d features ' % len(self.features)
