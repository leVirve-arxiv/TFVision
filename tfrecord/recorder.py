import tensorflow as tf


class Recorder:

    def __init__(self, path):
        self.path = path
        self.writer = None

    def creat_example(self, features):
        fs = {k: v.instance for k, v in features.items()}
        example = tf.train.Example(features=tf.train.Features(feature=fs))
        self.writer.write(example.SerializeToString())

    def __enter__(self):
        self.writer = tf.python_io.TFRecordWriter(self.path)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self.writer.close()
