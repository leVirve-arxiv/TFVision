import tensorflow as tf


class Recorder:

    def __init__(self, path):
        self.path = path
        self.writer = None

    def feature_extractor(self, filename):
        raise NotImplementedError(
            'You have to provide your own `feature_extractor()` '
            'through Recorder.feature_extractor = your_function.')

    def create_example(self, filename):
        result = self.feature_extractor(filename)
        fs = {k: v.instance for k, v in result.items()}
        example = tf.train.Example(features=tf.train.Features(feature=fs))
        self.writer.write(example.SerializeToString())

    def __enter__(self):
        self.writer = tf.python_io.TFRecordWriter(self.path)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self.writer.close()
