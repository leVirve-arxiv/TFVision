import tensorflow as tf


class Record:

    def __init__(self, path):
        self.path = path
        self.writer = None

    def write(self, example):
        self.writer.write(example.serialize())

    def __enter__(self):
        self.writer = tf.python_io.TFRecordWriter(self.path)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self.writer.close()
