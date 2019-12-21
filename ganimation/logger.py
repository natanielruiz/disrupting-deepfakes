import tensorflow as tf
import numpy as np


class Logger(object):
    """Tensorboard logger."""

    def __init__(self, log_dir):
        """Initialize summary writer."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Add scalar summary."""
        summary = tf.Summary(
            value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, name, x, step):
        x = x.numpy()[0, :, :, :]
        x = np.moveaxis(x, 0, -1)
        x = np.expand_dims(x, 0)

        tensor = tf.convert_to_tensor(
            x,
            dtype=tf.float32,
            name=None,
            preferred_dtype=None
        )

        print(tensor.value)

        summary = tf.summary.image(name=name, tensor=tensor)

        self.writer.add_summary(summary, step).eval()
