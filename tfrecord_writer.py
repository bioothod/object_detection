import tensorflow as tf

import logging
import os

class tf_records_writer(object):
    def __init__(self, base, total_items, max_items):
        self.counter = int(total_items / max_items)
        self.items = int(total_items % max_items)
        self.total_written = total_items
        self.max_items = max_items
        self.base = base
        self.writer = None
        self.reopen()

    def path(self):
        return '{}.{}'.format(self.base, self.counter)

    def new_counters(self):
        self.counter += 1
        self.items = 0

    def find_next_name(self):
        while os.path.exists(self.path()):
            self.new_counters()

        return self.path()

    def reopen(self):
        orig_path = self.path()

        path = self.find_next_name()

        if self.writer is not None:
            self.close()

        logging.info('tf_writer: opening file {} (originally {}), total_written: {}, counter: {}'.format(path, orig_path, self.total_written, self.counter))

        self.writer = tf.io.TFRecordWriter(path)

    def write(self, example):
        self.writer.write(example)
        self.items += 1
        self.total_written += 1
        if self.items == self.max_items:
            self.new_counters()
            self.reopen()

    def close(self):
        self.writer.flush()
        self.writer.close()
        self.writer = None
