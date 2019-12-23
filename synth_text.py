import logging

import numpy as np
import scipy.io as sio
import tensorflow as tf

logger = logging.getLogger('detection')

def load_synth_dataset(fn='/shared2/object_detection/datasets/text/synth_text/SynthText/gm.mat', data_dir='/shared2/object_detection/datasets/text/synth_text/SynthText/'):
    m = sio.loadmat(fn)

    filenames = m['imnames'][0]
    char_polys = m['charBB'][0]
    word_polys = m['wordBB'][0]
    texts = m['txt'][0]

    return filenames, char_polys, word_polys, texts
