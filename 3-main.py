#!/usr/bin/env python3

import matplotlib.image as mpimg
import os
import random
import numpy as np
import tensorflow as tf

NST = __import__('3-neural_style').NST


if __name__ == '__main__':
    style_image = np.random.uniform(0, 255, size=(400, 400, 3))
    content_image = np.random.uniform(0, 255, size=(400, 400, 3))

    # Reproducibility
    SEED = 0
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_ENABLE_ONEDNN_OPTS']= '0'
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    nst = NST(style_image, content_image)
    print(nst.gram_style_features)
    print(nst.content_feature)
