# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 13:16:36 2018

@author: Bart
"""

from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf


class ZeroMaskedEntries(Layer):
    """
    This layer is called after a Bidirectional layer with Masking active.
    It zeros out all of the masked-out entries that. This ise necessary
    because the Bidirectional layer does not output 0 for masked values, but
    rather just ignores them and outputs the last calculated value.

    It also swallows the mask without passing it on.
    You can change this to default pass-on behavior as follows:

    def compute_mask(self, x, mask=None):
        if not self.mask_zero:
            return None
        else:
            return K.not_equal(x, 0)
    """

    def __init__(self, **kwargs):
        self.support_mask = True
        super(ZeroMaskedEntries, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        return tf.where(mask, x, K.zeros_like(x))

    def compute_mask(self, input_shape, input_mask=None):
        return None