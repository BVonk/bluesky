# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 09:57:10 2019

@author: Bart
"""
import tensorflow as tf
import argparse
import keras.backend as backend
from keras.models import model_from_yaml
from custom_keras_layers import ZeroMaskedEntries
import os
from bicnet import BiCNet
import yaml
from keras.utils import CustomObjectScope
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import seaborn as sns



os.chdir('../../bluesky/tools')


path  = 'C:/Users/Bart/Documents/bluesky/output/20190214_134835/'
episode = 5000

actormodel = os.path.join(path, 'actor_model{0:05d}.yaml'.format(episode))
actorweights = os.path.join(path, 'actor_model{:05d}.h5'.format(episode))

# Load the keras model architecture from file and load the weights
stream = open(actormodel, 'r')
actor = model_from_yaml(stream, custom_objects={'ZeroMaskedEntries': ZeroMaskedEntries, "tf": tf})
stream.close()
actor.load_weights(actorweights)

weights, biases = actor.layers[8].get_weights()
#weights, biases = actor.layers[2].get_weights()
#weights, biases = actor.layers[10].get_weights()

cmap = sns.diverging_palette(240, 10 , center='light', as_cmap=True)
sns.heatmap(weights, cmap = cmap)
