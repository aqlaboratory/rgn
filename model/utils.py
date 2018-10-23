__author__ = "Mohammed AlQuraishi"
__copyright__ = "Copyright 2018, Harvard Medical School"
__license__ = "MIT"

import numpy as np
import tensorflow as tf
from ast import literal_eval

class switch(object):
    """Switch statement for Python, based on recipe from Python Cookbook."""

    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration
    
    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args: # changed for v1.5
            self.fall = True
            return True
        else:
            return False

def merge_two_dicts(x, y):
    """ Efficiently merges two dicts, giving precedence to second dict. """
    z = x.copy()
    z.update(y)
    return z

def merge_dicts(*dict_args):
    """ Efficiently merges arbitrary number of dicts, giving precedence to latter dicts. """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def ops_to_dict(session, ops):
    """ Helper function that converts canonical dict of TF ops to an actual dict. Runs ops first. """

    dict_ = dict(zip(ops.keys(), session.run(ops.values())))

    return dict_

def cum_quantile_positions(weights, quantiles = np.linspace(0.25, 0.99, 4)):
    """ Computes cumulative quantiles from curriculum weights. """
    if len(weights) != 0:
        return [next(x[0] + 1 for x in enumerate(np.cumsum(weights / sum(weights))) if x[1] > p) for p in quantiles]
    else:
        return []

def dict_to_init(dict_, seed=None, dtype=tf.float32):
    """ Accepts a dict in canonical config form and returns the appropriate initializer. """

    # effect defaults
    init_center = dict_.get('center', 0.0)
    init_range  = dict_.get('range',  0.01)
    init_dist   = dict_.get('dist',   'gaussian')
    init_scale  = dict_.get('scale',  1.0)
    init_mode   = dict_.get('mode',   'fan_in') # also accepts fan_out, fan_avg

    for case in switch(init_dist):
        if case('gaussian'):
            init = tf.initializers.random_normal(   init_center,              init_range,               seed=seed, dtype=dtype)
        elif case('uniform'):
            init = tf.initializers.random_uniform(  init_center - init_range, init_center + init_range, seed=seed, dtype=dtype)
        elif case('orthogonal'):
            init = tf.initializers.orthogonal(      init_scale,                                         seed=seed, dtype=dtype)
        elif case('gaussian_variance_scaling'):
            init = tf.initializers.variance_scaling(init_scale,               init_mode, 'normal',      seed=seed, dtype=dtype)
        elif case('uniform_variance_scaling'):
            init = tf.initializers.variance_scaling(init_scale,               init_mode, 'uniform',     seed=seed, dtype=dtype)

    return init

def dict_to_inits(dict_, seed=None, dtype=tf.float32):
    """ Accepts a dict of dicts, each of which contains a canonical config for an initializer. """

    inits = {k: dict_to_init(v, seed, dtype) for k, v in dict_.iteritems()}

    return inits
