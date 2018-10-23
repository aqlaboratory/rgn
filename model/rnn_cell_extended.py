"""Module for constructing RNN Cells.

## RNN Cell wrappers (RNNCells that wrap other RNNCells)

@@ZoneoutWrapper
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.rnn_cell import RNNCell, LSTMStateTuple

__author__ = "Mohammed AlQuraishi"
__copyright__ = "Copyright 2018, Harvard Medical School"
__license__ = "MIT"

class ZoneoutWrapper(RNNCell):
  """Operator adding zoneout to hidden state and memory of the given cell."""

  def __init__(self, cell, memory_cell_keep_prob=1.0, hidden_state_keep_prob=1.0,
               seed=None, is_training=True):
    """Create a cell with hidden state and memory zoneout.

    If this class is used to wrap a Dropout cell, then it will override the output 
    Dropout but maintain input Dropout. If a Dropout cell wraps a Zoneout cell,
    then both Dropout and Zoneout will be applied to the outputs.

    This function assumes that LSTM Cells are using the new tuple-based state.

    Args:
      cell: an BasicLSTMCell or LSTMCell
      memory_cell_keep_prob: unit Tensor or float between 0 and 1, memory cell
        keep probability; if it is float and 1, no zoneout will be added.
      hidden_state_keep_prob: unit Tensor or float between 0 and 1, hidden state
        keep probability; if it is float and 1, no zoneout will be added.
      seed: (optional) integer, the randomness seed.
      is_training: boolean, determines which mode of the zoneout is used.

    Raises:
      TypeError: if cell is not a BasicLSTMCell or LSTMCell.
      ValueError: if memory_cell_keep_prob or hidden_state_keep_prob is not between 0 and 1.
    """
    # if not (isinstance(cell, BasicLSTMCell) or isinstance(cell, LSTMCell)):
    #   raise TypeError("The parameter cell is not a BasicLSTMCell or LSTMCell.")
    if (isinstance(memory_cell_keep_prob, float) and
        not (memory_cell_keep_prob >= 0.0 and memory_cell_keep_prob <= 1.0)):
      raise ValueError("Parameter memory_cell_keep_prob must be between 0 and 1: %d"
                       % memory_cell_keep_prob)
    if (isinstance(hidden_state_keep_prob, float) and
        not (hidden_state_keep_prob >= 0.0 and hidden_state_keep_prob <= 1.0)):
      raise ValueError("Parameter hidden_state_keep_prob must be between 0 and 1: %d"
                       % hidden_state_keep_prob)
    self._cell = cell
    self._memory_cell_keep_prob = memory_cell_keep_prob    
    self._hidden_state_keep_prob = hidden_state_keep_prob
    self._seed = seed
    self._is_training = is_training

    self._has_memory_cell_zoneout  = (not isinstance(self._memory_cell_keep_prob, float) or
                                      self._memory_cell_keep_prob < 1)
    self._has_hidden_state_zoneout = (not isinstance(self._hidden_state_keep_prob, float) or
                                      self._hidden_state_keep_prob < 1)

  @property
  def input_size(self):
    return self._cell.input_size

  @property
  def output_size(self):
    return self._cell.output_size

  @property
  def state_size(self):
    return self._cell.state_size

  def __call__(self, inputs, state, scope=None):
    """Run the cell with the declared zoneouts."""

    # compute output and new state as before
    output, new_state = self._cell(inputs, state, scope)

    # if either hidden state or memory cell zoneout is applied, then split state and process
    if self._has_hidden_state_zoneout or self._has_memory_cell_zoneout:
      # split state
      c_old, m_old = state
      c_new, m_new = new_state

      # apply zoneout to memory cell and hidden state
      c_and_m = []
      for s_old, s_new, p, has_zoneout in [(c_old, c_new, self._memory_cell_keep_prob,  self._has_memory_cell_zoneout), 
                                           (m_old, m_new, self._hidden_state_keep_prob, self._has_hidden_state_zoneout)]:
        if has_zoneout:
          if self._is_training:
            mask = nn_ops.dropout(array_ops.ones_like(s_new), p, seed=self._seed) * p # this should just random ops instead. See dropout code for how.
            s = ((1. - mask) * s_old) + (mask * s_new)
          else:
            s = ((1. - p) * s_old) + (p * s_new)
        else:
          s = s_new

        c_and_m.append(s)

      # package final results
      new_state = LSTMStateTuple(*c_and_m)
      output = new_state.h

    return output, new_state
