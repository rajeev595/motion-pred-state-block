# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 14:12:25 2018

@author: rajeev
"""

class EncoderConv2DWrapper(RNNCell):
  """ A class for adding a convolution on the 4-frams input given to an RNN cell. """

  def __init__(self, cell, output_size):
    """Create a cell with with a conv2D encoder in space.

    Args:
      cell: an RNNCell. The input is passed through a conv2d layer.

    Raises:
      TypeError: if cell is not an RNNCell.
    """
    if not isinstance(cell, RNNCell):
      raise TypeError("The parameter cell is not a RNNCell.")

    self._cell = cell

    print( 'output_size = {0}'.format(output_size) )
    print( ' state_size = {0}'.format(self._cell.state_size) )

    # Tuple if multi-rnn
    if isinstance(self._cell.state_size,tuple):

      # Fine if GRU...
      insize = self._cell.state_size[-1]

      # LSTMStateTuple if LSTM
      if isinstance( insize, LSTMStateTuple ):
        insize = insize.h

    else:
      # Fine if not multi-rnn
      insize = self._cell.state_size

    self.W_kernel = tf.get_variable("W_kernel",
        [18, 3, output_size],
        dtype = tf.float32,
        initializer = tf.contrib.layers.xavier_initializer())

    self.conv2d_output_size = output_size

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self.linear_output_size