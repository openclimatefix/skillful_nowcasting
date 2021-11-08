"""Generator implementation."""

import functools
from . import discriminator
from . import latent_stack
from . import layers
import tensorflow.compat.v1 as tf


class Generator(object):
  """Generator for the proposed model."""

  def __init__(self, lead_time=90, time_delta=5):
    """Constructor.

    Args:
      lead_time: last lead time for the generator to predict. Default: 90 min.
      time_delta: time step between predictions. Default: 5 min.
    """
    self._cond_stack = ConditioningStack()
    self._sampler = Sampler(lead_time, time_delta)

  def __call__(self, inputs):
    """Connect to a graph.

    Args:
      inputs: a batch of inputs on the shape [batch_size, time, h, w, 1].
    Returns:
      predictions: a batch of predictions in the form
        [batch_size, num_lead_times, h, w, 1].
    """
    _, _, height, width, _ = inputs.shape.as_list()
    initial_states = self._cond_stack(inputs)
    predictions = self._sampler(initial_states, [height, width])
    return predictions

  def get_variables(self):
    """Get all variables of the module."""
    pass


class ConditioningStack(object):
  """Conditioning Stack for the Generator."""

  def __init__(self):
    self._block1 = discriminator.DBlock(output_channels=48, downsample=True)
    self._conv_mix1 = layers.SNConv2D(output_channels=48, kernel_size=3)
    self._block2 = discriminator.DBlock(output_channels=96, downsample=True)
    self._conv_mix2 = layers.SNConv2D(output_channels=96, kernel_size=3)
    self._block3 = discriminator.DBlock(output_channels=192, downsample=True)
    self._conv_mix3 = layers.SNConv2D(output_channels=192, kernel_size=3)
    self._block4 = discriminator.DBlock(output_channels=384, downsample=True)
    self._conv_mix4 = layers.SNConv2D(output_channels=384, kernel_size=3)

  def __call__(self, inputs):
    # Space to depth conversion of 256x256x1 radar to 128x128x4 hiddens.
    h0 = batch_apply(
        functools.partial(tf.nn.space_to_depth, block_size=2), inputs)

    # Downsampling residual D Blocks.
    h1 = time_apply(self._block1, h0)
    h2 = time_apply(self._block2, h1)
    h3 = time_apply(self._block3, h2)
    h4 = time_apply(self._block4, h3)

    # Spectrally normalized convolutions, followed by rectified linear units.
    init_state_1 = self._mixing_layer(h1, self._conv_mix1)
    init_state_2 = self._mixing_layer(h2, self._conv_mix2)
    init_state_3 = self._mixing_layer(h3, self._conv_mix3)
    init_state_4 = self._mixing_layer(h4, self._conv_mix4)

    # Return a stack of conditioning representations of size 64x64x48, 32x32x96,
    # 16x16x192 and 8x8x384.
    return init_state_1, init_state_2, init_state_3, init_state_4

  def _mixing_layer(self, inputs, conv_block):
    # Convert from [batch_size, time, h, w, c] -> [batch_size, h, w, c * time]
    # then perform convolution on the output while preserving number of c.
    stacked_inputs = tf.concat(tf.unstack(inputs, axis=1), axis=-1)
    return tf.nn.relu(conv_block(stacked_inputs))


class Sampler(object):
  """Sampler for the Generator."""

  def __init__(self, lead_time=90, time_delta=5):
    self._num_predictions = lead_time // time_delta
    self._latent_stack = latent_stack.LatentCondStack()

    self._conv_gru4 = ConvGRU()
    self._conv4 = layers.SNConv2D(kernel_size=1, output_channels=768)
    self._gblock4 = GBlock(output_channels=768)
    self._g_up_block4 = UpsampleGBlock(output_channels=384)

    self._conv_gru3 = ConvGRU()
    self._conv3 = layers.SNConv2D(kernel_size=1, output_channels=384)
    self._gblock3 = GBlock(output_channels=384)
    self._g_up_block3 = UpsampleGBlock(output_channels=192)

    self._conv_gru2 = ConvGRU()
    self._conv2 = layers.SNConv2D(kernel_size=1, output_channels=192)
    self._gblock2 = GBlock(output_channels=192)
    self._g_up_block2 = GBlock(output_channels=96)

    self._conv_gru1 = ConvGRU()
    self._conv1 = layers.SNConv2D(kernel_size=1, output_channels=96)
    self._gblock1 = GBlock(output_channels=96)
    self._g_up_block1 = UpsampleGBlock(output_channels=48)

    self._bn = layers.BatchNorm()
    self._output_conv = layers.SNConv2D(kernel_size=1, output_channels=4)

  def __call__(self, initial_states, resolution):
    init_state_1, init_state_2, init_state_3, init_state_4 = initial_states
    batch_size = init_state_1.shape.as_list()[0]

    # Latent conditioning stack.
    z = self._latent_stack(batch_size, resolution)
    hs = [z] * self._num_predictions

    # Layer 4 (bottom-most).
    hs, _ = tf.nn.static_rnn(self._conv_gru4, hs, init_state_4)
    hs = [self._conv4(h) for h in hs]
    hs = [self._gblock4(h) for h in hs]
    hs = [self._g_up_block4(h) for h in hs]

    # Layer 3.
    hs, _ = tf.nn.static_rnn(self._conv_gru3, hs, init_state_3)
    hs = [self._conv3(h) for h in hs]
    hs = [self._gblock3(h) for h in hs]
    hs = [self._g_up_block3(h) for h in hs]

    # Layer 2.
    hs, _ = tf.nn.static_rnn(self._conv_gru2, hs, init_state_2)
    hs = [self._conv2(h) for h in hs]
    hs = [self._gblock2(h) for h in hs]
    hs = [self._g_up_block2(h) for h in hs]

    # Layer 1 (top-most).
    hs, _ = tf.nn.static_rnn(self._conv_gru1, hs, init_state_1)
    hs = [self._conv1(h) for h in hs]
    hs = [self._gblock1(h) for h in hs]
    hs = [self._g_up_block1(h) for h in hs]

    # Output layer.
    hs = [tf.nn.relu(self._bn(h)) for h in hs]
    hs = [self._output_conv(h) for h in hs]
    hs = [tf.nn.depth_to_space(h, 2) for h in hs]

    return tf.stack(hs, axis=1)


class GBlock(object):
  """Residual generator block without upsampling."""

  def __init__(self, output_channels, sn_eps=0.0001):
    self._conv1_3x3 = layers.SNConv2D(
        output_channels, kernel_size=3, sn_eps=sn_eps)
    self._bn1 = layers.BatchNorm()
    self._conv2_3x3 = layers.SNConv2D(
        output_channels, kernel_size=3, sn_eps=sn_eps)
    self._bn2 = layers.BatchNorm()
    self._output_channels = output_channels
    self._sn_eps = sn_eps

  def __call__(self, inputs):
    input_channels = inputs.shape[-1]

    # Optional spectrally normalized 1x1 convolution.
    if input_channels != self._output_channels:
      conv_1x1 = layers.SNConv2D(
          self._output_channels, kernel_size=1, sn_eps=self._sn_eps)
      sc = conv_1x1(inputs)
    else:
      sc = inputs

    # Two-layer residual connection, with batch normalization, nonlinearity and
    # 3x3 spectrally normalized convolution in each layer.
    h = tf.nn.relu(self._bn1(inputs))
    h = self._conv1_3x3(h)
    h = tf.nn.relu(self._bn2(h))
    h = self._conv2_3x3(h)

    # Residual connection.
    return h + sc


class UpsampleGBlock(object):
  """Upsampling residual generator block."""

  def __init__(self, output_channels, sn_eps=0.0001):
    self._conv_1x1 = layers.SNConv2D(
        output_channels, kernel_size=1, sn_eps=sn_eps)
    self._conv1_3x3 = layers.SNConv2D(
        output_channels, kernel_size=3, sn_eps=sn_eps)
    self._bn1 = layers.BatchNorm()
    self._conv2_3x3 = layers.SNConv2D(
        output_channels, kernel_size=3, sn_eps=sn_eps)
    self._bn2 = layers.BatchNorm()
    self._output_channels = output_channels

  def __call__(self, inputs):
    # x2 upsampling and spectrally normalized 1x1 convolution.
    sc = layers.upsample_nearest_neighbor(inputs, upsample_size=2)
    sc = self._conv_1x1(sc)

    # Two-layer residual connection, with batch normalization, nonlinearity and
    # 3x3 spectrally normalized convolution in each layer, and x2 upsampling in
    # the first layer.
    h = tf.nn.relu(self._bn1(inputs))
    h = layers.upsample_nearest_neighbor(h, upsample_size=2)
    h = self._conv1_3x3(h)
    h = tf.nn.relu(self._bn2(h))
    h = self._conv2_3x3(h)

    # Residual connection.
    return h + sc


class ConvGRU(object):
  """A ConvGRU implementation."""

  def __init__(self, kernel_size=3, sn_eps=0.0001):
    """Constructor.

    Args:
      kernel_size: kernel size of the convolutions. Default: 3.
      sn_eps: constant for spectral normalization. Default: 1e-4.
    """
    self._kernel_size = kernel_size
    self._sn_eps = sn_eps

  def __call__(self, inputs, prev_state):

    # Concatenate the inputs and previous state along the channel axis.
    num_channels = prev_state.shape[-1]
    xh = tf.concat([inputs, prev_state], axis=-1)

    # Read gate of the GRU.
    read_gate_conv = layers.SNConv2D(
        num_channels, self._kernel_size, sn_eps=self._sn_eps)
    read_gate = tf.math.sigmoid(read_gate_conv(xh))

    # Update gate of the GRU.
    update_gate_conv = layers.SNConv2D(
        num_channels, self._kernel_size, sn_eps=self._sn_eps)
    update_gate = tf.math.sigmoid(update_gate_conv(xh))

    # Gate the inputs.
    gated_input = tf.concat([inputs, read_gate * prev_state], axis=-1)

    # Gate the cell and state / outputs.
    output_conv = layers.SNConv2D(
        num_channels, self._kernel_size, sn_eps=self._sn_eps)
    c = tf.nn.relu(output_conv(gated_input))
    out = update_gate * prev_state + (1. - update_gate) * c
    new_state = out

    return out, new_state


def time_apply(func, inputs):
  """Apply function func on each element of inputs along the time axis."""
  return layers.ApplyAlongAxis(func, axis=1)(inputs)


def batch_apply(func, inputs):
  """Apply function func on each element of inputs along the batch axis."""
  return layers.ApplyAlongAxis(func, axis=0)(inputs)

