from keras.layers import Conv2DTranspose
import keras.backend as K
import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations, constraints, initializers, regularizers
from tensorflow.python.keras.engine.base_layer import Layer, InputSpec

from tensorflow.python.keras.utils import conv_utils


class TiedConv2DTranspose(Conv2DTranspose):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 output_padding=None,
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 tied_to=None,
                 **kwargs):
        super(Conv2DTranspose, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs)
        self.tied_to = tied_to
        self.output_padding = output_padding
        if self.output_padding is not None:
            self.output_padding = conv_utils.normalize_tuple(
                self.output_padding, 2, 'output_padding')
            for stride, out_pad in zip(self.strides, self.output_padding):
                if out_pad >= stride:
                    raise ValueError('Stride ' + str(self.strides) + ' must be '
                                                                     'greater than output padding ' +
                                     str(self.output_padding))

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank 4. Received input '
                             'shape: ' + str(input_shape))
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        kernel_shape = self.kernel_size + (self.filters, input_dim)

        if self.tied_to is not None:
            self.kernel = self.tied_to.kernel
            self._non_trainable_weights.append(self.kernel)
        else:
            self.kernel = self.add_weight(
                name='kernel',
                shape=kernel_shape,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                trainable=True,
                dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        self.built = True


class DenseTied(Layer):
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 tied_to=None,
                 **kwargs):
        self.tied_to = tied_to
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super().__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        if self.tied_to is not None:
            self.kernel = K.transpose(self.tied_to.kernel)
            self._non_trainable_weights.append(self.kernel)
        else:
            self.kernel = self.add_weight(shape=(input_dim, self.units),
                                          initializer=self.kernel_initializer,
                                          name='kernel',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.built = True

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1] == self.units
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def call(self, inputs, **kwargs):
        output = K.dot(inputs, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output
