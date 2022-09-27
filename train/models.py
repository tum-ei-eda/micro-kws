# Copyright © 2021 Arm Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications Copyright 2022 Chair of Electronic Design Automation, TUM
"""Model definitions for simple keyword spotting."""

import math

import tensorflow as tf


def prepare_model_settings(
    label_count,
    sample_rate,
    clip_duration_ms,
    window_size_ms,
    window_stride_ms,
    dct_coefficient_count,
):
    """Calculates common settings needed for all models.

    Args:
        label_count: How many classes are to be recognized.
        sample_rate: Number of audio samples per second.
        clip_duration_ms: Length of each audio clip to be analyzed.
        window_size_ms: Duration of frequency analysis window.
        window_stride_ms: How far to move in time between frequency windows.
        dct_coefficient_count: Number of frequency bins to use for analysis.

    Returns:
        Dictionary containing common settings.
    """
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)
    length_minus_window = desired_samples - window_size_samples
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
    fingerprint_size = dct_coefficient_count * spectrogram_length

    return {
        "desired_samples": desired_samples,
        "window_size_samples": window_size_samples,
        "window_stride_samples": window_stride_samples,
        "spectrogram_length": spectrogram_length,
        "dct_coefficient_count": dct_coefficient_count,
        "fingerprint_size": fingerprint_size,
        "label_count": label_count,
        "sample_rate": sample_rate,
    }


def get_model(model_settings, model_architecture, model_size_info, is_training, model_name="kws_model"):
    """Builds a tf.keras model of the requested architecture compatible with the settings.

    Args:
        model_settings: Dictionary of information about the model.
        model_architecture: String specifying which kind of model to create.

    Returns:
        A tf.keras Model with the requested architecture.

    Raises:
        Exception: If the architecture type isn't recognized.
    """

    if model_architecture == "micro_speech":
        return create_micro_speech_model(model_settings, model_name=model_name)
    elif model_architecture == "custom":
        return create_custom_model(model_settings, model_name=model_name)
    elif model_architecture == "custom2":
        return create_custom2_model(model_settings, model_name=model_name)
    elif model_architecture == 'ds_cnn':
        return create_ds_cnn_model(model_settings, model_size_info)
    else:
        raise Exception(f"model_architecture argument {model_architecture} not recognized")

def create_ds_cnn_model(model_settings, model_size_info):
    """Builds a model with convolutional & depthwise separable convolutional layers.

    For more details see https://arxiv.org/abs/1711.07128.

    Args:
        model_settings: Dict of different settings for model training.
        model_size_info: Defines number of layers, followed by the DS-Conv layer
            parameters in the order {number of conv features, conv filter height,
            width and stride in y,x dir.} for each of the layers.

    Returns:
        tf.keras Model of the 'DS-CNN' architecture.
    """

    label_count = model_settings['label_count']
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']

    t_dim = input_time_size
    f_dim = input_frequency_size

    # Extract model dimensions from model_size_info.
    num_layers = model_size_info[0]
    conv_feat = [None]*num_layers
    conv_kt = [None]*num_layers
    conv_kf = [None]*num_layers
    conv_st = [None]*num_layers
    conv_sf = [None]*num_layers

    i = 1
    for layer_no in range(0, num_layers):
        conv_feat[layer_no] = model_size_info[i]
        i += 1
        conv_kt[layer_no] = model_size_info[i]
        i += 1
        conv_kf[layer_no] = model_size_info[i]
        i += 1
        conv_st[layer_no] = model_size_info[i]
        i += 1
        conv_sf[layer_no] = model_size_info[i]
        i += 1

    inputs = tf.keras.Input(shape=(model_settings['fingerprint_size']), name='input')

    # Reshape the flattened input.
    x = tf.reshape(inputs, shape=(-1, input_time_size, input_frequency_size, 1))

    # Depthwise separable convolutions.
    for layer_no in range(0, num_layers):
        if layer_no == 0:
            # First convolution.
            x = tf.keras.layers.Conv2D(filters=conv_feat[0],
                                       kernel_size=(conv_kt[0], conv_kf[0]),
                                       strides=(conv_st[0], conv_sf[0]),
                                       padding='SAME')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
        else:
            # Depthwise convolution.
            x = tf.keras.layers.DepthwiseConv2D(kernel_size=(conv_kt[layer_no], conv_kf[layer_no]),
                                                strides=(conv_sf[layer_no], conv_st[layer_no]),
                                                padding='SAME')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)

            # Pointwise convolution.
            x = tf.keras.layers.Conv2D(filters=conv_feat[layer_no], kernel_size=(1, 1))(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)

        t_dim = math.ceil(t_dim/float(conv_st[layer_no]))
        f_dim = math.ceil(f_dim/float(conv_sf[layer_no]))

    # Global average pool.
    x = tf.keras.layers.AveragePooling2D(pool_size=(t_dim, f_dim), strides=1)(x)

    # Squeeze before passing to output fully connected layer.
    x = tf.reshape(x, shape=(-1, conv_feat[layer_no]))

    # Output connected layer.
    output = tf.keras.layers.Dense(units=label_count, activation='softmax')(x)

    return tf.keras.Model(inputs, output)

def create_micro_speech_model(model_settings, model_name="kws_model"):
    """Builds a model with a single depthwise-convolution layer followed by a single fully-connected layer.
    Args:
        model_settings: Dict of different settings for model training.
    Returns:
        tf.keras Model of the 'CNN' architecture.
    """

    # Get relevant model setting.
    input_frequency_size = model_settings["dct_coefficient_count"]
    input_time_size = model_settings["spectrogram_length"]

    ### Task X: REPLACE CODE BELOW ###

    inputs = tf.keras.Input(shape=(model_settings["fingerprint_size"]), name="input")

    # Reshape the flattened input.
    x = tf.reshape(inputs, shape=(-1, input_time_size, input_frequency_size, 1))

    # First convolution.
    x = tf.keras.layers.DepthwiseConv2D(
        depth_multiplier=8,
        # kernel_size=(10, 8),
        kernel_size=(10, 8),
        strides=(2, 2),
        padding="SAME",
        activation="relu",
    )(x)

    # # Second convolution.
    # y = tf.keras.layers.DepthwiseConv2D(
    #     depth_multiplier=8,
    #     # kernel_size=(10, 8),
    #     kernel_size=(10, 8),
    #     strides=(2, 2),
    #     padding="SAME",
    #     activation="relu",
    # )(x)

    # Flatten for fully connected layers.
    x = tf.keras.layers.Flatten()(x)

    # Output fully connected.
    output = tf.keras.layers.Dense(units=model_settings["label_count"], activation="softmax")(x)

    ### Task X: REPLACE CODE ABOVE ###

    return tf.keras.Model(inputs, output)


def create_custom_model(model_settings, model_name="kws_model"):

    # Get relevant model setting.
    input_frequency_size = model_settings["dct_coefficient_count"]
    input_time_size = model_settings["spectrogram_length"]

    inputs = tf.keras.Input(shape=(model_settings["fingerprint_size"]), name="input")

    # Reshape the flattened input.
    x = tf.reshape(inputs, shape=(-1, input_time_size, input_frequency_size, 1))

    # First convolution.
    x = tf.keras.layers.DepthwiseConv2D(
        depth_multiplier=8,
        kernel_size=(20, 8),
        strides=(1, 1),
        padding="SAME",
        activation="relu",
    )(x)

    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
    )(x)

    # Second convolution.
    x = tf.keras.layers.Conv2D(
        filters=8,
        kernel_size=(10, 4),
        strides=(1, 1),
        padding="SAME",
        activation="relu",
    )(x)

    # Flatten for fully connected layers.
    x = tf.keras.layers.Flatten()(x)

    # Output fully connected.
    output = tf.keras.layers.Dense(units=model_settings["label_count"], activation="softmax")(x)

    ### Task X: REPLACE CODE ABOVE ###

    return tf.keras.Model(inputs, output, name=model_name)


def create_custom2_model(model_settings, model_name="kws_model"):

    # Get relevant model setting.
    input_frequency_size = model_settings["dct_coefficient_count"]
    input_time_size = model_settings["spectrogram_length"]

    inputs = tf.keras.Input(shape=(model_settings["fingerprint_size"]), name="input")

    # Reshape the flattened input.
    x = tf.reshape(inputs, shape=(-1, input_time_size, input_frequency_size, 1))

    # First convolution.
    x = tf.keras.layers.DepthwiseConv2D(
        depth_multiplier=16,
        kernel_size=(20, 8),
        strides=(1, 1),
        padding="SAME",
        activation="relu",
    )(x)

    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
    )(x)

    # Second convolution.
    x = tf.keras.layers.Conv2D(
        # filters=8,
        filters=16,
        kernel_size=(10, 4),
        strides=(1, 1),
        padding="SAME",
        activation="relu",
    )(x)

    # Flatten for fully connected layers.
    x = tf.keras.layers.Flatten()(x)

    # Output fully connected.
    output = tf.keras.layers.Dense(units=model_settings["label_count"], activation="softmax")(x)

    ### Task X: REPLACE CODE ABOVE ###

    return tf.keras.Model(inputs, output, name=model_name)
