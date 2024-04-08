#
# Copyright (c) 2022 TUM Department of Electrical and Computer Engineering.
#
# This file is part of MLonMCU.
# See https://github.com/tum-ei-eda/mlonmcu.git for further info.
#
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
r"""Converts WAV audio files into input features for neural networks."""

import argparse
import os.path
import sys
import struct
import numpy as np

import models
import data

from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()


def write_output_file(dest, features, fmt="bin", quantize=False):
    if quantize:
        features = (features.astype(np.int16) - 128).astype(np.int8)

    input_name = "serving_default_input:0"

    if fmt == "bin":
        data = b""
        for f in features:
            data += struct.pack("b" if quantize else "f", f)
        with open(dest, "wb") as handle:
            handle.write(data)
    elif fmt == "npy":
        np.save(dest, **{input_name: features[np.newaxis, :]})
    elif fmt == "npz":
        np.savez(dest, **{input_name: features[np.newaxis, :]})
    else:
        raise RuntimeError("Invalid output format")


def get_features_range(micro=False):
    """Returns the expected min/max for generated features.
    Args:
      If the micro preprocess is used.
    Returns:
      Min/max float pair holding the range of features.
    Raises:
      Exception: If preprocessing mode isn't recognized.
    """
    if micro:
        features_min = 0.0
        features_max = 26.0
    else:
        raise NotImplementedError

    return features_min, features_max


def wav_to_features(audio_processor, input_wav, micro=False, quantize=False):
    """Converts an audio file into its corresponding feature map.
    Args:
      sample_rate: Expected sample rate of the wavs.
      clip_duration_ms: Expected duration in milliseconds of the wavs.
      window_size_ms: How long each spectrogram timeslice is.
      window_stride_ms: How far to move in time between spectrogram timeslices.
      feature_bin_count: How many bins to use for the feature fingerprint.
      quantize: Whether to train the model for eight-bit deployment.
      input_wav: Path to the audio WAV file to read.
      micro: If the micro preprocess is used.
    """

    features, _ = audio_processor.process_path(
        input_wav,
        None,
        model_settings,
        0,
        0,
        0,
        False,
        None,
        micro=micro,
    )
    features = features.numpy()

    variable_base = os.path.splitext(os.path.basename(input_wav).lower())[0]

    if quantize:
        features_min, features_max = get_features_range(micro=micro)
        for i, value in enumerate(features):
            quantized_value = int(
                round((255 * (value - features_min)) / (features_max - features_min))
            )
            if quantized_value < 0:
                quantized_value = 0
            if quantized_value > 255:
                quantized_value = 255
            features[i] = quantized_value
        features = features.astype(np.uint8)

    return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_url",
        type=str,
        default="http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
        help="Location of speech training data archive on the web.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/tmp/speech_dataset/",
        help="""\
        Where to download the speech training data to.
        """,
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Expected sample rate of the wavs",
    )
    parser.add_argument(
        "--clip_duration_ms",
        type=int,
        default=1000,
        help="Expected duration in milliseconds of the wavs",
    )
    parser.add_argument(
        "--window_size_ms",
        type=float,
        default=30.0,
        help="How long each spectrogram timeslice is",
    )
    parser.add_argument(
        "--window_stride_ms",
        type=float,
        default=20.0,
        help="How long each spectrogram timeslice is",
    )
    parser.add_argument(
        "--dct_coefficient_count",
        type=int,
        default=40,
        help="How many bins to use for the MFCC fingerprint",
    )
    parser.add_argument(
        "--quantize",
        dest="quantize",
        action="store_true",
        default=True,
        help="Whether to quantize the model or convert to fp32 TFLite model. Defaults to True.",
    )
    parser.add_argument(
        "--no-quantize",
        dest="quantize",
        action="store_false",
    )
    parser.add_argument(
        "--output-format",
        "-f",
        type=str,
        choices=["bin", "npz", "npy"],
        default="bin",
        help="The file format of the output",
    )
    parser.add_argument(
        "--micro",
        dest="micro",
        action="store_true",
        default=True,
        help="Use micro preprocess. Defaults to True.",
    )
    parser.add_argument(
        "--no-micro",
        dest="micro",
        action="store_false",
    )
    parser.add_argument("WAV_FILE", type=str, help="Path to the audio WAV file to read")
    parser.add_argument("OUT_FILE", type=str, help="Path to the audio WAV file to read")

    FLAGS, _ = parser.parse_known_args()

    model_settings = models.prepare_model_settings(
        0,
        FLAGS.sample_rate,
        FLAGS.clip_duration_ms,
        FLAGS.window_size_ms,
        FLAGS.window_stride_ms,
        FLAGS.dct_coefficient_count,
    )

    audio_processor = data.AudioProcessor(
        data_url=FLAGS.data_url,
        data_dir=FLAGS.data_dir,
        silence_percentage=0.1,
        unknown_percentage=0.1,
        wanted_words=["yes", "no"],
        validation_percentage=0.1,
        testing_percentage=0.1,
        model_settings=model_settings,
        micro=FLAGS.micro,
        minimal=True,
    )

    features = wav_to_features(
        audio_processor, FLAGS.WAV_FILE, micro=FLAGS.micro, quantize=FLAGS.quantize
    )

    write_output_file(
        FLAGS.OUT_FILE,
        features,
        fmt=FLAGS.output_format,
        quantize=FLAGS.quantize,
    )

    print(f"Output file written to {FLAGS.OUT_FILE}")
