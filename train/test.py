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
"""Functions for testing trained keyword spotting models from checkpoint files."""

import os
import tempfile
import argparse

import numpy as np
import tensorflow as tf

import data
import models


def test(model, audio_processor, model_settings):
    """Calculate accuracy and confusion matrices on validation and test sets.

    Model is created and weights loaded from supplied command line arguments.
    """
    model.load_weights(FLAGS.checkpoint).expect_partial()
    if FLAGS.mode == "test":
        mode = audio_processor.Modes.TESTING
    elif FLAGS.mode == "validation":
        mode = audio_processor.Modes.VALIDATION
    elif FLAGS.mode == "train":
        mode = audio_processor.Modes.TRAINING
    else:
        raise RuntimeError(f"Unsupported mode: {mode}")

    data = audio_processor.get_data(mode).batch(FLAGS.batch_size)
    expected_indices = np.concatenate([y for x, y in data])

    predictions = model.predict(data)
    predicted_indices = tf.argmax(predictions, axis=1)

    print(f"Running testing on {FLAGS.mode} set...")
    accuracy = get_accuracy(expected_indices, predicted_indices)
    print(f"{mode} accuracy = {accuracy * 100:.2f}%(N={audio_processor.set_size(mode)})")
    print()

    confusion_matrix = get_confusion_matrix(expected_indices, predicted_indices, model_settings)
    print("confusion matrix:")
    print(confusion_matrix.numpy())


def get_confusion_matrix(expected_indices, predicted_indices, model_settings):
    confusion_matrix = tf.math.confusion_matrix(
        expected_indices,
        predicted_indices,
        num_classes=model_settings["label_count"],
    )
    return confusion_matrix


def get_accuracy(predicted_indices, expected_indices):
    """Calculates and returns accuracy.

    Args:
        predicted_indices: List of predicted integer indices.
        expected_indices: List of expected integer indices.

    Returns:
        Accuracy value between 0 and 1.
    """
    correct_prediction = tf.equal(predicted_indices, expected_indices)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_url",
        type=str,
        default="http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
        help="Location of speech training data archive on the web.",
    )
    try:
        login = os.getlogin()
    except:
        login = "unknown"
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.getenv(
            "SPEECH_COMMANDS_DIR",
            default=os.path.join(tempfile.gettempdir(), login, "speech_dataset"),
        ),
        help="""\
        Where to download the speech training data to.
        """,
    )
    parser.add_argument(
        "--silence_percentage",
        type=float,
        default=None,
        help="""\
        How much of the training data should be silence.
        """,
    )
    parser.add_argument(
        "--unknown_percentage",
        type=float,
        default=None,
        help="""\
        How much of the training data should be unknown words.
        """,
    )
    parser.add_argument(
        "--testing_percentage",
        type=int,
        default=10,
        help="What percentage of wavs to use as a test set.",
    )
    parser.add_argument(
        "--validation_percentage",
        type=int,
        default=10,
        help="What percentage of wavs to use as a validation set.",
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
        "--batch_size",
        type=int,
        default=100,
        help="How many items to train with at once",
    )
    parser.add_argument(
        "--wanted_words",
        type=str,
        default="yes,no,up,down,left,right,on,off,stop,go",
        help="Words to use (others will be added to an unknown label)",
    )
    parser.add_argument("--checkpoint", type=str, help="Checkpoint to load the weights from.")
    parser.add_argument(
        "--model_architecture",
        type=str,
        default="dnn",
        help="What model architecture to use",
    )
    parser.add_argument(
        "--model_size_info",
        type=int,
        nargs="+",
        default=[128, 128, 128],
        help="Model dimensions - different for various models",
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
    parser.add_argument("--mode", choices=["test", "validate", "train"], default="test")

    FLAGS, _ = parser.parse_known_args()

    model_settings = models.prepare_model_settings(
        len(data.prepare_words_list(FLAGS.wanted_words.split(","))),
        FLAGS.sample_rate,
        FLAGS.clip_duration_ms,
        FLAGS.window_size_ms,
        FLAGS.window_stride_ms,
        FLAGS.dct_coefficient_count,
    )

    model = models.get_model(model_settings, FLAGS.model_architecture)

    num_classes = len(FLAGS.wanted_words.split(",")) + 2

    if FLAGS.silence_percentage is None:
        FLAGS.silence_percentage = 100.0 / num_classes

    if FLAGS.unknown_percentage is None:
        FLAGS.unknown_percentage = 100.0 / num_classes

    audio_processor = data.AudioProcessor(
        data_url=FLAGS.data_url,
        data_dir=FLAGS.data_dir,
        silence_percentage=FLAGS.silence_percentage,
        unknown_percentage=FLAGS.unknown_percentage,
        wanted_words=FLAGS.wanted_words.split(","),
        validation_percentage=FLAGS.validation_percentage,
        testing_percentage=FLAGS.testing_percentage,
        model_settings=model_settings,
        micro=FLAGS.micro,
    )

    test(model, audio_processor, model_settings)
