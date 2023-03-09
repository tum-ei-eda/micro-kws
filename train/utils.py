# Copyright 2023 Chair of Electronic Design Automation, TUM. All Rights Reserved.
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
# ==============================================================================

import sys
import os
import wave
import netron
import numpy as np
import matplotlib.pyplot as plt
import IPython
from IPython.display import IFrame
import portpicker


def isColab():
    try:
      import google.colab
      return True
    except:
      return False


def showNetron(path):
    port = portpicker.pick_unused_port()
    if isColab():
        from google.colab import output
        # Read the model file and start the netron browser.
        with output.temporary():
          netron.start(path, port, browse=False)
        return output.serve_kernel_port_as_iframe(port, height='800')
    else:
        netron.start(path, port, browse=False)
        return IFrame(src=f"http://localhost:{port}/", width="100%", height=800)


def playAudio(path):
    return IPython.display.Audio(path)


def showWaveform(path, title="Sample"):
  spf = wave.open(path, "r")

  # Extract Raw Audio from Wav File
  signal = spf.readframes(-1)
  signal = np.frombuffer(signal, "int16")

  # If Stereo
  if spf.getnchannels() == 2:
      print("Just mono files")
      sys.exit(0)

  plt.figure(1)
  plt.title(title)
  fr = spf.getframerate()
  x = list(range(0, fr))
  x = [x_/fr for x_ in x]
  plt.plot(x, signal)
  plt.xlabel("Time [s]")
  plt.ylabel("Amplitude")
  plt.show()


def visualizeFeature(feature):
    # Utility to display a given feature from the dataset inside the notebook
    feature_data, feature_label = feature

    feature_data = feature_data.numpy()
    feature_label = feature_label.numpy()

    feature_label_str = (["silence", "unknown"] + FLAGS.wanted_words.split(","))[feature_label]

    feature_reshaped = np.reshape(feature_data, (49,40)).T

    p = plt.imshow(feature_reshaped, cmap='gray', vmin=0, vmax=26)
    plt.title(f"Label: {feature_label_str}")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [ƒ]")
