# Micro KWS Debugger
This small Python program is a debugging utility for the microKWS target software. It receives and visualizes data coming from the ESP32-C3 microcontroller via the USB serial connection.

**Disclaimer:** So far, this software is only tested on Ubuntu using Python 3.8! So we recommend that you also use Linux and Python 3.8.

## Prerequisites
- Open a **new** terminal with a clean shell environment (e.g. no ESP-IDF activated) and navigate into this directory
- Create a **new** Python virtual environment `virtualenv -p python3.8 venv`
- Activate the virtual environment `. ./venv/bin/activate`
- Install the dependencies `pip install -r requirements.txt`
- Launch the application `python debug.py`

In order to play back the recorded wav files, you will need an audio playback tool. For some reason, not all programs work. We recommend either Audacity `sudo apt install audacity` or SOund eXchanger `sudo apt install sox`.

## Commandline Arguments
By running `python debug.py --help`, you can display all available command-line options. Here are the most important ones that you will likely need to change:

- `-p, --port` takes a path to the USB serial device (the ESP32-C3). If it's not the default value, i.e. `/dev/ttyUSB0`, it will most likely be either `/dev/ttyUSB1` or `/dev/ttyUSB2`, depending on how many other devices you have connected. Make sure that you select the correct one corresponding to your ESP32-C3.
- `-cl, --category-labels` takes the category labels that your model can recognize as a list, excluding the first two standard ones, `silence` and `unknown`. The default categories are `yes no` (notice the space between every label).
- `-a, --audio` this flag activates the microphone debugger mode, with which you can record and save audio coming from the microphone to your computer. See the explanation below.

So if you wanted to run the debugger on `ttyUSB1` with the labels `up`, `down`, `left` and `right`, your command would look like this:
```
python debug.py --port /dev/ttyUSB0 --category-labels up down left right
```

## Operating Modes
There are three different modes the ESP32-C3 target software can operate in. Two of them can be used in conjunction with this debugging tool, as they stream special binary debugging data to the PC.

You can change the behaviour of the ESP32-C3 target software and enable debugging modes through the `idf.py menuconfig` command. Simply run this command in the `target/` directory and you should see a menu appear. From there, navigate to
```
(Top) → MicroKWS Options → MicroKWS Debug Settings
```
to see all available debugging related options of the target software. Changing things there will include or exclude certain compiler flags, which are then in turn used by `#ifdef` statements in the target software code to determine desired operating behaviour. This also means that you will need to recompile and upload the software for any changes to take effect.

### 1) Normal Operation
If you do not want to use the debugger, you do not need to change anything in menuconfig. By default, the inference results coming from the neural network model and the time between inferences should get printed in ASCII format via the serial interface. The debugging Python tool is of no need here. You should be able to observe the prints by running `idf.py monitor` in your terminal.

### 2) Debug Operation
In this mode, the ESP32-C3 is transmitting binary debugging data via the serial interface to the computer. In order for this mode to be activated in the target software, you will need to do the following:

1) In the menuconfig navigate to
```
(Top) → MicroKWS Options → MicroKWS Debug Settings
```
and choose mode `Debug: Stream data to debugger GUI` option in the `Mode for MicroKWS Execution` submenu. This will send the binary debugging data over the serial USB connection. However, by default, all print and debug statements in the target software are also forwarded and transmitted over this serial connection. They would interfere with our binary debugging data. Thus, we will need to disable the forwarding of print statements.

2) For this, navigate to
```
(Top) → Component config → ESP System Settings → Channel for console output
```
and change the setting from `Default: UART0` to `None`. Save the setting, recompile and upload the code to ESP32-C3. Now, the USB serial connection is solely being used by the binary debugging data stream.

When you now run the Python debugger, you should see the live data being displayed in the GUI.

### 3) Microphone Debugger Mode
This is the third of the three operating modes. It starts the ESP32-C3 in a special routine that records audio data on the device and then transmits it to your computer. In order to activate this mode on the ESP32-C3, navigate to
```
(Top) → MicroKWS Options → MicroKWS Debug Settings
```
in the menuconfig and choose the mode `Audio: Record audio on device and send it to host client` option in the `Mode for MicroKWS Execution` submenu. *Similarly to the other debug mode, you will also need to disable the serial console output, see step 2. above).* Save, recompile and upload the code to ESP32-C3. You should now see the RGB LED changing in colour.   
Let's quickly look at what is happening:
- Once the ESP32-C3 starts, it enters standby mode for approximately 2s. During this time, the RGB is *yellow*.
- After this, it switches to *orange* for about 1s. During this time, it is clearing its internal buffers, and you should get ready to speak.
- Once the LED turns *red*, the ESP32-C3 starts recording audio to an internal buffer. You can now speak and record audio for about 2s.
- After this, the LED turns back to *yellow* to indicate that the recording has stopped. The ESP32-C3 will now start transmitting the audio data in small packets over the serial connection. *It does not care if someone is actually listening on the other side; it simply sends data.*
- Once this is done, the LED turns *green* and the ESP32-C3 enters an endless loop. In order to restart the recording, simply press the reset button on the development board.

Now, in order to capture the audio data, you will need to start the debugger in audio capture mode, aka. "microphone debug" mode. For this, supply the `-a, --audio` flag the debugger. Make sure to start the debugger before you start the audio recording procedure on the ESP32-C3. The Python debugger will wait for the ESP32-C3 data to come in, but the ESP32-C3 won't wait for the debugger to listen. We recommend that you follow the following procedure:
1) Upload the target software to the ESP32-C3 with the microphone debugger mode enabled.
2) Press and hold the reset button on the ESP32-C3 until **step 4**.
3) Start the Python debugger on your PC with the `--audio` flag enabled. You should see it periodically print `Received 0 bytes. Waiting for audio data...`.
4) Release the reset button on the ESP32-C3 and follow the LED guided recording procedure.
5) After recording, the Python tool should inform you about the data packets it is receiving `Received 3200 bytes, 60800 bytes remaining...`
6) Either when all data is received or when no new data has been transmitted for a certain amount of time, the debugging tool will finish the transmission and save the data into a file `Received only 60800 of total 64000 bytes. But quit receiving prematurely due to time reasons.
Wrote 60800 bytes to audio_test.wav`
7) You can now listen to this recording with either Audacity (open a file explorer, navigate to the `audio_test.wav` file and *right click* "open with", select Audacity), or the command line tool sox: `play --volume 10 audio_test.wav`.

This should help you get a feeling for the audio quality and whether the microphone setup is working.  

## GUI Explained
In the top-left corner of the debugger GUI, you have a live view of the feature matrix that the convolutional neural network (CNN) uses for keyword detecting. This is the output of the feature-extraction frontend. In the top-right corner, you can see the current posterior values output by the CNN as a bar plot. This graph visualizes the instantaneous result, while the large history plot in the lower half of the GUI displays the results over time. Above the top-left corner of the history plot, you can see the current top category determined by the posterior post-processing (aka. the backend). This is the final output of the KWS detection system that also gets used for calling the `KeywordCallback()` function.

## FAQ
- **I am getting a `Wrong packet size` error**: If your packet is larger than expected, then you most likely did not correctly disable the serial debug prints. They are interfering with the binary debug data and are sending additional unwanted data. If your packet size is smaller than expected, then most likely, the serial-to-USB adapter or your computer driver is not able to keep up with the transmissions and dropping packets. Get in touch with us and we will try to figure out what's going on. This could also be caused by a mismatch in category numbers between target software and debugger. Make sure that you supply the exact amount of categories your target software is using to the Python debugger using the `--category-labels` flag.
- **Could not open port / No such file or directory**: If you get this error, then you are trying to connect to a USB/serial port that does not exist on your computer. Make sure that your ESP32-C3 is properly connected to your computer and verify that you selected the correct port (see the command line arguments above). If you are on Windows, you might also want to check your drivers. If this problem persists, get in touch with us.
