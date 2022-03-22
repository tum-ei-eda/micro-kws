# Target Software

Contains the target software that runs on the ESP32-C3.

The application listens to its surroundings with a microphone and indicates when it has detected a word.


## Deploy

The following instructions will help you build and deploy this sample to the [ESP32-C3](https://www.espressif.com/en/products/socs/esp32-c3) devices using the [ESP IDF](https://github.com/espressif/esp-idf).


### Install the ESP-IDF

Follow the instructions of the [ESP-IDF get started guide](https://docs.espressif.com/projects/esp-idf/en/latest/get-started/index.html) to setup the toolchain and the ESP-IDF itself.

**Note:** Make sure you are cloning `release/v4.4` and installing the IDF for ESP32-C3 targets `./install.sh esp32c3` as explained below!

#### Ubuntu / Debian

If you are using Ubuntu or Debian the steps would be as follows:

1. Install the required dependencies
```
sudo apt-get install git wget flex bison gperf python3 python3-pip python3-setuptools cmake ninja-build ccache libffi-dev libssl-dev dfu-util libusb-1.0-0
```

2. Ensure you have at least Python 3.6
```
python3 --version
```

3. Clone the ESP-IDF GitHub repository
```
git clone --recursive --single-branch -b release/v4.4 https://github.com/espressif/esp-idf.git
```
  *Ensure you are cloning the v4.4 repository, as this is the version we are using! Different versions might lead to compilation errors!*

4. Install the ESP-IDF
```
cd esp-idf
```
and
```
./install.sh esp32c3
```
This will install the IDF for ESP32-C3 targets.

5. You can now source the `export.sh` script. This will make the command-line tools available
```
source ./export.sh
```
  *Note:* `. ./export.sh` *does the exact same.*

6. You should now test your installation by running
```
idf.py --version
```
If you see something along the lines of `ESP-IDF v4.4-367-gc29343eb9` (note the `v4.4`) you have successfully installed the ESP-IDF.

**Note:** Whenever you open a new terminal you will need to repeat step 5., i.e. source the `export.sh` script again!


### Build the Code

Change into the `target/` directory (where the `sdkconfig.defaults` is located).

Set the chip target to ESP32-C3

```
idf.py set-target esp32c3
```

Then build with `idf.py`
```
idf.py build
```

### Upload and Run the Code

To flash/upload simply run
```
idf.py --port /dev/ttyUSB0 flash
```
**Note:** You might need to change the serial port of the device connected to your computer. You can list connected devices with `ls /dev/ttyUSB*` or `ls /dev/ttyACM*`. If you have more than one device simply unplug the ESP32-C3 USB cable and check which device disappears. You might even be able to leave out the `--port` argument and only run `idf.py flash`. This way the ESP-IDF will try to guess which device is the ESP32-C3. However, your mileage with this feature may vary...

As soon as the upload is finish the ESP32-C3 will reset and the code will start running!

To monitor the serial output of the ESP32-C3 coming back via UART, run
```
idf.py --port /dev/ttyUSB0 monitor
```

Use `CTRL+]` to exit (this is not a `J`, but a "closing bracket" `]`).

The previous two commands can also be combined
```
idf.py --port /dev/ttyUSB0 flash monitor
```
**Note:** `flash` will also run `build` before flashing. So no need to explicitly `build` before `flash`ing.

## TVM specific

TODO

## Configuration of MicroKWS

TODO

## TODO
- Refactor comments in header files to represent actual implementation and not generic description.
- Remove unnecessary arguemnts from function calls.
- Go through flags in `tflite-lib/CMakeLists.txt` add check whether actually needed.
- Reintroduce license headers.
- Remove magic numbers and / or explain what they do. Move as many as reasonable into model_settings.
