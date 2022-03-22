# Target Software

This directory contains the target software that runs on the ESP32-C3. The application listens to its surroundings with a microphone and indicates when it has detected a word.

## Deploy

The following instructions will help you build and deploy this target software to [ESP32-C3](https://www.espressif.com/en/products/socs/esp32-c3) devices using the [ESP-IDF](https://github.com/espressif/esp-idf).


### Install the ESP-IDF

**Note:** Since the ESP-IDF is already installed on the chair computers you can skip this part if you are working on one.

Follow the instructions of the [ESP-IDF get started guide](https://docs.espressif.com/projects/esp-idf/en/latest/get-started/index.html) to set up and install the ESP-IDF.

**Note:** Make sure you are cloning `release/v4.4` and installing the IDF for ESP32-C3 targets `./install.sh esp32c3` as explained below!

In later sections, it is assumed that the `IDF_DIR` environment variable points to the directory where `esp-idf` was cloned. On your local machine you will need to ensure that the `IDF_DIR` points to your ESP-IDF installation yourself. On chair computers this variable is set by the `setup.sh` script.

#### Ubuntu / Debian

If you are using Ubuntu or Debian the steps would be as follows:

1. Install the required dependencies
```
sudo apt-get install git wget flex bison gperf python3 python3-pip python3-setuptools cmake ninja-build ccache libffi-dev libssl-dev dfu-util libusb-1.0-0
```

2. Ensure you have at least Python 3.6, better Python 3.8
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
export IDF_DIR=$(pwd)
```
and
```
$IDF_DIR/install.sh esp32c3
```
This will install the IDF for ESP32-C3 targets.

5. You can now source the `export.sh` script. This will make the command-line tools available
```
source $IDF_DIR/export.sh
```
  *Note:* `. $IDF_DIR/export.sh` *does the exact same.*

6. You should now test your installation by running
```
idf.py --version
```
If you see something along the lines of `ESP-IDF v4.4.*` (note the `v4.4`) you have successfully installed the ESP-IDF.

**Note:** Whenever you open a new terminal you will need to repeat the first part of step 4., i.e. `export` the `IDF_DIR` variable to point to your `esp-idf` installation, as well as step 5., i.e. source the `export.sh` script!


### Build the Code

Don't forget to run `. $IDF_DIR/export.sh` once per terminal session before using the `idf.py` tool.

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

Don't forget to run `. $IDF_DIR/export.sh` once per terminal session before using the `idf.py` tool.

To flash/upload simply run
```
idf.py --port /dev/ttyUSB0 flash
```
**Note:** You might need to change the serial port of the device connected to your computer. You can list connected devices with `ls /dev/ttyUSB*` or `ls /dev/ttyACM*`. If you have more than one device, simply unplug the ESP32-C3 USB cable and check which device disappears. You might even be able to leave out the `--port` argument and only run `idf.py flash`. This way, the ESP-IDF will try to guess which device is the ESP32-C3. However, your mileage with this feature may vary...

As soon as the upload is finished, the ESP32-C3 will reset and the code will start running!

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

## Configuration of MicroKWS

The ESP-IDF uses so-called "KConfig" files to define options to easily change various features without modifying any code. It is documented here: https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/kconfig.html. The "GUI", which can be accessed via the `idf.py menuconfig` command, is text-based but should be straightforward given the shortcuts shown at the bottom of the window. Just make sure to properly save your changes to the `sdkconfig` file in the `target` directory (it will ask you whether you would like to save before you exit the GUI). After changing the project configuration, the program will be compiled from scratch.

We added a `MicroKWS Options` submenu to the `menuconfig` interface where you can change various things, such as:
- Modify model hyperparameters (These have to match the ones used during training)
- Configuration of the posterior handler
- Definition of used labels
- Debugging-related toggles, i.e. reduce or increase verbosity

## Debugging MicroKWS

As an alternative to `printf`-based debugging, which is often cumbersome for data-driven programs, we provide an interactive debugging tool to visualize the model's inputs and outputs. A detailed explanation can be found in the [`debug`](../debug/) directory at the top level of this repository.

To debug the static memory usage of a program, the ESP-IDF provides several commands that are explained in the official documentation (https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-guides/performance/size.html):
- `idf.py size`
- `idf.py size-components`
- `idf.py size-files`

## TVM specific details

The default generated artifacts (for `micro_kws_xs_yesno_quantized.tflite`) can be found in the `main/mlf` (or `main/mlf_tuned` for the autotuned version). For a detailed explanation of the contained files, please checkout [`../tvm/mlf_overview.md`](../tvm/mlf_overview.md) first. To use your newly generated MLF artifacts you can either replace the existing directories or use the `idf.py menuconfig`, as explained in the previous section, to change the used MLF path (i.e. to `../../tvm/gen/mlf_tuned`).
