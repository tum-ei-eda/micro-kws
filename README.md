Complete flow for keyword spotting on microcontrollers. From data collection to data preparation to training and deployment.

## Context
This project is used as part of the lab accompanying the lecture: Embedded System Design for Machine Learning offered by EDA@TUM.

Further, a number of out TInyML Workshops include a Hands-on base on the code you will find in this repository:
- Embedded World Conference 2026/2025/2024/2023:  Class 7.1 - Introduction to tinyML – Deploying Deep Learning Models Onto Low-power Micro-Controllers
- AI Factory Austria AI:AT 13.05.2026: tinyML: Deep Learning Models on Low-power Micro-Controllers (https://ai-at.eu/training/tinyml-deep-learning-models-on-low-power-micro-controllers/)

## Structure of this repository
The following directories can be found at the top level of this repository:
- `record/`: Provides utilities for recording and preprocessing new dataset samples (Optional)
- `train/`: Contains MicroKWS training flow and tutorial (Lab 1)
- `tvm/`: Contains a tutorial for generating MicroKWS kernels for a pre-trained model using the TVM Framework (Lab 2, Part 1)
- `target/`: Provides target software demo for deploying the MicroKWS application to a microcontroller (Lab 2, Part 2)
- `debug/`: Contains a python tool to debug the target application running on the device (Lab 2, optional)

## Usage

There are multiple approaches to try out the MicroKWS project, each has its own advantages and disadvantages.

### Google Colab

The fastest and easiest way to just run the Training and ML Compilation commands is via the provided Google Colaboratoy Notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tum-ei-eda/micro-kws/blob/workshop/MicroKWS.ipynb) (https://colab.research.google.com/github/tum-ei-eda/micro-kws/blob/workshop/MicroKWS.ipynb)

Most of the OS & Python dependencies are pre-installed and (depending on you tier) you will also get some GPU-resources for faster training.

The main limitation of the Colab-Notebook are:

- You can not forward your physical USB/Serial ports to the Colab instance, hence there is no way to "deploy" (aka. flash/upload) the compiled firmware to the microcontroller
- After some time of inactivity the instance will be killed. Make sure to not loose any data/modifications!
- There is a quota for GPU access for free users

<img width="1045" height="591" alt="grafik" src="https://github.com/user-attachments/assets/339ced7c-738a-4854-8d42-feb57e51aeeb" />

**Steps:**

1. Open notebook on Google Colab via the linke above
2. Click the "Connect" button
3. Either excute all cells via the "Run all" button or run each cell individually.
4. Use the sidebar to navigate and explore the generated artifacts on the file system
5. ...

### VSCode Devcontainer (GitHub hosted)

Another day to interfact with the MicroKWS project in the cloud is using the GitHub Codespaces Feature. It will basically open up a hosted instance running a VSCode server for development in the web browser or a locally installed VSCode app.

We provide a Docker image which inclused all dependencies (Tensorflow, TVM, ESP-IDF,...), hence you don't need to care about any prerequisites.

<img width="404" height="440" alt="grafik" src="https://github.com/user-attachments/assets/77ed8cda-2d25-49de-a7ed-f9f7e7b4c426" />

The main limitation of the Github-hosted Codespace (Devcontainer) are:

- The provided docker image is quite large (~25GB) and therefore need a fast internet connection and much disk space (on the provided GitHub machine) -> Expect some time to boot up the Image.
- Similary to Google Colab, you will also not be able to flash/upload any firmware to your devices
- The default instance is not very powerful (just 2 CPU cores). There is a way to upgrade to up to 32 cores, however the cost (per hour) wil also be 16x higher
- There is some free quota for CPU usage and disk storage depending on your GitHub subscription (Free, Pro,...). After the Quota is  exceeded there might be some fees. Hence, make sure to shutdown your instance if you don't need it
- After 30 minutes of inactivity the instance will be turned of. However you should still be able to reboot the insytance and access your data for a while.

*Warning*: Due to the size of the image, you will have to use a machine with at least 8 cores to have anough disk space available. We are trying to reduce the size of the image in the future.

**Steps:**

1. Open this Repository on GitHub
2. Click the "Clone" button, select the Codespaces tab and create an new codespace. You can also resume existing sessions here. (See screenshot)
3. ...

You can also use a local VSCode client instead of the web-version to connect to the instance:

<img width="411" height="601" alt="image" src="https://github.com/user-attachments/assets/2dbb324f-b4c4-4955-9cc4-9a14f27375e9" />


### VSCode Devcontainer (Local)

This approach uses the same Devcontainer mentioned above but utilizes the local machine (which needs a full VSCode and Docker installation) for executing the container. This make a lot of sense youi you have a powerful machine available but don't want to mess with installing all dependencies. Forwarding yout USB/Serial ports for flashing and monitory should be feasible (however we haven't tested it yet).

**Steps:**

1. Clone this repository and open it via VSCode
2. Make sure that the Dev Containers extension is installed
3. Press F1 and run the action "Dev conatiners: Open Folder in container"

<img width="741" height="103" alt="image" src="https://github.com/user-attachments/assets/9f14012f-c412-4ad9-9308-f3169172c781" />

### Fully-local Development

This way is the most flexible but needs some knownledge about installing Python and OS dependencies.

**Steps:**

1. Install VSCode
2. Clone and open the MicroKWS repository in VSCode
3. Install the ESP-IDF extensions and setup the Espressif toolchain by following the prompts
4. ...
