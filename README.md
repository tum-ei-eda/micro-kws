# Micro KWS

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

Steps:

1. Open notebook on Google Colab via the linke above
2. Click the "Connect" button
3. Either excute all cells via the "Run all" button or run each cell individually.
4. Use the sidebar to navigate and explore the generated artifacts on the file system

### VSCode Devcontainer (GitHub hosted)

TODO

### VSCode Devcontainer (Local)

TODO

### Fully-local Development

TODO
