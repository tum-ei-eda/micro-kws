# Micro KWS

Complete flow for keyword spotting on microcontrollers. From data collection to data preparation to training and deployment.

## Context
This project is used as part of the lab accompanying the lecture: Embedded System Design for Machine Learning offered by EDA@TUM.

## Structure of this repository
The following directories can be found at the top level of this repository:
- `record/`: Provides utilities for recording and preprocessing new dataset samples (Optional)
- `train/`: Contains MicroKWS training flow and tutorial (Lab 1)
- `tvm/`: Contains a tutorial for generating MicroKWS kernels for a pre-trained model using the TVM Framework (Lab 2, Part 1)
- `target/`: Provides target software demo for deploying the MicroKWS application to a microcontroller (Lab 2, Part 2)
- `debug/`: Contains a python tool to debug the target application running on the device (Lab 2, optional)
