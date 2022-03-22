# Install and Setup

In this first step you will prepare and install TVM and its support tools.

## Prerequisites

As a prerequisite, make sure that the following dependencies are installed on your system:
- Python 3.8 with virtualenv support or make sure that you are inside an anaconda (miniconda) environment
- LLVM version 11 or higher (see https://apt.llvm.org/ for Debian/Ubuntu)
- Utility to decompress `.tar` archives

On Ubuntu the following command will be helpful:

```bash
sudo apt-get install python3 python3-dev python3-venv python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
```

## Create and enter virtual environment

For this lab we will be using Python virtual environments. They help to simplify Python versions and Python package management. The recommended way of creating a virtual environment is as follows:
```bash
virtualenv -p python3.8 /path/to/venv
```
where `/path/to/venv` is the path to files for the virtual environment. A good choice is simply `./venv`, but you can choose what you like.

Make sure to enter the virtual environment using `source venv/bin/activate` before continuing.

### Install Python dependencies

To install the dependencies inside the virtualenv, run
```bash
pip install -r requirements.txt
```

## Installing TVM

### Using pre-compiled python packages

This is the recommended approach for this lab as it ensures that the used version of TVM is compatible with this repository. See [`whl/README.md`](whl/README.md) for more details.

### Compile from source manually

If you want to do your own build of the TVM compiler framework, please follow the instructions on https://tvm.apache.org/docs/install/from_source.html. The step is completely optional and can be skipped if you have already completed the previous step.

Make sure to use a fixed commit (e.g. `68950873a30638dd1623e15f325770fe87178cce`) while checking out the repository to work on the lab exercises.

To be able to follow the tutorial, make sure that you enable `USE_MICRO`, `USE_MICRO_STANDALONE_RUNTIME` and `USE_LLVM` in your `build/config.cmake`. In addition, configure the `PYTHONPATH` environment variable properly, e.g. run `export PYTHONPATH=$(pwd)/python` inside the top level directory of the TVM repository. For custom builds (compiled prom source), you will need to run `python -m tvm.driver.tvmc` instead of `tvmc` in later steps.
