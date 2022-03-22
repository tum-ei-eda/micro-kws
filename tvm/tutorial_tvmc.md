# MircoKWS deployment using TVMC command-line interface

This document will explain the required steps to process a model using the TVM machine learning compiler framework in detail. To generate the inference code for the application example of real-time keyword-spotting, the following steps have to be performed.

*While the following steps should work on modern versions of Windows, MacOS and Ubuntu, the complete flow was only tested on Ubuntu.*

**Before continuing: Make sure that your virtual environment from the TVM installation step is active (sourced).**

### Obtaining a model

The examples are intended to be used with an eight keyword model `data/micro_kws_m_yesnoupdownleftrightonoff_quantized.tflite` from lab 1, which you find in the `tvm/data/` directory. However, most steps should also be applicable to any other model. A very small example model using only two keywords is also provided: `data/micro_kws_xs_yesno_quantized.tflite`

Various "Model-Zoos" are available on the internet if you want to use existing pre-trained models for a given dataset/application. We also provide our chair's set of TinyML benchmarking models in a GitHub repository: https://github.com/tum-ei-eda/mlonmcu-models.

Beside of `.tflite` files, TVM also supports various other model formats such as ONNX. However, only quantized TFLite will be considered in this tutorial.

### Compiling the model

The term `compile` in the context of TVMC describes the complete compilation pipeline internally used by TVM (e.g. Relay frontend, Partitioning, Lowering, Code Generation etc.).

In this section, an example model is processed via the TVMC command line interface for two typical application scenarios.

#### For execution on host

First, we compile our model using the `--target llvm` option and test run the MicroKWS for a functionality check on the host machine.

```bash
tvmc compile data/micro_kws_m_yesnoupdownleftrightonoff_quantized.tflite --output gen/module.tar --target llvm
```

You should find a `module.tar` file in the `gen/` directory.

The used command line options can be explained as follows:
- `data/micro_kws_m_yesnoupdownleftrightonoff_quantized.tflite`: The (quantized) TFLite model to process
- `--output gen/module.tar`: The destination file
- `--target llvm`: Tell TVM that we want to generate an LLVM runtime library for the module

#### For execution on embedded device

The following (quite complex) command should be used to generate the TVM kernel implementations used in a later step:

```bash
tvmc compile data/micro_kws_m_yesnoupdownleftrightonoff_quantized.tflite --output gen/mlf.tar \
    --target c --runtime crt --executor-aot-interface-api c \
    --executor=aot --executor-aot-unpacked-api=1 --desired-layout NCHW \
    --output-format mlf --pass-config tir.disable_vectorize=1 \
    --pass-config tir.usmp.enable=1 --pass-config tir.usmp.algorithm=hill_climb
```

The used command line options can be explained as follows:
- `data/micro_kws_s.tflite`: The (quantized) TFLite model to process
- `--output gen/mlf.tar`: The destination file
- `--target c`: Tell TVM that we want to generate C kernels and not LLVM
- `--runtime=crt`: Use the standalone CRT as we want to use a minimal runtime environment (e.g. baremetal code)
- `--executor-aot-interface-api c`: Generate a straightforward interface to define input and output tensors.
- `--executor=aot`: Generate top-level model code using the Ahead-of-Time compiler to get rid of any runtime/interpreter related-overheads (Alternative: `graph` runtime using `graph.json` and `params.bin`)
- `--executor-aot-unpacked-api=1`: Use the "unpacked" calling convention for more compact code and less stack usage compared to TVM's default approach.
- `--desired-layout {NCHW,NHWC}`: Set the preferred layout of weights/kernels in the model (optional)
- `--output-format mlf`: Return MLF archive with codegen results. (Explained later)
- `--pass-config tir.disable_vectorize=1`: Disable optimizations which are not available on embedded platforms
- `--pass-config tir.usmp.enable=1`: Use the USMP (Unified Static Memory Planner) to minimize memory usage using a global tensor arena.
- `--pass-config tir.usmp.algorithm=hill_climb` Select the algorithm used by the USMP (Alternatives: `greedy_by_size`, `greedy_by_conflicts`)

Further information on the available options can be found using the `--help` flag.

#### Bonus: Using autotuned operators to build more efficient kernels

A major advantage of TVM's code-generation approach, besides the possibility to apply complex optimization at various abstraction layers, is the degree of freedom in the choice of the `compute` and `schedule` used to represent a given operator. While hand-crafted kernels (see TFLite for Microcontrollers) have to be as generic as possible to support a wide variety of different datatypes, shapes etc. TVM can choose from a number of possible parameterizable implementations for a given operator.

The challenge is to find the best implementation alongside a combination of parameters which has the "best" performance on a specific target device. An AutoTuner is provided by TVM to automate this process by exploring the search space using a number of exploration and optimization algorithms. As the autotuning procedure is quite time-intensive and requires a complex hardware/software setup, we will not invoke the AutoTuner here. Instead, we have done the autotuing for you. We provide the tuning records (`micro_kws_m_tuning_log_nchw_best.txt`) for the `data/micro_kws_s.tflite` model (see `tvm/data/`). Please use them for the following steps.

Add `--tuning-records data/micro_kws_m_tuning_log_nchw_best.txt` to the `tvmc compile` definition to use the tuning logs when compiling the model:

```bash
tvmc compile data/micro_kws_m_yesnoupdownleftrightonoff_quantized.tflite --output gen/mlf_tuned.tar \
    --target c --runtime crt --executor-aot-interface-api c \
    --executor=aot --executor-aot-unpacked-api=1 --desired-layout NCHW \
    --output-format mlf --pass-config tir.disable_vectorize=1 \
    --pass-config tir.usmp.enable=1 --pass-config tir.usmp.algorithm=hill_climb \
    --tuning-records data/micro_kws_m_tuning_log_nchw_best.txt
```

In a later experiment, we will see the impact of autotuning on the inference speed (performance).

### Running the model

Depending on the used options in the previous step, you will have one of the following artifacts inside `gen/module.tar` or `gen/mlf.tar`:

1. A compiled library (shared object) intended to be loaded by a CPUs LLVM runtime (contains `mod.so` (kernels), `mod.json` (graph) and `mod.params`)
2. A model library interface (MLF) archive containing the generated kernel in C and runtime required to invoke the model alongside with some additional metadata. For a more detailed explanation of the archive contents/directory structure, see [`mlf_overview.md`](mlf_overview.md)

If you want to manually inspect your `.tar` artifact, you can extract it:

```
mkdir -p gen/mlf
tar xvf gen/mlf.tar -C gen/mlf/

mkdir -p gen/mlf_tuned
tar xvf gen/mlf_tuned.tar -C gen/mlf_tuned/
```

### Testing on host

The `tvmc run` subcommand provides an interface to invoke the model on a certain set of targets e.g. CPU (the default option). You can also provide input data to validate if the model outputs match the expectations.

Execute the following:

```bash
tvmc run gen/module.tar --fill-mode random --print-time --print-top 10
```

The `--print-time` flag is just a benchmark option and can be omitted.

Instead of generating random input values, it is possible to provide actual features from the dataset using the `--inputs` option. You can find a script to generate a `.npz` file for TVM in `train/`, which can be used as follows:

```bash
python wav2features.py /path/to/speech_dataset/no/0137b3f4_nohash_1.wav gen/no.npz --output-format npz
tvmc run gen/module.tar --inputs gen/no.npz --print-top 4
```

The output should be similar to this:

```
[[   3    1    9    8    7    6    5    4    2    0]
 [ 125 -125 -128 -128 -128 -128 -128 -128 -128 -128]]
```

Pay attention to the changing output indices in the first row!

### Flashing to micro-controller

While functional verification using actual samples from the dataset is a useful first step, the main goal is to use the generated kernels on a real embedded device to run real-time inference using a microphone.

#### MicroTVM

The standard flow supported by TVM is known as MicroTVM and can be invoked via `tvmc micro [create-project|build|flash]`. However, as our ESP32-C3 target is not yet supported by MicroTVM, we will follow a different approach explained in the following section.

#### Deploying MicroKWS model to ESP32-C3 dev board using ESP-IDF

The target software used for the second lab assignment can be found in the [`target`](../target/) directory. Please continue there for part two of the lab.
