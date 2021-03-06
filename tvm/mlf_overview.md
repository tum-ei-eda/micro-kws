# TVM's Model Library Format (MLF) in detail

This document will introduce you to the MLF artifact generated by the TVM compiler when deploying neural network kernels to embedded devices.

## Directory structure / File hierarchy

In the following, a truncated version of the contents of the `mlf.tar` archive generated by the `tvmc compile` command is given.

```
.
├── codegen
│   └── host
│       ├── include
│       │   └── tvmgen_default.h
│       └── src
│           ├── default_lib0.c
│           └── default_lib1.c
├── metadata.json
├── parameters
│   └── default.params
├── runtime
│   └── ...
└── src
    └── relay.txt
```

The files can be extracted using `tar xvf mlf.tar -C mlf/`.

- `codegen/host/` Contains the main artifacts of the TVM compilation flow:
  - The generated kernels e.g. `src/default_lib1.c`
  - Interfacing code for calling the generated kernels, as well as for passing model inputs and outputs, can be found in `include/tvmgen_default.h` and `src/default_lib0.c`
  - *Optional:* If parts of the model are processed by i.e. a hardware accelerator, there may also exist `src/default_lib2.c` etc.
- `metadata.json` Contains some information about the model and used workspace buffers (only used by some executors).
- `parameters/default.params` Binary file containing the data of all constant tensors used in the model. (Ignored by AoT executor, which automatically includes them into the kernel sources.)
- `runtime/` Contains the additional sources required to make use of the generated kernels using the chosen executor/runtime. (here: TVM minimal crt and third party libraries)
- `src/relay.txt` The intermediate representation of the model graph before lowering (might be useful for debugging).

## TVM kernels

The `default_lib.1` can be split up into 3 sections:

1. Definition of parameters/constants used by the kernels
2. Actual kernel implementations
3. Model entry point `tvmgen_default___tvm_main__` for the inference.

### 1. Parameter definitions

The parameters of a model are declared as `const` as they are to be stored in ROM (e.g. `.rodata` segment of the ELF file).

**Example:**

```c
static const int64_t __attribute__((section(".rodata.tvm"), aligned(16))) constant_3[8] = {
    +0x0000008000000000LL, +0x0000008000000000LL, +0x0000008000000000LL, +0x0000008000000000LL, +0x0000008000000000LL, +0x0000008000000000LL, +0x0000008000000000LL, +0x0000008000000000LL
};
```

### 2. Kernel implementation

Due to fusing of different operators, which is enabled by default in TVM (See `--pass-config relay.FuseOps.max_depth=X`), the generated kernels can become very messy, especially if naive kernels schedules are replaced with complex low-level optimizations such as tiling etc.

**Example:**

```c
TVM_DLL int32_t tvmgen_default_fused_reshape_cast_subtract(int8_t* placeholder, int16_t* T_subtract, uint8_t* global_workspace_1_var) {
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 49; ++ax0_ax1_fused) {
    for (int32_t ax2 = 0; ax2 < 40; ++ax2) {
      int32_t cse_var_1 = ((ax0_ax1_fused * 40) + ax2);
      T_subtract[cse_var_1] = (((int16_t)placeholder[cse_var_1]) - (int16_t)-128);
    }
  }
  return 0;
}
```

### 3. Main entry point

The `tvmgen_default___tvm_main__` function has 3 main tasks:
- Manage global workspace buffers used by the model (intermediate tensors and temporary scratch buffers).
- Calling the kernels according to the computational graph of the model (generated automatically using the Ahead-of-Time (AoT) compiler to get rid of any runtime overheads).
- Error handling

**Example:**

```c
TVM_DLL int32_t tvmgen_default___tvm_main__(int8_t* input_buffer_var, int8_t* output_buffer_var, uint8_t* global_workspace_0_var) {
  void* sid_2_let = (&(global_workspace_0_var[0]));
  void* sid_3_let = (&(global_workspace_0_var[0]));
  void* sid_4_let = (&(global_workspace_0_var[12144]));
  void* sid_5_let = (&(global_workspace_0_var[12144]));
  void* sid_6_let = (&(global_workspace_0_var[0]));
  void* sid_7_let = (&(global_workspace_0_var[0]));
  if (tvmgen_default_fused_reshape_cast_subtract(input_buffer_var, sid_7_let, global_workspace_0_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip(sid_7_let, sid_6_let, global_workspace_0_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_max_pool2d_cast_subtract(sid_6_let, sid_5_let, global_workspace_0_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip_1(sid_5_let, sid_4_let, global_workspace_0_var) != 0 ) return -1;
  if (tvmgen_default_fused_reshape_cast_subtract_1(sid_4_let, sid_3_let, global_workspace_0_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_contrib_dense_pack_add_fixed_point_multiply_add_clip_cast_cast_subtract_cb9548905a4a8167_(sid_3_let, sid_2_let, global_workspace_0_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_softmax_divide_add_clip_round_cast(sid_2_let, output_buffer_var, global_workspace_0_var) != 0 ) return -1;
  return 0;
}
```

## Model Metadata

The model metadata (not to be confused with the `mod.json` graph JSON file) contains some information about the kernels used by the model and their workspaces.

**Example:**

```json
{
  "executors": [
    "aot"
  ],
  "export_datetime": "2022-04-12 13:36:51Z",
  "external_dependencies": [
    {
      "short_name": "tvm_standalone_crt",
      "url": "./runtime",
      "url_type": "mlf_path",
      "version_spec": "0.9.dev0"
    }
  ],
  "memory": {
    "functions": {
      "main": [
        {
          "constants_size_bytes": 0,
          "device": 1,
          "io_size_bytes": 1964,
          "workspace_size_bytes": 23664
        }
      ],
      "operator_functions": [
        {
          "function_name": "tvmgen_default_fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip_1",
          "workspace": [
            {
              "device": 1,
              "workspace_size_bytes": 0
            }
          ]
        },
        ...
      ]
    }
  },
  "model_name": "default",
  "style": "full-model",
  "target": {
    "1": "c -keys=cpu -link-params=0"
  },
  "version": 5
}
```

**Hint:** When enabling USMP (Unified Static Memory Planner) in TVM, all temporary buffer allocations are replaced by a global workspace buffer (e.g. `global_workspace_0_var`), thus `workspace_size_bytes` might be zero for every kernel function.

## Relay IR

"Relay" is one of the intermediate representations used in the TVM compilation flow. You will find more information on its syntax [here](https://tvm.apache.org/docs/arch/relay_intro.html).

**Example:**

```
def @main(%input: Tensor[(1, 1960), int8], %v_param_1: Tensor[(20, 8, 1, 8), int8], %v_param_2: Tensor[(8), int32], %v_param_3: Tensor[(10, 4, 8, 8), int8], %v_param_4: Tensor[(8), int32], %v_param_5: Tensor[(4, 3840), int8], %v_param_6: Tensor[(4), int32], output_tensor_names=["Identity"]) {
  %0 = reshape(%input, newshape=[-1, 49, 40, 1]);
  %1 = qnn.conv2d(%0, %v_param_1, -128, 0, 0.101562f, meta[relay.Constant][0], padding=[9, 3, 10, 4], channels=8, kernel_size=[20, 8], data_layout="NHWC", kernel_layout="HWIO", out_dtype="int32");
  %2 = nn.bias_add(%1, %v_param_2, axis=3);
  %3 = qnn.requantize(%2, meta[relay.Constant][1], 0, 0.141817f, -128, axis=3, rounding="UPWARD", compute_dtype="int64", out_dtype="int8");
  %4 = clip(%3, a_min=-128f, a_max=127f);
  %5 = nn.max_pool2d(%4, pool_size=[2, 2], strides=[2, 2], padding=[0, 0, 0, 0], layout="NHWC");
  %6 = qnn.conv2d(%5, %v_param_3, -128, 0, 0.141817f, meta[relay.Constant][2], padding=[4, 1, 5, 2], channels=8, kernel_size=[10, 4], data_layout="NHWC", kernel_layout="HWIO", out_dtype="int32");
  %7 = nn.bias_add(%6, %v_param_4, axis=3);
  %8 = qnn.requantize(%7, meta[relay.Constant][3], 0, 0.0582336f, -128, axis=3, rounding="UPWARD", compute_dtype="int64", out_dtype="int8");
  %9 = clip(%8, a_min=-128f, a_max=127f);
  %10 = reshape(%9, newshape=[-1, 3840]);
  %11 = reshape(%10, newshape=[-1, 3840]);
  %12 = qnn.dense(%11, %v_param_5, -128, 0, 0.0582336f, 0.00373117f, units=4, out_dtype="int32");
  %13 = nn.bias_add(%12, %v_param_6);
  %14 = qnn.requantize(%13, 0.000217279f, 0, 0.264173f, 27, rounding="UPWARD", compute_dtype="int64", out_dtype="int8");
  %15 = qnn.dequantize(%14, 0.264173f, 27);
  %16 = nn.softmax(%15);
  qnn.quantize(%16, 0.00390625f, -128, out_dtype="int8")
}
```
