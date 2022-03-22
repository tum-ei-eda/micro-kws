/*
 * Copyright (c) 2022 TUM Department of Electrical and Computer Engineering.
 *
 * This file is part of the MicroKWS project.
 * See https://gitlab.lrz.de/de-tum-ei-eda-esl/ESD4ML/micro-kws for further
 * info.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tvm_wrapper.h"

#include <cstdarg>
#include <cstdlib>

#include "esp_log.h"
#include "sdkconfig.h"
#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/crt/error_codes.h"
#include "tvmgen_default.h"

#ifdef _DEBUG
#define DBGPRINTF(format, ...) ESP_LOGI(__FILE__, format, ...)
#else
#define DBGPRINTF(format, ...)
#endif

// Define data for input and output tensors
char input0_data[1960];
void* inputs[] = {input0_data};
struct tvmgen_default_inputs tvmgen_default_inputs = {input0_data};
char output0_data[CONFIG_MICRO_KWS_NUM_CLASSES];
void* outputs[] = {output0_data};
struct tvmgen_default_outputs tvmgen_default_outputs = {output0_data};

void TVMLogf(const char* msg, ...) {
  va_list args;
  va_start(args, msg);
  DBGPRINTF(args, msg)
  va_end(args);
}

tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev,
                                          void** out_ptr) {
  return kTvmErrorFunctionCallNotImplemented;
}

tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
  return kTvmErrorFunctionCallNotImplemented;
}

void __attribute__((noreturn)) TVMPlatformAbort(tvm_crt_error_t code) {
  exit(1);
}

void* model_input_ptr(size_t index) { return inputs[index]; }

void* model_output_ptr(size_t index) { return outputs[index]; }

esp_err_t model_invoke() {
  if (tvmgen_default_run(&tvmgen_default_inputs, &tvmgen_default_outputs)) {
    TVMPlatformAbort(kTvmErrorPlatformCheckFailure);
    return ESP_FAIL;
  }
  return ESP_OK;
}
