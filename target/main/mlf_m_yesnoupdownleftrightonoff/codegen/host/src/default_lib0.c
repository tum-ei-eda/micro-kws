#include "tvm/runtime/c_runtime_api.h"
#ifdef __cplusplus
extern "C" {
#endif
__attribute__((section(".bss.noinit.tvm"), aligned(16)))
static uint8_t global_workspace[46560];
#include <tvmgen_default.h>
TVM_DLL int32_t tvmgen_default___tvm_main__(void* serving_default_input_0,void* output0,uint8_t* global_workspace_0_var);
int32_t tvmgen_default_run(struct tvmgen_default_inputs* inputs,struct tvmgen_default_outputs* outputs) {return tvmgen_default___tvm_main__(inputs->serving_default_input_0,outputs->StatefulPartitionedCall_0,&global_workspace);
}
#ifdef __cplusplus
}
#endif
;