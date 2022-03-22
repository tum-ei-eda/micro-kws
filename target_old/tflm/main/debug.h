#ifndef DEBUG_H
#define DEBUG_H

#include <cstdint>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

TfLiteStatus DebugInit(tflite::ErrorReporter* error_reporter,
                       uint32_t baud_rate = 256000, uint32_t tx_pin = 7);

TfLiteStatus DebugRun(tflite::ErrorReporter* error_reporter,
                      int8_t* feature_data, int8_t* category_data);

#endif  // DEBUG_H
