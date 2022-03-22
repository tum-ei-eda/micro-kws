#ifndef FRONTEND_H
#define FRONTEND_H

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

// Sets up any resources needed for the feature generation pipeline.
TfLiteStatus InitializeFrontend(tflite::ErrorReporter* error_reporter);

// Converts audio sample data into a more compact form that's appropriate for
// feeding into a neural network.
TfLiteStatus GenerateFrontendData(tflite::ErrorReporter* error_reporter,
                                  const int16_t* input, size_t input_size,
                                  int8_t* output);

#endif  // FRONTEND_H
