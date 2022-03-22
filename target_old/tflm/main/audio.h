#ifndef AUDIO_H
#define AUDIO_H

#include <cstdint>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

TfLiteStatus InitializeAudio(tflite::ErrorReporter* error_reporter);

TfLiteStatus GetAudioData(tflite::ErrorReporter* error_reporter,
                          uint32_t requested_size, uint32_t* actual_size,
                          int8_t* data);

#endif  // AUDIO_H
