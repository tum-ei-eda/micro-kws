#include <cstdint>
#include <cstdio>

// clang-format off
#include "audio.h"
#include "debug.h"
#include "frontend.h"
#include "model_settings.h"
#include "model.h"
// clang-format on

#include "driver/i2s.h"
#include "driver/uart.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// TODO(fabianpedd): Does this really have to be global/static? Or can we also
// just make the task stack really large and make the tensor_arena local?
uint8_t tensor_arena[tensor_arena_size];

void micro_kws(void* params) {
  // Create an ErrorReporter object and get a reference.
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  // Parse the model data.
  const tflite::Model* model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(
        error_reporter,
        "ERROR: Schema version %d does not equal supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Could also use `tflite::AllOpsResolver resolver;` instead.
  tflite::MicroMutableOpResolver<4> micro_op_resolver(error_reporter);
  if (micro_op_resolver.AddDepthwiseConv2D() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "ERROR: In AddDepthwiseConv2D().");
    return;
  }
  if (micro_op_resolver.AddFullyConnected() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "ERROR: In AddFullyConnected().");
    return;
  }
  if (micro_op_resolver.AddSoftmax() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "ERROR: In AddSoftmax().");
    return;
  }
  if (micro_op_resolver.AddReshape() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "ERROR: In AddReshape().");
    return;
  }
  // if (micro_op_resolver.AddL2Pool2D() != kTfLiteOk) {
  //   TF_LITE_REPORT_ERROR(error_reporter, "ERROR: In AddL2Pool2D().");
  //   return;
  // }

  // Create interpreter and allocate tensors.
  tflite::MicroInterpreter interpreter(model, micro_op_resolver, tensor_arena,
                                       tensor_arena_size, error_reporter);

  if (interpreter.AllocateTensors() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "ERROR: In AllocateTensors().");
    return;
  }

  // TODO(fabianpedd): Do similar checks on output.
  if (interpreter.input(0)->dims->size != 2 ||
      interpreter.input(0)->dims->data[0] != 1 ||
      interpreter.input(0)->dims->data[1] != feature_element_count ||
      interpreter.input(0)->type != kTfLiteInt8) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "ERROR: Bad input tensor parameters in model.");
    return;
  }

  if (InitializeFrontend(error_reporter) != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "ERROR: In InitializeFrontend()");
  }

  if (InitializeAudio(error_reporter) != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "ERROR: In InitializeAudio()");
  }

  // This is only relevant when using the Python visualizer via the additional
  // UART interface
  DebugInit(error_reporter);

  // We are collecting 20ms of new audio data and are reusing 10ms of past data.
  // time    30ms = 20ms + 10ms
  // samples 480 = 320 new + 160 old
  // bytes   960 = 640 new + 320 old
  int8_t audio_buffer[960] = {0};

  // Contains our features. Interpreted as 40 by 49 byte 2d array.
  int8_t feature_buffer[feature_element_count];

  // Endless loop of main function.
  while (true) {
    // Get audio data from audio input and create slices until no more data is
    // available. But at most `feature_slize_count` times, which is equal to
    // 960ms of data. If we would
    for (uint32_t i = 0; i < feature_slize_count; i++) {
      // Get audio data via I2S from the audio driver.
      uint32_t actual_bytes_read = 0;
      int8_t i2s_read_buffer[640] = {0};
      if (GetAudioData(error_reporter, 640, &actual_bytes_read,
                       i2s_read_buffer) != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter, "ERROR: In GetAudioData().");
        return;
      }

      // If there is no more audio data available at the moment, exit the
      // loop and continue with inference.
      if (actual_bytes_read < 640) {
        break;
      }

      // If there is a full 20ms / 320 samples / 640 bytes available, move the
      // old data (10ms / 160 samples / 320 bytes) to the top of the buffer and
      // fill the rest with the new audio data from the i2s_read_buffer.
      memcpy(audio_buffer, audio_buffer + 640, 320);
      memcpy(audio_buffer, i2s_read_buffer, 640);

      // Generate new slice of features from audio samples using a 512 point
      // FFT and the Micro frontend.
      int8_t new_slice_buffer[feature_slice_size] = {0};
      if (GenerateFrontendData(error_reporter, (int16_t*)audio_buffer, 512,
                               new_slice_buffer) != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter,
                             "ERROR: In GenerateFrontendData().");
        return;
      }

      // Move other slices by one, i.e. make room to store new slice at the end.
      // TODO(fabianpedd): This is actually really inefficient. Using a
      // ringbuffer would be a lot more efficient but also more
      // complicated. The ringbuffer could be used in such a way as to
      // minimize the amount of copying required. It would only be
      // necessary to copy the data when a ringbuffer overflow occurs. The
      // ringbuffer, in turn, would have to be at least 2-3x times the size of
      // the data we are expecting in order for this method to bring any
      // improvement. So we are basically trading in storage (of which we should
      // have plenty) for computations. But then again, how expensive are a
      // couple of memmoves and memcpys in the grand scheme of things here?
      memmove(feature_buffer, feature_buffer + feature_slice_size,
              feature_element_count - feature_slice_size);
      memcpy(feature_buffer + feature_element_count - feature_slice_size,
             new_slice_buffer, feature_slice_size);
    }

    // // From file 060cd039_nohash_0.c
    // uint8_t data_buffi[] = {
    //     241, 226, 244, 224, 243, 223, 245, 225, 244, 227, 248, 227, 245, 227,
    //     246, 222, 240, 221, 236, 221, 230, 218, 235, 217, 231, 208, 224, 211,
    //     216, 210, 220, 207, 211, 195, 203, 199, 208, 214, 223, 202, 234, 211,
    //     229, 209, 232, 213, 206, 203, 233, 202, 219, 193, 238, 211, 229, 214,
    //     228, 205, 215, 201, 228, 209, 219, 196, 211, 202, 214, 192, 203, 184,
    //     207, 192, 195, 197, 200, 189, 203, 169, 195, 182, 231, 210, 227, 192,
    //     221, 198, 216, 185, 214, 200, 213, 203, 232, 197, 215, 195, 211, 175,
    //     217, 198, 223, 195, 217, 185, 205, 185, 203, 174, 205, 171, 194, 180,
    //     196, 193, 205, 182, 200, 178, 195, 175, 226, 201, 216, 137, 197, 183,
    //     198, 192, 225, 203, 220, 184, 218, 185, 207, 180, 193, 189, 214, 174,
    //     210, 185, 186, 170, 210, 195, 196, 167, 197, 177, 192, 170, 203, 171,
    //     192, 167, 186, 170, 193, 162, 209, 171, 207, 188, 212, 166, 195, 144,
    //     202, 182, 208, 177, 218, 178, 204, 157, 198, 178, 202, 165, 201, 170,
    //     204, 192, 204, 169, 198, 180, 190, 182, 193, 146, 189, 168, 197, 163,
    //     193, 160, 188, 156, 217, 181, 195, 161, 195, 182, 199, 116, 196, 179,
    //     188, 163, 202, 175, 197, 187, 203, 178, 204, 173, 194, 167, 184, 167,
    //     199, 177, 192, 174, 204, 165, 187, 162, 185, 159, 188, 175, 193, 156,
    //     182, 148, 196, 102, 166, 137, 191, 163, 195, 166, 191, 145, 213, 182,
    //     180, 160, 198, 149, 200, 168, 204, 179, 195, 158, 203, 137, 188, 164,
    //     193, 156, 190, 158, 189, 143, 186, 164, 184, 141, 177, 136, 176, 150,
    //     164, 0,   204, 177, 195, 178, 196, 102, 170, 144, 192, 122, 172, 161,
    //     193, 134, 192, 171, 206, 170, 188, 164, 190, 172, 186, 165, 192, 132,
    //     180, 136, 167, 119, 187, 128, 177, 154, 185, 134, 173, 136, 120, 144,
    //     206, 148, 173, 171, 206, 166, 182, 152, 159, 151, 198, 148, 192, 179,
    //     194, 141, 200, 148, 197, 157, 192, 142, 174, 152, 182, 138, 172, 95,
    //     179, 142, 175, 132, 180, 142, 185, 140, 174, 122, 190, 131, 150, 161,
    //     202, 117, 161, 99,  185, 152, 197, 78,  142, 0,   184, 170, 205, 112,
    //     200, 169, 184, 162, 201, 181, 199, 174, 207, 179, 202, 175, 199, 179,
    //     190, 123, 168, 157, 200, 179, 195, 160, 148, 133, 196, 160, 193, 174,
    //     194, 146, 137, 102, 174, 134, 172, 161, 183, 129, 176, 134, 172, 149,
    //     171, 95,  171, 131, 169, 123, 177, 120, 153, 85,  175, 126, 176, 119,
    //     173, 125, 159, 90,  166, 102, 180, 142, 185, 169, 145, 112, 177, 174,
    //     184, 151, 173, 138, 175, 122, 168, 122, 164, 135, 168, 156, 191, 51,
    //     180, 99,  168, 130, 175, 138, 191, 90,  158, 105, 175, 107, 176, 114,
    //     156, 102, 149, 90,  197, 130, 158, 85,  178, 170, 191, 117, 186, 127,
    //     177, 142, 169, 153, 187, 51,  173, 122, 188, 178, 177, 78,  171, 143,
    //     177, 105, 149, 68,  157, 95,  144, 126, 175, 129, 157, 90,  152, 78,
    //     149, 105, 168, 116, 187, 164, 167, 68,  177, 85,  119, 0,   159, 0,
    //     166, 85,  170, 129, 192, 162, 167, 149, 185, 125, 169, 150, 167, 85,
    //     154, 131, 170, 95,  167, 90,  170, 133, 153, 102, 154, 105, 156, 68,
    //     175, 123, 183, 159, 184, 0,   117, 151, 182, 85,  176, 126, 117, 123,
    //     191, 122, 154, 159, 175, 102, 139, 51,  172, 0,   170, 123, 161, 138,
    //     165, 139, 161, 51,  154, 117, 138, 140, 143, 51,  155, 119, 151, 105,
    //     200, 172, 198, 144, 116, 134, 152, 123, 139, 0,   0,   51,  166, 0,
    //     117, 116, 171, 102, 179, 123, 154, 85,  137, 68,  164, 107, 140, 90,
    //     168, 119, 133, 51,  139, 68,  160, 120, 177, 114, 184, 162, 173, 0,
    //     148, 0,   166, 153, 161, 85,  163, 167, 172, 0,   160, 136, 186, 161,
    //     180, 133, 182, 78,  150, 85,  135, 78,  147, 152, 159, 51,  167, 78,
    //     142, 99,  152, 123, 162, 139, 177, 120, 125, 131, 167, 0,   161, 0,
    //     139, 134, 166, 68,  164, 51,  136, 51,  175, 145, 184, 152, 187, 150,
    //     157, 120, 149, 90,  162, 78,  163, 68,  163, 0,   139, 102, 129, 78,
    //     144, 0,   137, 68,  141, 51,  105, 0,   0,   0,   174, 95,  174, 0,
    //     147, 130, 167, 78,  169, 117, 164, 0,   157, 51,  157, 78,  168, 0,
    //     170, 131, 166, 105, 159, 0,   140, 51,  134, 95,  154, 68,  160, 126,
    //     152, 78,  168, 151, 149, 0,   0,   117, 171, 85,  139, 90,  155, 0,
    //     51,  0,   110, 78,  119, 0,   164, 122, 166, 136, 167, 0,   117, 0,
    //     143, 68,  170, 90,  133, 95,  141, 122, 179, 120, 163, 78,  139, 0,
    //     151, 112, 99,  51,  149, 102, 174, 51,  143, 0,   161, 0,   112, 90,
    //     148, 68,  162, 157, 166, 68,  174, 102, 161, 105, 161, 140, 169, 85,
    //     144, 51,  180, 131, 175, 163, 195, 168, 185, 147, 162, 99,  171, 133,
    //     141, 0,   51,  0,   167, 68,  157, 0,   180, 0,   105, 0,   0,   68,
    //     172, 114, 146, 107, 178, 102, 140, 95,  149, 117, 166, 90,  145, 112,
    //     178, 120, 172, 197, 206, 184, 197, 143, 156, 85,  159, 107, 51,  0,
    //     143, 51,  160, 0,   139, 119, 156, 0,   139, 0,   0,   0,   133, 0,
    //     126, 78,  144, 110, 119, 51,  158, 102, 125, 107, 167, 126, 159, 114,
    //     178, 179, 205, 184, 191, 125, 150, 110, 138, 0,   217, 202, 211, 188,
    //     216, 201, 202, 90,  116, 68,  128, 0,   90,  0,   149, 102, 152, 51,
    //     141, 99,  176, 168, 201, 195, 218, 196, 216, 200, 207, 194, 227, 209,
    //     228, 205, 215, 170, 188, 170, 209, 198, 217, 177, 192, 220, 238, 189,
    //     224, 198, 152, 102, 105, 0,   125, 105, 171, 90,  127, 116, 151, 68,
    //     126, 183, 221, 213, 229, 196, 227, 204, 220, 203, 233, 213, 237, 208,
    //     216, 174, 206, 193, 212, 191, 206, 158, 184, 198, 216, 172, 233, 210,
    //     193, 142, 159, 0,   99,  68,  168, 0,   125, 0,   180, 152, 185, 197,
    //     222, 190, 216, 197, 217, 179, 200, 171, 212, 183, 206, 179, 194, 0,
    //     95,  0,   51,  51,  182, 0,   182, 179, 194, 183, 236, 212, 209, 184,
    //     90,  0,   68,  0,   149, 0,   185, 145, 164, 151, 211, 194, 213, 192,
    //     216, 179, 190, 128, 193, 164, 194, 157, 189, 152, 185, 90,  105, 0,
    //     107, 0,   169, 0,   184, 168, 183, 190, 224, 195, 228, 197, 179, 141,
    //     171, 0,   68,  90,  180, 144, 179, 181, 221, 199, 210, 184, 213, 175,
    //     180, 119, 181, 160, 174, 130, 182, 119, 175, 123, 99,  0,   0,   0,
    //     186, 68,  195, 159, 162, 171, 195, 136, 229, 196, 177, 105, 102, 0,
    //     128, 95,  166, 149, 204, 185, 210, 182, 204, 186, 211, 170, 177, 122,
    //     175, 161, 174, 95,  177, 112, 182, 116, 78,  110, 174, 105, 197, 117,
    //     197, 164, 161, 146, 172, 139, 218, 165, 198, 162, 187, 0,   78,  0,
    //     180, 161, 200, 180, 183, 129, 173, 148, 177, 107, 112, 0,   144, 116,
    //     143, 0,   138, 0,   182, 147, 90,  0,   0,   0,   188, 0,   191, 152,
    //     158, 138, 129, 0,   187, 0,   114, 131, 183, 85,  159, 90,  183, 126,
    //     198, 182, 155, 0,   105, 0,   143, 0,   0,   0,   68,  0,   78,  0,
    //     0,   0,   160, 140, 148, 143, 188, 140, 187, 68,  157, 0,   0,   0,
    //     0,   0,   0,   0,   148, 78,  158, 51,  112, 0,   160, 0,   0,   0,
    //     51,  0,   0,   0,   51,  0,   105, 0,   102, 0,   0,   0,   0,   0,
    //     177, 123, 127, 90,  123, 0,   156, 0,   68,  0,   0,   0,   0,   0,
    //     0,   0,   133, 0,   68,  0,   123, 68,  51,  0,   126, 0,   0,   0,
    //     0,   0,   0,   0,   68,  0,   0,   0,   0,   0,   0,   0,   136, 51,
    //     110, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    //     78,  0,   0,   0,   146, 0,   0,   0,   107, 0,   0,   0,   0,   0,
    //     0,   51,  165, 0,   123, 68,  185, 180, 192, 185, 224, 207, 222, 200,
    //     211, 189, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   120, 0,
    //     0,   51,  105, 85,  150, 0,   0,   0,   0,   0,   0,   0,   0,   0,
    //     147, 0,   125, 99,  174, 170, 203, 177, 214, 203, 229, 215, 237, 206,
    //     0,   0,   107, 0,   0,   0,   0,   0,   0,   0,   0,   0,   95,  68,
    //     130, 0,   0,   90,  102, 0,   0,   0,   0,   0,   0,   0,   105, 0,
    //     90,  51,  180, 174, 194, 170, 204, 197, 236, 217, 228, 198, 0,   0,
    //     0,   0,   0,   0,   0,   0,   0,   0,   51,  51,  170, 90,  159, 0,
    //     102, 51,  112, 0,   0,   0,   0,   0,   0,   0,   0,   0,   146, 78,
    //     175, 180, 196, 166, 195, 192, 216, 204, 230, 209, 0,   0,   0,   0,
    //     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   102, 102, 159, 0,
    //     0,   0,   0,   0,   0,   0,   0,   0,   116, 0,   99,  102, 183, 152,
    //     178, 164, 205, 192, 222, 187, 208, 195, 0,   0,   68,  0,   0,   0,
    //     0,   0,   0,   0,   68,  0,   141, 120, 155, 105, 146, 78,  51,  0,
    //     0,   0,   0,   0,   0,   0,   0,   0,   0,   68,  176, 107, 147, 153,
    //     190, 183, 225, 205, 218, 177, 0,   0,   0,   0,   0,   0,   0,   0,
    //     0,   0,   0,   0,   157, 0,   110, 0,   0,   0,   0,   0,   51,  0,
    //     0,   0,   0,   0,   130, 0,   95,  0,   143, 51,  138, 119, 199, 197,
    //     219, 182, 215, 167, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    //     163, 0,   51,  127, 156, 90,  0,   0,   0,   0,   78,  0,   0,   0,
    //     0,   0,   125, 0,   117, 0,   138, 68,  125, 85,  160, 166, 204, 134,
    //     186, 167, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    //     144, 120, 156, 0,   78,  78,  0,   0,   0,   0,   90,  0,   0,   0,
    //     107, 0,   51,  78,  162, 90,  144, 85,  197, 165, 165, 0,   152, 125,
    //     99,  0,   68,  0,   0,   0,   0,   0,   0,   0,   51,  0,   0,   0,
    //     85,  90,  134, 105, 0,   0,   90,  0,   0,   0,   0,   0,   157, 0,
    //     155, 175, 190, 0,   107, 0,   128, 0,   51,  0,   85,  0,   0,   0,
    //     51,  0,   0,   0,   0,   0,   0,   0,   117, 0,   0,   0,   116, 0,
    //     0,   0,   0,   0,   0,   0,   0,   0,   0,   133, 169, 90,  116, 159,
    //     173, 0,   126, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    //     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    //     107, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    //     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    //     0,   0,   0,   0,   0,   0,   99,  0,   95,  114, 0,   0,   0,   0,
    //     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    //     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    //     0,   0,   114, 127, 157, 107, 151, 0,   0,   0,   0,   0,   0,   0,
    //     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    //     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    //     0,   138, 154, 119, 68,  0,   51,  0,   0,   0,   0,   0,   0,   0,
    //     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    //     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    //     107, 68,  107, 85,  51,  0,   0,   0,   0,   0,   0,   0,   0,   0,
    //     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    // };

    // vTaskDelay(pdMS_TO_TICKS(200));
    // for (int i = 0; i < 1960; i++) {
    //   feature_buffer[i] = (int8_t)((int32_t)data_buffi[i] - 128);
    // }

    // Copy the feature buffer into the model input buffer and run the
    // inference.
    memcpy(interpreter.input(0)->data.int8, feature_buffer,
           feature_element_count);
    if (interpreter.Invoke() != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(error_reporter, "ERROR: Invoke failed.");
      return;
    }

    // Send the feature buffer and inferences results to the computer for
    // analysis and debugging.
    DebugRun(error_reporter, feature_buffer, interpreter.output(0)->data.int8);
  }
}

extern "C" void app_main(void) {
  xTaskCreate(&micro_kws,   // Function that implements the task.
              "micro_kws",  // Text name for the task.
              32 * 1024,    // Stack size in bytes, so 32KB.
              NULL,         // Parameter passed into the task.
              8,            // Priority of task. Higher number, higher prio.
              NULL          // Task handle.
  );
}
