#include "audio.h"

#include <cstring>

#include "driver/i2s.h"
#include "esp_spi_flash.h"
#include "freertos/ringbuf.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

static RingbufHandle_t buf_handle = NULL;

static void CaptureAudioSamples(void* arg) {
  tflite::ErrorReporter* error_reporter = (tflite::ErrorReporter*)arg;

  constexpr uint32_t bytes_to_read = 512;
  uint32_t bytes_read = 0;
  int8_t data_buf[bytes_to_read] = {0};

  while (1) {
    i2s_read(I2S_NUM_0, (void*)data_buf, (size_t)bytes_to_read,
             (size_t*)&bytes_read, pdMS_TO_TICKS(100));

    if (bytes_read < bytes_to_read) {
      TF_LITE_REPORT_ERROR(
          error_reporter,
          "ERROR: In i2s_read(). Could ony read %d of %d bytes.", bytes_read,
          bytes_to_read);
      return;
    }

    if (xRingbufferSend(buf_handle, data_buf, bytes_read, pdMS_TO_TICKS(100)) !=
        pdTRUE) {
      TF_LITE_REPORT_ERROR(
          error_reporter,
          "ERROR: In xRingbufferSend(). Could not send %d bytes.", bytes_read);
      return;
    }
  }
}

TfLiteStatus InitializeAudio(tflite::ErrorReporter* error_reporter) {
  i2s_config_t i2s_config = {
      /* Master should supply clock and we are only receiving data. */
      .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
      /* Sample rate of 16KHz */
      .sample_rate = 16000,
      /* 16bit per sample, i.e. two bytes. */
      .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
      /* We only have a mono microphone which is outputting its audio data into
         the left channel of the I2S interface (L/R pin connected to GND). Thus,
         we are only interested in reading the left channel of the I2S
         interface. */
      .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
      /* We are using a standard I2S interface. */
      .communication_format = I2S_COMM_FORMAT_STAND_I2S,
      /* Interrupt level for the I2S hardware interrupt. */
      .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
      /* Using 3 internal buffers with 300 samples each. */
      .dma_buf_count = 3,
      .dma_buf_len = 300,
      /* No need for the higher resolution APLL clock. */
      .use_apll = false,
      /* No need for the auto clear feature since we are not transmitting. */
      .tx_desc_auto_clear = false,
  };

  i2s_pin_config_t pin_config = {
      /* No master clock needed. We are "only" using the bit clock. */
      .mck_io_num = I2S_PIN_NO_CHANGE,
      .bck_io_num = 8,
      /* Word select line. Selects the left or right channel of the slave
         device. */
      .ws_io_num = 9,
      /* Data out is not needed as we are only working with data source */
      .data_out_num = I2S_PIN_NO_CHANGE,
      /* Data in from slave device. */
      .data_in_num = 10};

  esp_err_t ret = ESP_OK;

  ret = i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  if (ret != ESP_OK) {
    TF_LITE_REPORT_ERROR(error_reporter, "Error in i2s_driver_install().");
    return kTfLiteError;
  }

  ret = i2s_set_pin(I2S_NUM_0, &pin_config);
  if (ret != ESP_OK) {
    TF_LITE_REPORT_ERROR(error_reporter, "Error in i2s_set_pin().");
    return kTfLiteError;
  }

  ret = i2s_zero_dma_buffer(I2S_NUM_0);
  if (ret != ESP_OK) {
    TF_LITE_REPORT_ERROR(error_reporter, "Error in i2s_zero_dma_buffer().");
    return kTfLiteError;
  }

  buf_handle = xRingbufferCreate(1024 * 64, RINGBUF_TYPE_BYTEBUF);
  if (buf_handle == NULL) {
    TF_LITE_REPORT_ERROR(error_reporter, "Error in xRingbufferCreate().");
    return kTfLiteError;
  }

  // TODO(fabianpedd): Error checking on ret val.
  xTaskCreate(CaptureAudioSamples, "CaptureAudioSamples", 1024 * 64,
              (void*)error_reporter, 10, NULL);

  return kTfLiteOk;
}

TfLiteStatus GetAudioData(tflite::ErrorReporter* error_reporter,
                          uint32_t requested_size, uint32_t* actual_size,
                          int8_t* data) {
  *actual_size = 0;

  // Peak into the Ringbuffer.
  uint32_t bytes_waiting = 0;
  vRingbufferGetInfo(buf_handle, NULL, NULL, NULL, NULL,
                     (UBaseType_t*)&bytes_waiting);

  // Check if we actually have the requested amount of bytes available. If yes,
  // get the data. If not, simply return zero.
  if (bytes_waiting >= requested_size) {
    // Get the data from the Ringbuffer.
    uint32_t bytes_received = 0;
    int8_t* buf_data =
        (int8_t*)xRingbufferReceiveUpTo(buf_handle, (size_t*)&bytes_received,
                                        pdMS_TO_TICKS(100), requested_size);

    // Check whether we have encountered a wraparound in the Ringbuffer. If so,
    // we need to read a second time in order to retreive all data. See here
    // https://docs.espressif.com/projects/esp-idf/en/v4.4/esp32c3/api-reference/system/freertos_additions.html#_CPPv422xRingbufferReceiveUpTo15RingbufHandle_tP6size_t10TickType_t6size_t
    if (buf_data != NULL && bytes_received < requested_size) {
      // Copy the data from the Ringbuffer into output buffer and free the
      // ringbuffer data.
      memcpy(data, buf_data, bytes_received);
      vRingbufferReturnItem(buf_handle, (void*)buf_data);

      // Move the data pointer and adjust the amount of data needed accordingly.
      data += bytes_received;
      *actual_size += bytes_received;
      requested_size -= bytes_received;

      // TF_LITE_REPORT_ERROR(
      //     error_reporter,
      //     "Warning: Encountered wraparound and only read %d of %d bytes.",
      //     *actual_size, requested_size);

      // Get the rest of the data from the top of the Ringbuffer.
      buf_data =
          (int8_t*)xRingbufferReceiveUpTo(buf_handle, (size_t*)&bytes_received,
                                          pdMS_TO_TICKS(100), requested_size);
    }

    if (buf_data != NULL && bytes_received == requested_size) {
      // Copy the data from the Ringbuffer into output buffer and free the
      // ringbuffer data.
      memcpy(data, buf_data, bytes_received);
      vRingbufferReturnItem(buf_handle, (void*)buf_data);

      *actual_size += bytes_received;
      return kTfLiteOk;

    } else {
      TF_LITE_REPORT_ERROR(
          error_reporter,
          "ERROR: Only read %d of %d bytes from Ringbuffer. Something went "
          "wrong, as there should be enough data available.",
          *actual_size, requested_size);
      // TODO(fabianpedd): Should not be needed here. But maybe reintroduce as a
      // safety measure?
      // vRingbufferReturnItem(buf_handle, (void *)buf_data);
      return kTfLiteError;
    }
  } else {
    data = NULL;
    return kTfLiteOk;
  }
}
