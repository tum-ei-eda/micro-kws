#include "debug.h"

#include <cstring>

// clang-format off
#include "model_settings.h"
// clang-format on

#include "driver/uart.h"
#include "freertos/task.h"
#include "esp_timer.h"

// TODO(fabianpedd): If we have two cores, like on the ESP32, we can run the
// DebugWorker() task on a different core than the main task that runs the
// inference. This would allow debugging with minimal runtime overhead, as only
// the memcpy operation would be added to the main task.

static RingbufHandle_t buf_handle = NULL;

typedef struct __attribute__((packed)) {
  int8_t feature_data[feature_element_count];
  int8_t category_data[category_count];
} debug_data_t;

static void DebugWorker(void* arg) {
  tflite::ErrorReporter* error_reporter = (tflite::ErrorReporter*)arg;

  while (true) {
    uint32_t item_size = 0;
    debug_data_t* data = (debug_data_t*)xRingbufferReceive(
        buf_handle, (size_t*)&item_size, pdMS_TO_TICKS(500));

    if (data == NULL) {
      TF_LITE_REPORT_ERROR(error_reporter,
                           "ERROR: In xRingbufferReceive() in DebugWorker().");
      return;
    }

    if (item_size != sizeof(debug_data_t) ||
        // TODO(fabianpedd): Is only a static sanity check. Remove me at some
        // point.
        feature_element_count + category_count != sizeof(debug_data_t)) {
      TF_LITE_REPORT_ERROR(
          error_reporter,
          "ERROR: Received size %d does not match expected size %d.", item_size,
          sizeof(debug_data_t));
      return;
    }
    // else {
    //   MicroPrintf("Received %d bytes from ringbuffer.", item_size);
    // }

    // TODO: Maybe include the size of the packet or smth simiar (maybe even a
    // checksum).

    // Last five bytes will be 0, 1, 2, 3, 4 in order to help
    // sycronize the UART packet once sent to the PC.
    uint8_t packet_footer[] = {0, 1, 2, 3, 4};

    // Total size of packet consists of data and packet footer size
    uint32_t total_packet_size =
        feature_element_count + category_count + sizeof(packet_footer);
    int8_t packet_buffer[total_packet_size] = {0};

    // Copy the data from the UART into the packet buffer and append the packet
    // footer to the end.
    memcpy(packet_buffer, data, item_size);
    memcpy(packet_buffer + feature_element_count + category_count,
           packet_footer, sizeof(packet_footer));

    // Now we can free the data from the internal UART buffer.
    vRingbufferReturnItem(buf_handle, (void*)data);

    // Send debug data via the auxiliary UART to the Python script.
    uint32_t data_sent =
        uart_write_bytes(UART_NUM_1, (int8_t*)packet_buffer, total_packet_size);

    if (data_sent < total_packet_size) {
      TF_LITE_REPORT_ERROR(error_reporter,
                           "ERROR: Only sent %d of %d bytes via debug UART.",
                           data_sent, total_packet_size);
      return;
    }
    // else {
    //   MicroPrintf("Sent %d bytes via UART.", data_sent);
    // }
  }
}

static void DebugPrintStats(void* arg) {
  (void)arg;
  while (true) {
    char buffer[1024] = {0};
    // TODO(fabianpedd): The vTaskGetRunTimeStats() appears to be broken, in
    // that it only prints up to a certain number of characters. If you have
    // more tasks active they will not be displayed.
    vTaskGetRunTimeStats(buffer);
    MicroPrintf("%s\n", buffer);
    vTaskDelay(100 / portTICK_PERIOD_MS);
  }
}

TfLiteStatus DebugInit(tflite::ErrorReporter* error_reporter,
                       uint32_t baud_rate, uint32_t tx_pin) {
  uart_config_t uart_config = {
      .baud_rate = (int)baud_rate,
      .data_bits = UART_DATA_8_BITS,
      .parity = UART_PARITY_DISABLE,
      .stop_bits = UART_STOP_BITS_1,
      .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
      .source_clk = UART_SCLK_APB,
  };

  if (uart_driver_install(UART_NUM_1, 2048 * 2, 0, 0, NULL, 0) != ESP_OK) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "ERROR: In uart_driver_install() in DebugInit().");
    return kTfLiteError;
  }

  if (uart_param_config(UART_NUM_1, &uart_config)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "ERROR: In uart_param_config() in DebugInit().");
    return kTfLiteError;
  }

  if (uart_set_pin(UART_NUM_1, 7, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE,
                   UART_PIN_NO_CHANGE)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "ERROR: In uart_set_pin() in DebugInit().");
    return kTfLiteError;
  }

  buf_handle = xRingbufferCreate(1024 * 10, RINGBUF_TYPE_NOSPLIT);
  if (buf_handle == NULL) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "ERROR: In xRingbufferCreate() in DebugInit().");
    return kTfLiteError;
  }

  if (xTaskCreate(DebugWorker, "DebugWorker", 1024 * 10, (void*)error_reporter,
                  10, NULL) != pdPASS) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "ERROR: In xTaskCreate(DebugWorker) in DebugInit().");
    return kTfLiteError;
  }

  // if (xTaskCreate(DebugPrintStats, "DebugPrintStats", 1024 * 8, NULL, 10,
  //                 NULL) != pdPASS) {
  //   TF_LITE_REPORT_ERROR(
  //       error_reporter,
  //       "ERROR: In xTaskCreate(DebugPrintStats) in DebugInit().");
  //   return kTfLiteError;
  // }

  return kTfLiteOk;
}

TfLiteStatus DebugRun(tflite::ErrorReporter* error_reporter,
                      int8_t* feature_data, int8_t* category_data) {
  for (uint32_t i = 0; i < category_count; i++) {
    printf("%s:%4d  ", category_labels[i], category_data[i]);
  }

  static uint64_t last_time = esp_timer_get_time();
  uint64_t this_time = esp_timer_get_time();
  int32_t delta_time = (uint32_t)(this_time - last_time) / 1000;
  printf("\t Δ%dms \n", delta_time);
  last_time = this_time;

  debug_data_t debug_data;
  memcpy(debug_data.feature_data, feature_data, feature_element_count);
  memcpy(debug_data.category_data, category_data, category_count);

  if (xRingbufferSend(buf_handle, (void*)&debug_data, sizeof(debug_data),
                      pdMS_TO_TICKS(100)) != pdTRUE) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "ERROR: In xRingbufferSend() in DebugRun(). Most "
                         "likely the Ringbuffer is full.");
    return kTfLiteError;
  }

  return kTfLiteOk;
}
