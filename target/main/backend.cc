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

#include "backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>

#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "gpio.h"
#include "model_settings.h"

// TODO(fabianpedd): In order to improve the detection accuracy we could
// introduce a rejection difference threshold. The difference between the top
// category and the 2nd top category would have to be higher than this
// threshold. If smaller, we would simply categorize silence or unknown. This
// would help to supress detections where the NN is unsure about two categories.
// However, this might lead to worse detection sensitivity.

// TODO(fabianpedd): Make threshold (or better history length) independent from
// inference performance / inferences per second.
constexpr size_t posterior_supression_ms =
    CONFIG_MICRO_KWS_POSTERIOR_SUPRESSION_MS;
constexpr size_t posterior_history_length =
    CONFIG_MICRO_KWS_POSTERIOR_HISTORY_LENGTH;
constexpr size_t posterior_trigger_threshold =
    CONFIG_MICRO_KWS_POSTERIOR_TRIGGER_THRESHOLD_SINGLE *
    CONFIG_MICRO_KWS_POSTERIOR_HISTORY_LENGTH;

esp_err_t KeywordCallback(const char* category) {
  /****************************************************************************/
  /************************ Student work starts here **************************/
  /****************************************************************************/

  // Your task is to implement RGB feedback indicating which category was last
  // detected. Unkown should be indicated with orange color. For silence the
  // LED should stay off!
  //
  // Please use RED and GREEN as indicator for  the keywords no and yes.
  // Colors for the other 6 categories can be taken from gpio.h

  // TODO(fabianpedd): Remove sample solution

  // printf("Detected a new keyword %s\n", category);
  if (strcmp(category, CONFIG_MICRO_KWS_CLASS_LABEL_0) == 0) {
    SetLEDColor(LED_RGB_BLACK);
  } else if (strcmp(category, CONFIG_MICRO_KWS_CLASS_LABEL_1) == 0) {
    SetLEDColor(LED_RGB_ORANGE);
  } else if (strcmp(category, CONFIG_MICRO_KWS_CLASS_LABEL_2) == 0) {
    SetLEDColor(LED_RGB_GREEN);
  } else if (strcmp(category, CONFIG_MICRO_KWS_CLASS_LABEL_3) == 0) {
    SetLEDColor(LED_RGB_RED);
#ifdef CONFIG_MICRO_KWS_CLASS_LABEL_4
  } else if (strcmp(category, CONFIG_MICRO_KWS_CLASS_LABEL_4) == 0) {
    SetLEDColor(LED_RGB_BLUE);
#endif  // CONFIG_MICRO_KWS_CLASS_LABEL_4
#ifdef CONFIG_MICRO_KWS_CLASS_LABEL_5
  } else if (strcmp(category, CONFIG_MICRO_KWS_CLASS_LABEL_5) == 0) {
    SetLEDColor(LED_RGB_YELLOW);
#endif  // CONFIG_MICRO_KWS_CLASS_LABEL_5
#ifdef CONFIG_MICRO_KWS_CLASS_LABEL_6
  } else if (strcmp(category, CONFIG_MICRO_KWS_CLASS_LABEL_6) == 0) {
    SetLEDColor(LED_RGB_CYAN);
#endif  // CONFIG_MICRO_KWS_CLASS_LABEL_6
#ifdef CONFIG_MICRO_KWS_CLASS_LABEL_7
  } else if (strcmp(category, CONFIG_MICRO_KWS_CLASS_LABEL_7) == 0) {
    SetLEDColor(LED_RGB_MAGENTA);
#endif  // CONFIG_MICRO_KWS_CLASS_LABEL_7
#ifdef CONFIG_MICRO_KWS_CLASS_LABEL_8
  } else if (strcmp(category, CONFIG_MICRO_KWS_CLASS_LABEL_8) == 0) {
    SetLEDColor(LED_RGB_PURPLE);
#endif  // CONFIG_MICRO_KWS_CLASS_LABEL_8
#ifdef CONFIG_MICRO_KWS_CLASS_LABEL_9
  } else if (strcmp(category, CONFIG_MICRO_KWS_CLASS_LABEL_9) == 0) {
    SetLEDColor(LED_RGB_MINT);
#endif  // CONFIG_MICRO_KWS_CLASS_LABEL_9
  } else {
    SetLEDColor(LED_RGB_BLACK);
    return ESP_FAIL;
  }

  /****************************************************************************/
  /************************ Student work stops here ***************************/
  /****************************************************************************/
  return ESP_OK;
}

esp_err_t HandlePosteriors(uint8_t new_posteriors[category_count],
                           size_t* top_category_index) {
  // A 'posterior_history_length x category_count' matrix of past posteriors.
  static uint8_t posterior_history[posterior_history_length][category_count] = {
      0};
  // For efficiency reasons we keep an accumulator (transforms the problem of
  // calculating the top category from O(posterior_history_length) to O(2) for
  // each category).
  static uint32_t posterior_accumulator[category_count] = {0};
  static size_t posterior_history_pointer = 0;
  size_t top_canidate_index = 0;
  // Iterate over all category values in posterior_history and replace with new
  // posteriors.
  for (size_t i = 0; i < category_count; i++) {
    posterior_accumulator[i] -= posterior_history[posterior_history_pointer][i];
    posterior_accumulator[i] += new_posteriors[i];
    posterior_history[posterior_history_pointer][i] = new_posteriors[i];
    if (posterior_accumulator[i] > posterior_accumulator[top_canidate_index])
      top_canidate_index = i;
  }
  // Bump posterior history pointer and handle wraparound.
  if (++posterior_history_pointer >= posterior_history_length)
    posterior_history_pointer = 0;

  // Keeps track of last top posterior category.
  static size_t top_posterior_index = 0;
  // Keeps track of last detection time of top posterior category (need to
  // divide by 1000 because esp_timer_get_time() returns microseconds).
  static uint32_t top_posterior_ms = (uint32_t)(esp_timer_get_time() / 1000);
  // Top posterior must be larger than threshold. Additionally, top posterior
  // must either be a new posterior category or last invocation must be older
  // than suppression value.
  if (posterior_accumulator[top_canidate_index] >=
          posterior_trigger_threshold &&
      (top_posterior_index != top_canidate_index ||
       (uint32_t)(esp_timer_get_time() / 1000) >=
           top_posterior_ms + posterior_supression_ms)) {
    top_posterior_index = top_canidate_index;
    top_posterior_ms = (uint32_t)(esp_timer_get_time() / 1000);
    KeywordCallback(category_labels[top_posterior_index]);
  }
  *top_category_index = top_posterior_index;
  return ESP_OK;
}
