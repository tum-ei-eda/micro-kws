#ifndef MODEL_SETTINGS_H
#define MODEL_SETTINGS_H

#include <cstdint>

// Create a tensor area with 30KB.
constexpr int32_t tensor_arena_size = 32 * 1024;

// The size of the input time series data we pass to the FFT to produce the
// frequency information. This has to be a power of two, and since we're dealing
// with 30ms of 16KHz inputs, which means 480 samples, this is the next larger
// value, i.e. 512.
constexpr int32_t max_audio_sample_size = 512;
constexpr int32_t audio_sample_frequency = 16000;

// The feature (powerspectrum image) on which the convolutional neural network
// operates on has 49 slices, each containing 40 grayscale pixels. So basically
// a 49 by 40 grayscale picture.
constexpr int32_t feature_slice_size = 40;
constexpr int32_t feature_slize_count = 49;
constexpr int32_t feature_element_count =
    feature_slice_size * feature_slize_count;

constexpr int32_t feature_slice_stride_ms = 20;
constexpr int32_t feature_slice_duration_ms = 30;

constexpr int32_t category_count = 4;
extern const char* category_labels[category_count];

#endif  // MODEL_SETTINGS_H
