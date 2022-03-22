idf_component_register(
    SRCS
    audio.cc
    debug.cc
    frontend.cc
    main.cc
    model_settings.cc
    model.cc
    PRIV_REQUIRES
    tflite-lib
    spi_flash
    INCLUDE_DIRS
    "."
)
idf_build_set_property(COMPILE_OPTIONS "-Wno-missing-field-initializers" APPEND)
