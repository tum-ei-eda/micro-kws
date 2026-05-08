ARG BASE_IMAGE=espressif/idf:v6.0.1

FROM ${BASE_IMAGE} AS philippvk/microkws
LABEL maintainer="Philipp van Kempen"

ENV DEBIAN_FRONTEND=noninteractive

RUN pip install --no-cache-dir \
    ipykernel \
    jupyter \
    notebook

RUN cd /tmp/ \
    && wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz \
    && mkdir -p speech_dataset \
    && tar xf speech_commands_v0.02.tar.gz -C speech_dataset \

ENV SPEECH_COMMANDS_DIR=/tmp/speech_dataset
