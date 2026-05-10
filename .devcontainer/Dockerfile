ARG BASE_IMAGE=espressif/idf:v6.0.1
FROM ${BASE_IMAGE} AS microkws
LABEL maintainer="Philipp van Kempen"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-pip python3-venv \
    && apt-get clean \
    && rm -rf /var/cache/apt/archives/ \
    && rm -rf /var/lib/apt/lists

ENV VENV_DIR=/opt/microkws/venv

# Create virtual environment
RUN python3 -m venv $VENV_DIR

# Activate venv by default
ENV PATH="$VENV_DIR/bin:${PATH}"

# Upgrade pip inside venv
RUN pip install --upgrade pip

# Install Python tools
RUN pip install --no-cache-dir \
    ipykernel \
    jupyter \
    notebook

ENV SPEECH_COMMANDS_DIR=/opt/microkws/speech_dataset

RUN cd /tmp/ \
    && wget --no-verbose http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz \
    && mkdir -p $SPEECH_COMMANDS_DIR \
    && tar xf speech_commands_v0.02.tar.gz -C $SPEECH_COMMANDS_DIR

RUN pip install --no-cache-dir \
    tensorflow matplotlib

RUN pip install --no-cache-dir \
    pyusb pyserial tflite

RUN pip install --no-cache-dir \
    tvm==0.16.0 -f https://philippvk.github.io/tlc-pack.github.io/wheels
