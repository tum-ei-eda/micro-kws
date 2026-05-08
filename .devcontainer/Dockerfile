ARG BASE_IMAGE=espressif/idf:v6.0.1

FROM ${BASE_IMAGE} AS microkws
LABEL maintainer="Philipp van Kempen"

ENV DEBIAN_FRONTEND=noninteractive

RUN pip install --no-cache-dir \
    ipykernel \
    jupyter \
    notebook
