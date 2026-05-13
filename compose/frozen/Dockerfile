FROM --platform=linux/amd64 ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:fenics-packages/fenics \
    && apt-get update \
    && apt-get install -y fenics \
    && apt-get clean

ENTRYPOINT ["python3"]