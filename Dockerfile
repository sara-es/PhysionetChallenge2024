FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu20.04

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install your dependencies here using apt install, etc.
# Install python3.11
RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update && apt install -y python3.11

# Python 3.8.1 by default, so force upgrade to 3.11
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 2 \
    && update-alternatives --set python /usr/bin/python3.11 \
    && apt-get install -y python3.11-distutils 

# Dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 libsm6 libxrender1 libxext6

# Install pip
RUN apt-get -y update && apt-get -y install curl
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

## Include the following line if you have a requirements.txt file.
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt