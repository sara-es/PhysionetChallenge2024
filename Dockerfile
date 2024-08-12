FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install your dependencies here using apt install, etc.
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6

## Include the following line if you have a requirements.txt file.
RUN pip install -r requirements.txt
