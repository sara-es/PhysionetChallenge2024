# Code from team "The Easy Geese" for the George B. Moody PhysioNet Challenge 2024
*Team members: Matti Kaisti, Tuija Leinonen, George Searle, Sara Summerton, Dave C. Wong*

## What's in this repository?

This repository contains a Python entry for the [George B. Moody PhysioNet Challenge 2024](https://physionetchallenges.org/2024/), as well as two sets of 15 sample images, wfdb data and header files that can be used to quickly test the code.

## Quickstart guide
First, download this repository and install the requirements:

    pip install -r requirements.txt
    
A small subset of data and labels have been provided in the `tiny_testset` folder. If you simply want to test that your code runs on as few as 15 samples, run the following commands:

    python train_model.py -d tiny_testset/lr_gt -m model -v

    python run_model.py -d tiny_testset/lr_hidden_clean -m model -o tiny_testset/test_outputs

    python evaluate_model.py -d tiny_testset/lr_gt -o tiny_testset/test_outputs

`tiny_testset` contains a number of folders with images generated *from the same data source* but with different parameters:
- `hr_gt`: clean images (500 Hz .dat files)
- `lr_gt`: clean images (100 Hz .dat files)
- `lr_gt_noisy`: some distortions (rotations, shadow) (100 Hz .dat files)
- `lr_gt_noisy_bw`: distortions plus random grid color, some black and white images (100 Hz .dat files)
- `lr_gt_refpulse`: clean but with reference pulses on all images (100 Hz .dat files)
All of these include the ground truth signal .dat files so we can check model performance.

To make sure that the model actually runs inference directly from generated images, there's `lr_hidden_clean` (100 Hz .hea files) and `hr_hidden_clean` (500 Hz .hea files) with labels removed.

The full instructions, copied from the [Physionet python example code](https://github.com/physionetchallenges/python-example-2024) and [Physionet scoring code](https://github.com/physionetchallenges/evaluation-2024), have been copied below. 

# Instructions for python template code for the George B. Moody PhysioNet Challenge 2024
## How do I run these scripts?

First, you can download and create data for these scripts by following the instructions in the following section.

Second, you can install the dependencies for these scripts by creating a Docker image (see below) or [virtual environment](https://docs.python.org/3/library/venv.html) and running

    pip install -r requirements.txt

You can train your model(s) by running

    python train_model.py -d training_data -m model

where

- `training_data` (input; required) is a folder with the training data files, including the images and diagnoses (you can use the `ptb-xl/records100/00000` folder from the below steps); and
- `model` (output; required) is a folder for saving your model(s).

You can run your trained model(s) by running

    python run_model.py -d test_data -m model -o test_outputs

where

- `test_data` (input; required) is a folder with the validation or test data files, excluding the images and diagnoses (you can use the `ptb-xl/records100_hidden/00000` folder from the below steps, but it would be better to repeat these steps on a new subset of the data that you did not use to train your model);
- `model` (input; required) is a folder for loading your model(s); and
- `test_outputs` is a folder for saving your model outputs.

The [Challenge website](https://physionetchallenges.org/2024/#data) provides a training database with a description of the contents and structure of the data files.

The evaluation code has been copied from the provided [evaluation code](https://github.com/physionetchallenges/evaluation-2024). This and other test scripts are located in the `evaluation` folder. You can evaluate the model's performance by running

    python evaluate_model.py -d labels -o test_outputs -s scores.csv

from within the evaluation folder, where

- `labels` is a folder with labels for the data, such as the training database on the PhysioNet webpage (you can use the `ptb-xl/records100/00000` folder from the below steps, but it would be better to repeat these steps on a new subset of the data that you did not use to train your model);
- `test_outputs` is a folder containing files with your model's outputs for the data; and
- `scores.csv` (optional) is file with a collection of scores for your model.

## How do I create data for these scripts?

You can use the scripts in this repository to generate synthetic ECG images for the [PTB-XL dataset](https://www.nature.com/articles/s41597-020-0495-6). You will need to generate or otherwise obtain ECG images before running the above steps.

1. Download (and unzip) the [PTB-XL dataset](https://physionet.org/content/ptb-xl/). We will use `ptb-xl` as the folder name that contains the data for these commands (the full folder name for the PTB-XL dataset is currently `ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3`), but you can replace it with the absolute or relative path on your machine.

2. Add information from various spreadsheets from the PTB-XL dataset to the WFDB header files:

        python prepare_ptbxl_data.py \
            -i ptb-xl/records100/00000 \
            -d ptb-xl/ptbxl_database.csv \
            -s ptb-xl/scp_statements.csv \
            -o ptb-xl/records100/00000

3. [Generate synthetic ECG images](https://github.com/alphanumericslab/ecg-image-kit/tree/main/codes/ecg-image-generator) on the dataset:

        python gen_ecg_images_from_data_batch.py \
            -i ptb-xl/records100/00000 \
            -o ptb-xl/records100/00000 \
            --print_header

4. Add the file locations for the synthetic ECG images to the WFDB header files. (The expected image filenames for record `12345.png` are of the form `12345-0.png`, `12345-1.png`, etc., which should be in the same folder.) You can use the `ptb-xl/records100/00000` folder for the `train_model` step:

        python add_image_filenames.py \
            -i ptb-xl/records100/00000 \
            -o ptb-xl/records100/00000

5. Remove the waveforms, certain information about the waveforms, and the demographics and diagnoses to create a version of the data for inference. You can use the `ptb-xl/records100_hidden/00000` folder for the `run_model` step, but it would be better to repeat the above steps on a new subset of the data that you will not use to train your model:

        python remove_hidden_data.py \
            -i ptb-xl/records100/00000 \
            -o ptb-xl/records100_hidden/00000

## Which scripts I can edit?

Please edit the following script to add your code:

* `team_code.py` is a script with functions for training and running your trained model(s).

Please do **not** edit the following scripts. We will use the unedited versions of these scripts when running your code:

* `train_model.py` is a script for training your model(s).
* `run_model.py` is a script for running your trained model(s).
* `helper_code.py` is a script with helper functions that we used for our code. You are welcome to use them in your code.

These scripts must remain in the root path of your repository, but you can put other scripts and other files elsewhere in your repository.

## How do I train, save, load, and run my model?

You can choose to create waveform reconstruction and/or classification models.

To train and save your model(s), please edit the `train_digitization_model` and `train_diagnosis_model` functions in the `team_code.py` script. Please do not edit the input or output arguments of these function.

To load and run your trained model(s), please edit the `load_digitization_model`, `load_diagnosis_model`, `run_digitization_model`, and `run_diagnosis_model` functions in the `team_code.py` script. Please do not edit the input or output arguments of these functions.

## How do I run these scripts in Docker?

Docker and similar platforms allow you to containerize and package your code with specific dependencies so that your code can be reliably run in other computational environments.

To increase the likelihood that we can run your code, please [install](https://docs.docker.com/get-docker/) Docker, build a Docker image from your code, and run it on the training data. To quickly check your code for bugs, you may want to run it on a small subset of the training data, such as 100 records.

If you have trouble running your code, then please try the follow steps to run the example code.

1. Create a folder `example` in your home directory with several subfolders.

        user@computer:~$ cd ~/
        user@computer:~$ mkdir example
        user@computer:~$ cd example
        user@computer:~/example$ mkdir training_data test_data model test_outputs

2. Download the training data from the [Challenge website](https://physionetchallenges.org/2024/#data). Put some of the training data in `training_data` and `test_data`. You can use some of the training data to check your code (and you should perform cross-validation on the training data to evaluate your algorithm).

3. Download or clone this repository in your terminal.

        user@computer:~/example$ git clone https://github.com/physionetchallenges/python-example-2024.git

4. Build a Docker image and run the example code in your terminal.

        user@computer:~/example$ ls
        model  python-example-2024  test_data  test_outputs  training_data

        user@computer:~/example$ cd python-example-2024/

        user@computer:~/example/python-example-2024$ docker build -t image .

        Sending build context to Docker daemon  [...]kB
        [...]
        Successfully tagged image:latest

        user@computer:~/example/python-example-2024$ docker run -it -v ~/example/model:/challenge/model -v ~/example/test_data:/challenge/test_data -v ~/example/test_outputs:/challenge/test_outputs -v ~/example/training_data:/challenge/training_data image bash

        root@[...]:/challenge# ls
            Dockerfile             README.md         test_outputs
            evaluate_model.py      requirements.txt  training_data
            helper_code.py         team_code.py      train_model.py
            LICENSE                run_model.py      [...]

        root@[...]:/challenge# python train_model.py -d training_data -m model -v

        root@[...]:/challenge# python run_model.py -d test_data -m model -o test_outputs -v

        root@[...]:/challenge# python evaluate_model.py -d test_data -o test_outputs
        [...]

        root@[...]:/challenge# exit
        Exit

## What else do I need?

This repository does not include data or the code for generating ECG images. Please see the above instructions for how to download and prepare the data.

# Scoring code for the George B. Moody PhysioNet Challenge 2024

This repository contains the Python and MATLAB evaluation code for the George B. Moody PhysioNet Challenge 2024.

The `evaluate_model` script evaluates the outputs of your models using the evaluation metric that is described on the [webpage](https://physionetchallenges.org/2024/) for the 2024 Challenge. This script reports multiple evaluation metrics, so check the [scoring section](https://physionetchallenges.org/2024/#scoring) of the webpage to see how we evaluate and rank your models.

## Python

You can run the Python evaluation code by installing the NumPy package and running the following command in your terminal:

    python evaluate_model.py -d labels -o outputs -s scores.csv

where

- `labels` (input; required) is a folder with labels for the data, such as the [training data](https://physionetchallenges.org/2024/#data) on the PhysioNet Challenge webpage;
- `outputs` (input; required) is a folder containing files with your model's outputs for the data; and
- `scores.csv` (output; optional) is a collection of scores for your model.

## Troubleshooting

Unable to run this code with your code? Try one of the [example codes](https://physionetchallenges.org/2024/#submissions) on the [training data](https://physionetchallenges.org/2024/#data). Unable to install or run Python? Try [Python](https://www.python.org/downloads/), [Anaconda](https://www.anaconda.com/products/individual), or your package manager.

## How do I learn more? How do I share more?

Please see the [Challenge website](https://physionetchallenges.org/2024/) for more details. Please post questions and concerns on the [Challenge discussion forum](https://groups.google.com/forum/#!forum/physionet-challenges). Please do not make pull requests, which may share information about your approach.

## Useful links

* [Challenge website](https://physionetchallenges.org/2024/)
* [MATLAB example code](https://github.com/physionetchallenges/matlab-example-2024)
* [Evaluation code](https://github.com/physionetchallenges/evaluation-2024)
* [Frequently asked questions (FAQ) for this year's Challenge](https://physionetchallenges.org/2024/faq/)
* [Frequently asked questions (FAQ) about the Challenges in general](https://physionetchallenges.org/faq/)
