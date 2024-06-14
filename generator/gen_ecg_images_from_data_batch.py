import os, sys, argparse
import random
import csv

from tqdm import tqdm

from helper_functions import find_records
from gen_ecg_image_from_data import run_single_file
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")


def run(args):
    random.seed(args.seed)

    if not os.path.isabs(args.input_directory):
        args.input_directory = os.path.normpath(os.path.join(os.getcwd(), args.input_directory))
    if not os.path.isabs(args.output_directory):
        original_output_dir = os.path.normpath(os.path.join(os.getcwd(), args.output_directory))
    else:
        original_output_dir = args.output_directory

    if os.path.exists(args.input_directory) == False or os.path.isdir(args.input_directory) == False:
        raise Exception("The input directory does not exist, Please re-check the input arguments!")

    if not os.path.exists(original_output_dir):
        os.makedirs(original_output_dir)

    i = 0
    full_header_files, full_recording_files = find_records(args.input_directory, original_output_dir)

    for full_header_file, full_recording_file in tqdm(zip(full_header_files, full_recording_files),
                                                      total=len(full_header_files),
                                                      desc='Generating images from data'):
        filename = full_recording_file
        header = full_header_file
        args.input_file = os.path.join(args.input_directory, filename)
        args.header_file = os.path.join(args.input_directory, header)
        args.start_index = -1

        folder_struct_list = full_header_file.split('/')[:-1]
        args.output_directory = os.path.join(original_output_dir, '/'.join(folder_struct_list))
        args.encoding = os.path.split(os.path.splitext(filename)[0])[1]

        i += run_single_file(args)

        if args.max_num_images != -1 and i >= args.max_num_images:
            break


# if __name__ == '__main__':
#     path = os.path.join(os.getcwd(), sys.argv[0])
#     parentPath = os.path.dirname(path)
#     os.chdir(parentPath)
#     run(get_parser().parse_args(sys.argv[1:]))
