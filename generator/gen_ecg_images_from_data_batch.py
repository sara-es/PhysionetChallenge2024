import os, sys, argparse
import random
import csv

from tqdm import tqdm

from generator.helper_functions import find_records
from generator.gen_ecg_image_from_data import run_single_file
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")


def run(args, records_to_process=None):
    random.seed(args.seed)

    i = 0
    if records_to_process is None:
        full_header_files, full_recording_files = find_records(args.input_directory, args.output_directory)
    else:
        full_header_files = [r + '.hea' for r in records_to_process]
        full_recording_files = [r + '.dat' for r in records_to_process]

    for full_header_file, full_recording_file in tqdm(zip(full_header_files, full_recording_files),
                                                      total=len(full_header_files)):
        filename = full_recording_file
        header = full_header_file
        args.input_file = os.path.join(args.input_directory, filename)
        args.header_file = os.path.join(args.input_directory, header)
        args.start_index = -1

        folder_struct_list = full_header_file.split('/')[:-1]
        args.output_directory = os.path.join(args.output_directory, '/'.join(folder_struct_list))
        args.encoding = os.path.split(os.path.splitext(filename)[0])[1]

        i += run_single_file(args)

        if args.max_num_images != -1 and i >= args.max_num_images:
            break


# if __name__ == '__main__':
#     path = os.path.join(os.getcwd(), sys.argv[0])
#     parentPath = os.path.dirname(path)
#     os.chdir(parentPath)
#     run(get_parser().parse_args(sys.argv[1:]))
