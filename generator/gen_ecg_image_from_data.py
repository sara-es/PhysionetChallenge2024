import os, sys, argparse, json
import random
import csv
from PIL import Image
import numpy as np
from scipy.stats import bernoulli
from generator.extract_leads import get_paper_ecg
from generator.CreasesWrinkles.creases import get_creased
from generator.augment import get_augment
import warnings
from generator.helper_functions import read_config_file

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")


def writeCSV(args):
    csv_file_path = os.path.join(args.output_directory, 'Coordinates.csv')
    if not os.path.isfile(csv_file_path):
        with open(csv_file_path, 'a') as ground_truth_file:
            writer = csv.writer(ground_truth_file)
            if args.start_index != -1:
                writer.writerow(["Filename", "class", "x_center", "y_center", "width", "height"])

    grid_file_path = os.path.join(args.output_directory, 'gridsizes.csv')
    if not os.path.isfile(grid_file_path):
        with open(grid_file_path, 'a') as gridsize_file:
            writer = csv.writer(gridsize_file)
            if args.start_index != -1:
                writer.writerow(["filename", "xgrid", "ygrid", "lead_name", "start", "end"])


def run_single_file(args):
    if hasattr(args, 'st'):
        random.seed(args.seed)
        args.encoding = args.input_file

    filename = args.input_file
    header = args.header_file
    resolution = random.choice(range(50, args.resolution + 1)) if args.random_resolution else args.resolution
    padding = random.choice(range(0, args.pad_inches + 1)) if args.random_padding else args.pad_inches

    papersize = ''
    lead = args.add_lead_names

    bernoulli_dc = bernoulli(args.calibration_pulse)
    bernoulli_bw = bernoulli(args.random_bw)
    bernoulli_grid = bernoulli(args.random_grid_present)
    if args.print_header:
        bernoulli_add_print = bernoulli(1)
    else:
        bernoulli_add_print = bernoulli(args.random_print_header)

    font = os.path.join('Fonts', random.choice(os.listdir(os.path.join("generator", "Fonts"))))

    if args.random_bw == 0:
        if not args.random_grid_color:
            standard_colours = args.standard_grid_color
        else:
            standard_colours = -1
    else:
        standard_colours = False

    configs = read_config_file(os.path.join(os.getcwd(), "generator", args.config_file))

    out_array = get_paper_ecg(input_file=filename, header_file=header, configs=configs,
                              mask_unplotted_samples=args.mask_unplotted_samples, start_index=args.start_index,
                              store_configs=args.store_config, store_text_bbox=args.lead_name_bbox,
                              output_directory=args.output_directory, resolution=resolution, papersize=papersize,
                              add_lead_names=lead, add_dc_pulse=bernoulli_dc, add_bw=bernoulli_bw,
                              show_grid=bernoulli_grid, add_print=bernoulli_add_print, pad_inches=padding,
                              font_type=font, standard_colours=standard_colours, full_mode=args.full_mode,
                              bbox=args.lead_bbox, columns=args.num_columns, seed=args.seed, 
                              single_channel=args.single_channel, copy_data_files=args.copy_data_files)

    for out in out_array:
        if args.store_config:
            rec_tail, extn = os.path.splitext(out)
            with open(rec_tail + '.json', 'r') as file:
                json_dict = json.load(file)
        else:
            json_dict = None
        if args.fully_random:
            wrinkles = random.choice((True, False))
            augment = random.choice((True, False))
        else:
            wrinkles = args.wrinkles
            augment = args.augment

        if wrinkles:
            ifWrinkles = True
            ifCreases = True
            crease_angle = args.crease_angle if (args.deterministic_angle) else random.choice(
                range(0, args.crease_angle + 1))
            num_creases_vertically = args.num_creases_vertically if (args.deterministic_vertical) else random.choice(
                range(1, args.num_creases_vertically + 1))
            num_creases_horizontally = args.num_creases_horizontally if (
                args.deterministic_horizontal) else random.choice(range(1, args.num_creases_horizontally + 1))
            out = get_creased(out, output_directory=args.output_directory, ifWrinkles=ifWrinkles, ifCreases=ifCreases,
                              crease_angle=crease_angle, num_creases_vertically=num_creases_vertically,
                              num_creases_horizontally=num_creases_horizontally, bbox=args.lead_bbox)
        else:
            crease_angle = 0
            num_creases_horizontally = 0
            num_creases_vertically = 0

        if args.store_config == 2:
            json_dict['wrinkles'] = bool(wrinkles)
            json_dict['crease_angle'] = crease_angle
            json_dict['number_of_creases_horizontally'] = num_creases_horizontally
            json_dict['number_of_creases_vertically'] = num_creases_vertically

        if augment:
            noise = args.noise if args.deterministic_noise else random.choice(range(1, args.noise + 1))

            if not args.lead_bbox:
                do_crop = random.choice((True, False))
                if do_crop:
                    crop = args.crop
                else:
                    crop = args.crop
            else:
                crop = 0
            blue_temp = random.choice((True, False))

            if blue_temp:
                temp = random.choice(range(2000, 4000))
            else:
                temp = random.choice(range(10000, 20000))
            rotate = random.randint(-args.rotate, args.rotate)
            # note if args.store_config!=2 then json_dict is uninitialized
            out = get_augment(out, output_directory=args.output_directory, rotate=rotate, noise=noise, crop=crop,
                              temperature=temp, bbox=args.lead_bbox, store_text_bounding_box=args.lead_name_bbox,
                              json_dict=json_dict)

        else:
            crop = 0
            temp = 0
            rotate = 0
            noise = 0
        if args.store_config == 2:
            json_dict['augment'] = bool(augment)
            json_dict['crop'] = crop
            json_dict['temperature'] = temp
            json_dict['rotate'] = rotate
            json_dict['noise'] = noise

        if args.store_config:
            json_object = json.dumps(json_dict, indent=4)

            with open(rec_tail + '.json', "w") as f:
                f.write(json_object)

    return len(out_array)


# if __name__ == '__main__':
#     path = os.path.join(os.getcwd(), sys.argv[0])
#     parentPath = os.path.dirname(path)
#     os.chdir(parentPath)
#     run_single_file(get_parser().parse_args(sys.argv[1:]))
