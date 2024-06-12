import os
import subprocess
from itertools import islice
from multiprocessing import Process


# either run this in the ecg-image-kit/ecg-image-generator/codes directory or change the path to the script

def run_script(input_path: str,
               output_path: str,
               seed: int,
               grid: bool,
               ref_pulse: bool,
               wrinkles: bool = False,
               header: bool = False) -> None:
    command = ['python', 'gen_ecg_images_from_data_batch.py',
               '-i', input_path,
               '-o', output_path,
               '-se', str(seed),
               '--print_header' if header else '',
               '--random_grid_present', '1' if grid else '0',
               '--store_config']

    if ref_pulse:  # add ref pulse
        command.extend(['--random_dc', '1'])

    if wrinkles:  # add shadow
        command.append('--wrinkles')

    subprocess.run(command)
    print(f'Completed: {output_path}')


def take(n: int, iterable: iter):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


def pair_files(directory: str) -> dict[str, tuple[str, str]]:
    # Initialize dictionaries to hold the files
    dat_files = {}
    hea_files = {}

    # Populate dictionaries with filenames, keying by the numeric prefix
    for filename in os.listdir(directory):
        # Split the filename into name and extension
        prefix, extension = os.path.splitext(filename)
        if extension == '.dat':
            dat_files[prefix] = os.path.join(directory, filename)
        elif extension == '.hea':
            hea_files[prefix] = os.path.join(directory, filename)

    # Match dat_files with hea_files
    paired_files = {}
    for prefix in dat_files:
        if prefix in hea_files:
            paired_files[prefix] = (dat_files[prefix], hea_files[prefix])

    return paired_files


if __name__ == '__main__':
    RANDOM_SEED = 42
    CLEAN_PATH = f'images/clean/'
    CLEAN_GRID_PATH = f'images/clean_grid/'
    CLEAN_PATH_REF_PULSE = f'images/clean_ref_pulse/'
    CLEAN_GRID_PATH_REF_PULSE = f'images/clean_grid_ref_pulse/'

    SHADOW_PATH = f'images/shadow/'
    SHADOW_GRID_REF_PULSE = f'images/shadow_grid_ref_pulse/'
    SHADOW_GRID_PATH = f'images/shadow_grid/'
    SHADOW_PATH_REF_PULSE = f'images/shadow_ref_pulse/'

    os.makedirs(CLEAN_PATH, exist_ok=True)
    os.makedirs(CLEAN_GRID_PATH, exist_ok=True)
    os.makedirs(CLEAN_PATH_REF_PULSE, exist_ok=True)
    os.makedirs(CLEAN_GRID_PATH_REF_PULSE, exist_ok=True)
    os.makedirs(SHADOW_PATH, exist_ok=True)
    os.makedirs(SHADOW_GRID_PATH, exist_ok=True)
    os.makedirs(SHADOW_PATH_REF_PULSE, exist_ok=True)
    os.makedirs(SHADOW_GRID_REF_PULSE, exist_ok=True)

    # for folder in tqdm(os.listdir('data/records100/')[0]):
    folder = os.listdir('data/')[0]

    print(f'directory list: {os.listdir("data/records100/")[0]}')
    print(f'Generating images from folder records100 {folder}')
    INPUT_PATH = f'data/records100/{folder}'

    RANDOM_SEED = 42
    # Create the output folders. Some of these are superfluous. I don't care.
    # no headers for now :shrug:
    processes = [
        Process(target=run_script, args=(INPUT_PATH, CLEAN_PATH, RANDOM_SEED, False, False, False)),
        Process(target=run_script, args=(INPUT_PATH, CLEAN_PATH_REF_PULSE, RANDOM_SEED, False, True, False)),
        Process(target=run_script, args=(INPUT_PATH, CLEAN_GRID_PATH, RANDOM_SEED, True, False, False)),
        Process(target=run_script, args=(INPUT_PATH, CLEAN_GRID_PATH_REF_PULSE, RANDOM_SEED, True, True, False)),
        Process(target=run_script, args=(INPUT_PATH, SHADOW_PATH, RANDOM_SEED, False, False, True)),
        Process(target=run_script, args=(INPUT_PATH, SHADOW_PATH_REF_PULSE, RANDOM_SEED, False, True, True)),
        Process(target=run_script, args=(INPUT_PATH, SHADOW_GRID_PATH, RANDOM_SEED, True, False, True)),
        Process(target=run_script, args=(INPUT_PATH, SHADOW_GRID_REF_PULSE, RANDOM_SEED, True, True, True))
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join()
    print('Done')
