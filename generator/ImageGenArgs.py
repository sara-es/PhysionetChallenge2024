import os, sys
os.path.join(sys.path[0], '..')

class DefaultArgs():
    def __init__(self) -> None:
        self.seed = -1
        self.num_leads = 'twelve'
        self.max_num_images = -1
        self.config_file = 'config.yaml'
        self.resolution = 200
        self.pad_inches = 0
        self.print_header = False
        self.num_columns = -1
        self.full_mode = 'II'
        self.mask_unplotted_samples = False
        self.add_qr_code = False
        self.link = ''
        self.num_words = 5
        self.x_offset = 30
        self.y_offset = 30
        self.handwriting_size_factor = 0.2
        self.crease_angle = 90
        self.num_creases_vertically = 10
        self.num_creases_horizontally = 10
        self.rotate = 0
        self.noise = 50
        self.crop = 0.01
        self.temperature = 40000
        self.random_resolution = False
        self.random_padding = False
        self.random_grid_color = False
        self.standard_grid_color = 5
        self.calibration_pulse = 1
        self.random_grid_present = 1
        self.random_print_header = 0
        self.random_bw = 0
        self.add_lead_names = True
        self.lead_name_bbox = False
        self.store_config = 0
        self.deterministic_offset = False
        self.deterministic_num_words = False
        self.deterministic_hw_size = False
        self.deterministic_angle = False
        self.deterministic_vertical = False
        self.deterministic_horizontal = False
        self.deterministic_rot = False
        self.deterministic_noise = False
        self.deterministic_crop = False
        self.deterministic_temp = False
        self.fully_random = False
        self.hw_text = False
        self.wrinkles = False
        self.augment = False
        self.lead_bbox = False
        self.single_channel = False
        self.copy_data_files = True
        self.input_directory = ''
        self.output_directory = ''
        self.input_file = ''
        self.header_file = ''
        self.start_index = -1
        self.encoding = ''

class MaskArgs():
    def __init__(self) -> None:
        self.seed = -1
        self.num_leads = 'twelve'
        self.max_num_images = -1
        self.config_file = 'config.yaml'
        self.resolution = 200
        self.pad_inches = 0
        self.print_header = False
        self.num_columns = -1
        self.full_mode = 'II'
        self.mask_unplotted_samples = False
        self.add_qr_code = False
        self.link = ''
        self.num_words = 5
        self.x_offset = 30
        self.y_offset = 30
        self.handwriting_size_factor = 0.2
        self.crease_angle = 90
        self.num_creases_vertically = 10
        self.num_creases_horizontally = 10
        self.rotate = 0
        self.noise = 50
        self.crop = 0.01
        self.temperature = 40000
        self.random_resolution = False
        self.random_padding = False
        self.random_grid_color = False
        self.standard_grid_color = 5
        self.calibration_pulse = 1
        self.random_grid_present = 0
        self.random_print_header = 0
        self.random_bw = 1.0
        self.add_lead_names = False
        self.lead_name_bbox = False
        self.store_config = 0
        self.deterministic_offset = False
        self.deterministic_num_words = False
        self.deterministic_hw_size = False
        self.deterministic_angle = False
        self.deterministic_vertical = False
        self.deterministic_horizontal = False
        self.deterministic_rot = False
        self.deterministic_noise = False
        self.deterministic_crop = False
        self.deterministic_temp = False
        self.fully_random = False
        self.hw_text = False
        self.wrinkles = False
        self.augment = False
        self.lead_bbox = False
        self.single_channel = True
        self.copy_data_files = False
        self.input_directory = ''
        self.output_directory = ''
        self.input_file = ''
        self.header_file = ''
        self.start_index = -1
        self.encoding = ''