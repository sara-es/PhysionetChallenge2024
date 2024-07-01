import os
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import AutoMinorLocator
from generator.TemplateFiles.generate_template import generate_template
from math import ceil
from PIL import Image

# Constants (moved to the top for easy access and modification)
STANDARD_VALUES = {
    'y_grid_size': 0.5,
    'x_grid_size': 0.2,
    'y_grid_inch': 5 / 25.4,
    'x_grid_inch': 5 / 25.4,
    'grid_line_width': 0.5,
    'lead_name_offset': 0.5,
    'lead_fontsize': 11,
    'x_gap': 1,
    'y_gap': 0.5,
    'display_factor': 1,
    'line_width': 0.75,
    'row_height': 8,
    'dc_offset_length': 0.2,
    'lead_length': 3,
    'V1_length': 12,
    'width': 11,
    'height': 8.5
}

STANDARD_MAJOR_COLORS = {
    'colour1': (0.4274, 0.196, 0.1843),  # brown
    'colour2': (1, 0.796, 0.866),  # pink
    'colour3': (0.0, 0.0, 0.4),  # blue
    'colour4': (0, 0.3, 0.0),  # green
    'colour5': (1, 0, 0)  # red
}

STANDARD_MINOR_COLORS = {
    'colour1': (0.5882, 0.4196, 0.3960),
    'colour2': (0.996, 0.9294, 0.9725),
    'colour3': (0.0, 0, 0.7),
    'colour4': (0, 0.8, 0.3),
    'colour5': (0.996, 0.8745, 0.8588)
}

PAPERSIZE_VALUES = {
    'A0': (33.1, 46.8),
    'A1': (33.1, 23.39),
    'A2': (16.54, 23.39),
    'A3': (11.69, 16.54),
    'A4': (8.27, 11.69),
    'letter': (8.5, 11)
}


def inches_to_dots(value, resolution):
    return value * resolution


def initialize_plot(width, height, resolution):
    fig, ax = plt.subplots(figsize=(width, height), dpi=resolution)
    fig.subplots_adjust(hspace=0, wspace=0, left=0, right=1, bottom=0, top=1)
    return fig, ax


def setup_grid(ax, x_min, x_max, y_min, y_max, x_grid_size, y_grid_size, grid_line_width, color_major, color_minor,
               show_grid):
    if show_grid:
        ax.set_xticks(np.arange(x_min, x_max, x_grid_size))
        ax.set_yticks(np.arange(y_min, y_max, y_grid_size))
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(which='major', linestyle='-', linewidth=grid_line_width, color=color_major)
        ax.grid(which='minor', linestyle='-', linewidth=grid_line_width, color=color_minor)
        ax.text(2, 0.5, '25mm/s', fontsize=STANDARD_VALUES['lead_fontsize'])
        ax.text(4, 0.5, '10mm/mV', fontsize=STANDARD_VALUES['lead_fontsize'])
    else:
        ax.grid(False)


def get_colors(style, standard_colours):
    if style == 'bw':
        color_major = color_minor = (0.4, 0.4, 0.4)
        color_line = (0, 0, 0)
    elif standard_colours > 0:
        color_major = STANDARD_MAJOR_COLORS[f'colour{standard_colours}']
        color_minor = STANDARD_MINOR_COLORS[f'colour{standard_colours}']
        color_line = (random.uniform(0, 0.2),) * 3
    else:
        color_major = tuple(random.uniform(0, 0.8) for _ in range(3))
        color_minor = tuple(min(1, x + random.uniform(0, 0.2)) for x in color_major)
        color_line = (random.uniform(0, 0.2),) * 3
    return color_major, color_minor, color_line


def plot_lead(ax, ecg, leadName, x_offset, y_offset, dc_offset, x_gap, step, line_width, color_line, show_dc_pulse):
    if show_dc_pulse:
        dc_pulse_x = np.arange(0, STANDARD_VALUES['dc_offset_length'] * step + 4 * step, step)
        dc_pulse_y = np.concatenate(((0, 0), np.ones(len(dc_pulse_x) - 4), (0, 0)))
        ax.plot(dc_pulse_x + x_offset + x_gap, dc_pulse_y + y_offset, linewidth=line_width * 1.5, color=color_line)

    lead_x = np.arange(0, len(ecg[leadName]) * step, step) + x_offset + dc_offset + x_gap
    lead_y = ecg[leadName] + y_offset
    return ax.plot(lead_x, lead_y, linewidth=line_width, color=color_line)


def plot_full_lead(ax, ecg, full_mode, x_gap, dc_offset, step, line_width, color_line, row_height, lead_name_offset):
    lead_x = np.arange(0, len(ecg['full' + full_mode]) * step, step) + x_gap + dc_offset
    lead_y = ecg['full' + full_mode] + row_height / 2 - lead_name_offset + 0.8
    return ax.plot(lead_x, lead_y, linewidth=line_width, color=color_line)


def calculate_layout(lead_index, format_config, rhythm_leads):
    """
    Calculate the layout based on the format configuration and rhythm leads.

    :param lead_index: List of lead names to be plotted
    :param format_config: Configuration for the desired format
    :param rhythm_leads: List of rhythm lead names
    :return: Tuple of (rows, columns, lead_positions, rhythm_positions)
    """
    rows = len(format_config)
    columns = max(len(row) for row in format_config)
    lead_positions = {}

    for i, row in enumerate(format_config):
        for j, lead in enumerate(row):
            if lead in lead_index:
                lead_positions[lead] = (i, j)

    rhythm_positions = {}
    rhythm_row = rows
    for i, lead in enumerate(rhythm_leads):
        rhythm_positions[lead] = (rhythm_row, 0, columns)  # (row, start_column, span)
        rhythm_row += 1

    rows += len(rhythm_leads)

    return rows, columns, lead_positions, rhythm_positions


def plot_lead_in_position(ax, ecg, leadName, row, col, rows, columns, x_gap, y_gap, secs, step, line_width, color_line,
                          show_dc_pulse, is_rhythm=False, column_span=1):
    """
    Plot a lead in a specific position on the grid.

    :param ax: Matplotlib axis object
    :param ecg: ECG data dictionary
    :param leadName: Name of the lead to plot
    :param row: Row position
    :param col: Column position
    :param rows: Total number of rows
    :param columns: Total number of columns
    :param x_gap: Gap in x-axis
    :param y_gap: Gap in y-axis
    :param secs: Seconds of ECG to plot
    :param step: Time step between samples
    :param line_width: Width of the ECG line
    :param color_line: Color of the ECG line
    :param show_dc_pulse: Whether to show DC pulse
    :param is_rhythm: Whether this is a rhythm lead
    :param column_span: Number of columns this lead spans (for rhythm leads)
    :return: Plot object
    """
    y_offset = (rows - row - 1) * (1 / rows) + y_gap
    x_offset = col * (secs / columns) + x_gap

    if show_dc_pulse and not is_rhythm:
        dc_pulse_x = np.arange(0, STANDARD_VALUES['dc_offset_length'] * step + 4 * step, step)
        dc_pulse_y = np.concatenate(((0, 0), np.ones(len(dc_pulse_x) - 4), (0, 0)))
        ax.plot(dc_pulse_x + x_offset, dc_pulse_y + y_offset, linewidth=line_width * 1.5, color=color_line)

    lead_data = ecg[leadName]
    if is_rhythm:
        lead_x = np.arange(0, len(lead_data) * step, step) + x_offset
        lead_y = lead_data / (rows * 2) + y_offset
        return ax.plot(lead_x, lead_y, linewidth=line_width, color=color_line)
    else:
        lead_x = np.arange(0, len(lead_data) * step, step) + x_offset
        lead_y = lead_data / (rows * 2) + y_offset
        return ax.plot(lead_x, lead_y, linewidth=line_width, color=color_line)


def add_lead_name(ax, leadName, x_offset, y_offset, lead_name_offset, lead_fontsize):
    return ax.text(x_offset, y_offset - lead_name_offset - 0.2, leadName, fontsize=lead_fontsize)


def save_output(fig, output_path, resolution, pad_inches, single_channel):
    plt.savefig(output_path, dpi=resolution)
    plt.close(fig)

    if pad_inches != 0:
        ecg_image = Image.open(output_path)
        padding = int(pad_inches * resolution)
        new_size = (ecg_image.width + 2 * padding, ecg_image.height + 2 * padding)
        padded_image = Image.new(ecg_image.mode, new_size, (255, 255, 255))
        padded_image.paste(ecg_image, (padding, padding))

        if single_channel:
            img_arr = np.array(padded_image)
            result_image = ~img_arr[:, :, 0].astype(bool)
            result_image[0, :] = result_image[-1, :] = result_image[:, 0] = result_image[:, -1] = False
            np.save(output_path.replace('.png', '.npy'), result_image)
        else:
            padded_image.save(output_path)


def ecg_plot(ecg, configs, sample_rate, rec_file_name, output_dir, resolution, pad_inches, lead_index,
             rhythm_leads, store_text_bbox, full_header_file, units='', papersize='', x_gap=STANDARD_VALUES['x_gap'],
             y_gap=STANDARD_VALUES['y_gap'], display_factor=STANDARD_VALUES['display_factor'],
             line_width=STANDARD_VALUES['line_width'], title='', style=None, show_lead_name=True,
             show_grid=False, show_dc_pulse=False, standard_colours=False, bbox=False, print_txt=False,
             json_dict=dict(), start_index=-1, store_configs=0, lead_length_in_seconds=10, rhythm_length_in_seconds=10):
    if not ecg:
        return

    # Get format configuration
    format_config = configs.get('format_config', configs['format_4_by_3'])

    # Calculate layout
    rows, columns, lead_positions, rhythm_positions = calculate_layout(lead_index, format_config, rhythm_leads)

    # Setup basic parameters
    secs = lead_length_in_seconds
    rhythm_secs = rhythm_length_in_seconds
    step = 1.0 / sample_rate

    # Setup plot size
    width, height = PAPERSIZE_VALUES.get(papersize, (STANDARD_VALUES['width'], STANDARD_VALUES['height']))

    # Initialize plot
    fig, ax = initialize_plot(width, height, resolution)
    fig.suptitle(title)

    # Setup colors
    color_major, color_minor, color_line = get_colors(style, standard_colours)

    # Setup grid
    setup_grid(ax, 0, width, 0, height, STANDARD_VALUES['x_grid_size'], STANDARD_VALUES['y_grid_size'],
               STANDARD_VALUES['grid_line_width'], color_major, color_minor, show_grid)

    # Plot leads
    leads_ds = []
    for leadName in lead_index:
        if leadName in lead_positions:
            row, col = lead_positions[leadName]
            plot = plot_lead_in_position(ax, ecg, leadName, row, col, rows, columns, x_gap, y_gap,
                                         secs, step, line_width, color_line, show_dc_pulse)

            current_lead_ds = {"lead_name": leadName}

            if show_lead_name:
                text = add_lead_name(ax, leadName, col * (secs / columns) + x_gap,
                                     (rows - row - 1) * (1 / rows) + y_gap,
                                     STANDARD_VALUES['lead_name_offset'],
                                     STANDARD_VALUES['lead_fontsize'])
                if store_text_bbox:
                    current_lead_ds["text_bounding_box"] = get_bounding_box(text, fig, height)

            if bbox:
                current_lead_ds["lead_bounding_box"] = get_bounding_box(plot[0], fig, height)

            current_lead_ds["start_sample"] = start_index
            current_lead_ds["end_sample"] = start_index + len(ecg[leadName])
            current_lead_ds["plotted_pixels"] = get_plotted_pixels(plot[0], ax, height)

            leads_ds.append(current_lead_ds)

    # Plot rhythm leads
    for leadName in rhythm_leads:
        if leadName in rhythm_positions:
            row, col, span = rhythm_positions[leadName]
            plot = plot_lead_in_position(ax, ecg, leadName, row, col, rows, columns, x_gap, y_gap,
                                         rhythm_secs, step, line_width, color_line, False, True, span)

            current_lead_ds = {"lead_name": leadName, "is_rhythm": True}

            if show_lead_name:
                text = add_lead_name(ax, leadName, x_gap,
                                     (rows - row - 1) * (1 / rows) + y_gap,
                                     STANDARD_VALUES['lead_name_offset'],
                                     STANDARD_VALUES['lead_fontsize'])
                if store_text_bbox:
                    current_lead_ds["text_bounding_box"] = get_bounding_box(text, fig, height)

            if bbox:
                current_lead_ds["lead_bounding_box"] = get_bounding_box(plot[0], fig, height)

            current_lead_ds["start_sample"] = start_index
            current_lead_ds["end_sample"] = start_index + len(ecg[leadName])
            current_lead_ds["plotted_pixels"] = get_plotted_pixels(plot[0], ax, height)

            leads_ds.append(current_lead_ds)

    # Save output
    output_path = os.path.join(output_dir, os.path.basename(rec_file_name) + '.png')
    save_output(fig, output_path, resolution, pad_inches,
                False)  # Changed single_channel to False as it's not applicable for multi-lead plots

    json_dict["leads"] = leads_ds

    return STANDARD_VALUES['x_grid_inch'] * resolution, STANDARD_VALUES['y_grid_inch'] * resolution


def get_bounding_box(artist, fig, height):
    bb = artist.get_window_extent(renderer=fig.canvas.get_renderer())
    bb_data = bb.transformed(fig.transFigure.inverted())
    x1, y1, x2, y2 = bb_data.x0, bb_data.y0, bb_data.x1, bb_data.y1
    return {
        0: [round(height - y2, 2), round(x1, 2)],
        1: [round(height - y2, 2), round(x2, 2)],
        2: [round(height - y1, 2), round(x2, 2)],
        3: [round(height - y1, 2), round(x1, 2)]
    }


def get_plotted_pixels(artist, ax, height):
    xy_pixels = ax.transData.transform(artist.get_xydata())
    return [[round(height - y, 2), round(x, 2)] for x, y in xy_pixels]


# Main execution
if __name__ == "__main__":
    # Add your main execution code here
    pass
