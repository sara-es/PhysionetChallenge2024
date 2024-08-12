import os, sys
sys.path.append(os.path.join(sys.path[0], '..'))
import json
from utils import team_helper_code

def prepare_label_text(metadata, lead_text_label=None, short_lead_label=0, long_lead_label=1):
    """
    Get the bounding box from the json file and convert it to xywh format.
    xywh is <x_center, y_center, width, height>
    
    Each image should have a corresponding label text file, where each line of the file is 
    class label, x_center, y_center, width, height. The class label is an integer in the 
    range [0, n_classes-1].
    """
    out_str = ""

    image_width = metadata["width"]
    image_height = metadata["height"]
    sampling_frequency = metadata["sampling_frequency"]

    for lead in metadata["leads"]:
        # lead names
        if lead_text_label is not None:
            x_min = lead["text_bounding_box"]["0"][1] / image_width
            x_max = lead["text_bounding_box"]["2"][1] / image_width
            y_min = lead["text_bounding_box"]["0"][0] / image_height
            y_max = lead["text_bounding_box"]["2"][0] / image_height

            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min

            out_str += f"{lead_text_label} {x_center} {y_center} {width} {height}\n"

        # leads
        x_min = lead["lead_bounding_box"]["0"][1] / image_width
        x_max = lead["lead_bounding_box"]["2"][1] / image_width
        y_min = lead["lead_bounding_box"]["0"][0] / image_height
        y_max = lead["lead_bounding_box"]["2"][0] / image_height

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min


        if lead["end_sample"] - lead["start_sample"] == sampling_frequency*10:
            out_str += f"{long_lead_label} {x_center} {y_center} {width} {height}\n"
        else:
            out_str += f"{short_lead_label} {x_center} {y_center} {width} {height}\n"

    return out_str


def prepare_label_files(ids, json_file_dir, label_file_dir, verbose):
    """
    Prepare label files for the images in the dataset.

    params:
    ids: list of image ids with respect to the root folder. e.g. if folder is "ptb-xl/records100"
        then records is a list of strings ["00000/00157_lr", "01000/01128_lr", ...].
    json_file_dir: path to the directory containing the json files created at generation
    label_file_dir: path to the directory where the label files will be saved
    """
    os.makedirs(label_file_dir, exist_ok=True)
    # delete existing labels if present to avoid confusion
    for file in os.listdir(label_file_dir):
        os.remove(os.path.join(label_file_dir, file))

    json_ids = team_helper_code.find_files(json_file_dir, extension_str='.json')
    id_tails = [f.split(os.sep)[-1] for f in ids]
    label_ids_to_save = set(json_ids).union(set(id_tails))
    if set(json_ids) != set(id_tails) and verbose:
        print(f"Found {len(json_ids)} json files, for {len(id_tails)} requested images.")
        print(f"Missing json files: {set(id_tails) - set(json_ids)}")
        print(f"Missing image files: {set(json_ids) - set(id_tails)}")
        print(f"Some requested json files are missing from {json_file_dir}. "+\
              f"Generating labels for {len(label_ids_to_save)} images.")

    for i, json_file in enumerate(label_ids_to_save):
        json_path = os.path.join(json_file_dir, json_file + ".json")
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        label_text = prepare_label_text(metadata)
        with open(os.path.join(label_file_dir, json_file + ".txt"), 'w') as f:
            f.write(label_text)



# if __name__ == "__main__":
#     ids = set([f.split(".")[0] for f in os.listdir("yolo_data\\val_images\\images") if f.endswith(".png")]) 
#     json_file_dir = "yolo_data\\val_images"
#     label_file_dir = "yolo_data\\val_images\\labels"
#     prepare_label_files(ids, json_file_dir, label_file_dir)