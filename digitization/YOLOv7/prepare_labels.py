import os
import json

def prepare_label_text(metadata, lead_text_label=0, short_lead_label=1, long_lead_label=2):
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


def prepare_label_files(ids, json_file_dir, label_file_dir):
    """
    Prepare label files for the images in the dataset.
    """
    os.makedirs(label_file_dir, exist_ok=True)

    ids = [f.split(os.sep)[-1] for f in ids] # Make sure IDs are strings and not paths
    ids = [f[:8] for f in ids] # take the first 8 characters
    all_files = os.listdir(json_file_dir)
    json_filenames = [f for f in all_files if f.endswith('.json')]
    matching_json_filenames = [f for f in json_filenames if f[:8] in ids]
    if len(matching_json_filenames) != len(ids):
        raise FileNotFoundError(f"Some requested json files are missing from {json_file_dir}. "+\
                                "Please check that you have generated the json files for the images.")

    for json_file in matching_json_filenames:
        json_path = os.path.join(json_file_dir, json_file)
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        label_text = prepare_label_text(metadata)
        with open(os.path.join(label_file_dir, json_file.replace(".json", ".txt")), 'w') as f:
            f.write(label_text)


if __name__ == "__main__":
    ids = set([f.split(".")[0] for f in os.listdir("test_data\\train_images\\images") if f.endswith(".png")]) 
    json_file_dir = "test_data\\train_images"
    label_file_dir = "test_data\\train_images\\labels"
    prepare_label_files(ids, json_file_dir, label_file_dir)