import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))

import json

json_file = "test_data\\images\\00045_hr-0.json"

with open(json_file, 'r') as f:
    metadata = json.load(f)

print(metadata["leads"][0])

out_str = ""
lead_text_label = 0
short_lead_label = 1
long_lead_label = 2

sampling_frequency = metadata["sampling_frequency"]

for lead in metadata["leads"]:
    # lead names
    x_min = lead["text_bounding_box"]["0"][0]
    x_max = lead["text_bounding_box"]["2"][0]
    y_min = lead["text_bounding_box"]["0"][1]
    y_max = lead["text_bounding_box"]["2"][1]

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    out_str += f"{lead_text_label} {x_center} {y_center} {width} {height}\n"

    # leads
    x_min = lead["lead_bounding_box"]["0"][0]
    x_max = lead["lead_bounding_box"]["2"][0]
    y_min = lead["lead_bounding_box"]["0"][1]
    y_max = lead["lead_bounding_box"]["2"][1]

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min


    if lead["end_sample"] - lead["start_sample"] == sampling_frequency*10:
        out_str += f"{long_lead_label} {x_center} {y_center} {width} {height}\n"
    else:
        out_str += f"{short_lead_label} {x_center} {y_center} {width} {height}\n"

print(out_str)