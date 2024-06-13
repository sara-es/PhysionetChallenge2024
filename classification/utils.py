import os
import numpy as np
import helper_code

def get_demographic_features(record):
    # TODO: add missing flags
    age_gender = np.zeros(3)
    header = helper_code.load_header(record)
    
    age, has_age = helper_code.get_variable(header, 'Age')
    height, has_height = helper_code.get_variable(header, 'Height')
    weight, has_weight = helper_code.get_variable(header, 'Weight')
    sex, has_sex = helper_code.get_variable(header, 'Sex')

    if has_age:
        age_gender[0] = int(age)/100.
    if has_sex:
        age_gender[1] = int(sex == 'Female')
        age_gender[2] = int(sex == 'Male')
    return age_gender

def get_training_data(record, data_folder):
    # get headers for labels and demographic info
    record_path = os.path.join(data_folder, record) 
    header_txt = helper_code.load_header(record_path)
    labels = helper_code.load_labels(record_path)
    if labels: # only process records with labels for training
        # get demographic info
        # TODO: add missing flags
        age_gender = get_demographic_features(record_path)
        fs = helper_code.get_sampling_frequency(header_txt)
    else:
        return None, None
    
    data = [record_path, fs, age_gender]

    return data, labels