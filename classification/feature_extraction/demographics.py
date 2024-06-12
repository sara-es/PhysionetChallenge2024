import numpy as np
import helper_code

def extract_features(record):
    # TODO:
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