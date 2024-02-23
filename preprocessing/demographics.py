import numpy as np
import helper_code

def extract_features(record):
    age_gender = np.zeros(3)
    header = helper_code.load_header(record)
    age_gender[0] = int(helper_code.get_variable(header, '#Age:')[0])/100.
    age_gender[1] = int(helper_code.get_variable(header, '#Sex:')[0] == 'Female')
    age_gender[2] = int(helper_code.get_variable(header, '#Sex:')[0] == 'Male')
    return age_gender