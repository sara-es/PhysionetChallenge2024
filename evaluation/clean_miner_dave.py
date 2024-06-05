from clean_miner_visualization import *
import os
os.chdir('C:/Users/hssdwo/Documents/GitHub/PhysionetChallenge2024')
#import helper_code

data_folder = 'C:\\Users\\hssdwo\\Documents\\GitHub\\PhysionetChallenge2024\\tiny_testset\\hr_gt'
output_folder = 'C:\\Users\\hssdwo\\Documents\\GitHub\\PhysionetChallenge2024\\tiny_testset\\test_outputs'
verbose = True

# Find data files.
records = helper_code.find_records(data_folder)
if len(records) == 0:
    raise FileNotFoundError('No data was provided.')

# Create a folder for the images if it does not already exist.
os.makedirs(output_folder, exist_ok=True)

mean_snrs = np.zeros(len(records))

for i in tqdm(range(len(records))):  
    record = os.path.join(data_folder, records[i])
    record_name = records[i]

    # get ground truth signal and metadata
    header_file = helper_code.get_header_file(record)
    header = helper_code.load_text(header_file)
    num_samples = helper_code.get_num_samples(header)

    # get filenames/paths of all records to be reconstructed
    image_files = team_helper_code.load_image_paths(record)
    image_file = image_files[0]
    if len(image_files) > 1:
        if verbose:
            print(f"Multiple images found, using image at {image_file}.")
    
    # run the digitization model     
    trace, signal, gridsize = run_digitization_model(image_file, num_samples, verbose=True)

    # get digitization output
    signal = np.asarray(signal*1000, dtype=np.int16)
    
    # run_model.py just rewrites header file to output folder here, so we can skip that step
    # uncomment following line if we want to follow challenge save/load protocols exactly
    # output_signal, output_fields = save_and_load_wfdb(header, signal, output_folder=output_folder, record=record_name)        
    output_signal, output_fields = format_wfdb_signal(header, signal) # output_record is the filepath the output signal will be saved to
    
    # get ground truth signal
    label_signal, label_fields = helper_code.load_signal(record)

    # match signal lengths: make sure channel orders match and trim output signal to match label signal length
    output_signal, output_fields, label_signal, label_fields = match_signal_lengths(output_signal, output_fields, label_signal, label_fields)

    # compute SNR vs ground truth
    mean_snr, mean_snr_median, mean_ks_metric, mean_asci_metric, mean_weighted_absolute_difference_metric = single_signal_snr(output_signal, output_fields, label_signal, label_fields, record_name, extra_scores=True)
    
    # add metrics to dataframe to save later
    mean_snrs[i] = mean_snr
    #TODO

    # plot signal reconstruction
    plot_signal_reconstruction(label_signal, output_signal, output_fields, mean_snr, trace, image_file, output_folder, record_name=record_name, gridsize=gridsize)

print(f"Finished. Overall mean SNR: {np.nanmean(mean_snrs):.2f} over {len(records)} records.")
