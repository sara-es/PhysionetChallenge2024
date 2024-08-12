import os, sys
sys.path.append(os.path.join(sys.path[0], '..'))
import team_code, helper_code

# Run the code.
def run(data_folder, output_folder, model_folder, verbose, allow_failures):
    # Load the models.
    if verbose:
        print('Loading the Challenge model...')

    # You can use these functions to perform tasks, such as loading your models, that you only need to perform once.
    digitization_model, classification_model = team_code.load_models(model_folder, verbose) ### Teams: Implement this function!!!

    # Find the Challenge data.
    if verbose:
        print('Finding the Challenge data...')

    records = helper_code.find_records(data_folder)
    num_records = len(records)

    if num_records==0:
        raise Exception('No data were provided.')

    # Create a folder for the Challenge outputs if it does not already exist.
    os.makedirs(output_folder, exist_ok=True)

    # Run the team's models on the Challenge data.
    if verbose:
        print('Running the Challenge model(s) on the Challenge data...')

    # Iterate over the records.
    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        data_record = os.path.join(data_folder, records[i])
        output_record = os.path.join(output_folder, records[i])

        # Run the models. Allow or disallow the models to fail on some of the data, which can be helpful for debugging.
        try:
            signals, labels = team_code.run_models(data_record, digitization_model, classification_model, verbose) ### Teams: Implement this function!!!
        except:
            if allow_failures:
                if verbose:
                    print('... failed.')
                signals = None
                labels = None
            else:
                raise

        # Save Challenge outputs.
        output_path = os.path.split(output_record)[0]
        os.makedirs(output_path, exist_ok=True)

        data_header = helper_code.load_header(data_record)
        helper_code.save_header(output_record, data_header)

        if signals is not None:
            comments = [l for l in data_header.split('\n') if l.startswith('#')]
            helper_code.save_signals(output_record, signals, comments)
        if labels is not None:
            helper_code.save_labels(output_record, labels)

    if verbose:
        print('Done.')

if __name__ == "__main__":
    data_folder = os.path.join('tiny_testset', 'hr_hidden')
    output_folder = os.path.join('tiny_testset', 'test_outputs')
    model_folder = os.path.join('model')
    verbose = True
    allow_failures = False
    run(data_folder, output_folder, model_folder, verbose, allow_failures)