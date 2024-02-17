# For collaborators: information on file structure and code
Physionet Challenges require that we keep certain scripts in the root path of the repository. 
 - `team_code.py` is a script with functions for training and running our trained model(s).
Please do not edit the following scripts. The Physionet team will use the unedited versions of these scripts when running the code:
- `train_model.py` is a script for training the model(s).
- `run_model.py` is a script for running the trained model(s).
- `helper_code.py` is a script with example helper functions. 
These scripts must remain in the root path of the repository, and will be overwritten by the Physionet team when running the code.

The structure of subfolders in the repository is as follows:
```
├───classification        # code for the classification task
├───evaluation            # scripts to evaluate the performance of the models, including 5-fold cross-validation
├───models                # saved models, e.g. .sav and .pth
├───preprocessing         # code for preprocessing the data and images
├───reconstruction        # code for the reconstruction task
├───tests                 # unittests, etc.
├───tiny_testset          # 15 samples for fast testing
│   ├───records100           # 100hz data and images
│   ├───records100_hidden    # 100hz images with data occluded, for inference testing
│   ├───records500           # 500hz data and images
│   ├───records500_hidden    # 500hz images with data occluded, for inference testing
│   └───test_outputs         # send model outputs here
└───utils                 # constants, helper functions, etc.
```

## VERY IMPORTANT: READ BEFORE PUSHING CODE
Please add your model to `utils/default_models.py` to the list of models for either digitization (reconstruction) or dx (classification). This makes sure it's called in the respective train, load, and test sections in `team_code.py`. Then, make sure both `train_model.py` and `test_model.py` run, without errors, before submitting a pull request. If you wish to test your model outside of `team_code.py`, please do so in a separate file in the `evaluation` or `tests` directory.
#### Please make sure that `train_model.py` and `test_model.py` run without errors before submitting a pull request.