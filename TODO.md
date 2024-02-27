## Classification

### TODO
- add verbose flag to seresnet training loop
- there are duplicate training loops for seresnet18 atm, figure this out

#### Update 27 Feb:
- renamed `seresnet18.py` to `SEResNet.py` - this will only contain the resnet class definitions
- renamed `train_utils.py` to `seresnet18.py` - this will contain the function call, train and test functions
- moved `preprocess_labels` and `compute_classification_metrics` from `metrics.py` to `team_helper_code.py`. Deleted `metrics.py`.

#### Update 26th Feb:
- check why the ready-made function to load signal loads the signal in shape (5000, 12) instaed of (12, 5000)?!
- finish args dict for initialisation
- check: what if ECGdatast gets only records as load_signal() function loads the signal AND the corresponding header?!

#### After it works
- right now we make a list of file paths to pass into resnet, probably a better option?
- right now, demographics for classifcation are sex and age, is it worth adding height and weight?
- metric functions/validation
- dynamically determine number of ECG leads
- refactor train test split in team_code.py

#### Low priority
- investigate 'Bottleneck' in seresnet18.py