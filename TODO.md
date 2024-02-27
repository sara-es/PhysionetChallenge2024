## Classification

### TODO
- there are duplicate training loops for seresnet18 atm, figure this out

#### After it works
- now we make a list of file paths to pass into resnet, there is probably a better option?
- right now, demographics for classification are sex and age, is it worth adding height and weight?
- metric functions/validation
- dynamically determine number of ECG leads
- refactor train test split in team_code.py

#### Low priority
- investigate 'Bottleneck' in seresnet18.py
- turn `list[list]` into np.array in ECGDataset

### Changelog/comments
#### SS Update 27 Feb:
- renamed `seresnet18.py` to `SEResNet.py` - this will only contain the resnet class definitions
- renamed `train_utils.py` to `seresnet18.py` - this will contain the function call, train and test functions
- moved `preprocess_labels` and `compute_classification_metrics` from `metrics.py` to `team_helper_code.py`. Deleted `metrics.py`.
- added verbose flags throughout `seresnet18` training loop
- added `epochs` arg to `train` function in `seresnet18.py`

- changed `n_splits` in `seresnet18.train()` to 1 by default (Is there any benefit to keeping k-fold CV here?)
- resnet training loop now returns trained model state dict regardless of performing validation or not
- fixed circular imports in `seresnet18.py` 
- added resnet to list of default Dx models

#### TL Update 26th Feb:
- check why the ready-made function to load signal loads the signal in shape (5000, 12) instaed of (12, 5000)?! 
- finish args dict for initialisation
- check: what if ECGdatast gets only records as load_signal() function loads the signal AND the corresponding header?!

## General
- cross-validation script with examples (SS todo)