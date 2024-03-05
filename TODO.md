## Classification

### TODO
- test on larger dataset (on tiny test we are returning all 'Norm')
- unittests for dataloaders 
- 5-fold CV: Evaluation
- single preprocessing function for both train and test
- looks like frequency (fs) is passed to dataloaders but not used, do we need this?
- currently, we make a list of file paths to pass into resnet, there is probably a better option?
- currently, demographics for classification are sex and age, is it worth adding height and weight?
- ~~refactor train test split in team_code.py~~ done SS

#### Low priority
- investigate 'Bottleneck' in seresnet18.py
- turn `list[list]` into np.array in ECGDataset
- allow for multiple GPUs if multiple devices are available
- worth dynamically determining number of ECG leads?

### Changelog/comments
#### SS Update 01 Mar:
- Big refactor of `seresnet18.py`: removed Training and Predicting classes, now has only train and predict functions
- Removed all references to multiple devices (as we were hard coding this to 1 anyway)
- Fixed `team_helper_code.computer_classification_metrics()`: it was passing one-hot encoded labels, but the helper code version of compute f-measure accepts string labels. Instead, we can use the new `team_helper_code.compute_f_measure_from_onehot()` function.

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
- added resnet to `run_dx_model()` in `team_code.py`
- added `Predicting()` class and `predict_proba()` function to `seresnet18.py`
- changed order of arguments in `ECGDataset` dataloader to account for absence of labels in test set
- added `multiclass_predict_from_logits` to `team_helper_code.py` to return multiclass predictions from logits (need to look at this)

#### TL Update 26th Feb:
- check why the ready-made function to load signal loads the signal in shape (5000, 12) instaed of (12, 5000)?! 
- finish args dict for initialisation
- check: what if ECGdatast gets only records as load_signal() function loads the signal AND the corresponding header?!

## General
- cross-validation script with examples (SS todo)