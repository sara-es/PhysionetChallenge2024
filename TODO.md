## Classification
### TODO
- test on larger dataset (on tiny test we are returning all 'Norm')
- unittests for dataloaders 
- single preprocessing function for both train and test
- looks like frequency (fs) is passed to dataloaders but not used, do we need this?
- currently, we make a list of file paths to pass into resnet, there is probably a better option?
- currently, demographics for classification are sex and age, is it worth adding height and weight?

#### Low priority
- investigate 'Bottleneck' in seresnet18.py
- turn `list[list]` into np.array in ECGDataset
- allow for multiple GPUs if multiple devices are available
- worth dynamically determining number of ECG leads?

## Reconstruction
### TODO
- multiple image files for a single ECG are ignored currently
- clean up `ECGClass.py` (random `__main__` here, commented out for now)
- add option to return cleaned image for inspection before reconstruction step

## General
- cross-validation for classification
- validation script for reconstruction