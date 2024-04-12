## Classification
### TODO
- currently, we make a list of file paths to pass into resnet, there is probably a better option?
 - update first submission 03/04: model uses file paths to .dat files when training, but directly receives reconstructed signal when testing. May want to also train on reconstructed signals, or make other adjustments for robustness
- currently, demographics for classification are sex and age, is it worth adding height and weight?
 - note demographics are not present in hidden test set unless extracted from image - currently useless in inference.
- ~~check labels/label encoding: training loop has comment that labels are SNOMED CT codes, but this is not the case~~ SS 10/04

#### Low priority
- looks like frequency (fs) is passed to dataloaders but not used, do we need this?
- investigate 'Bottleneck' in seresnet18.py
- turn `list[list]` into np.array in ECGDataset

## Reconstruction
### TODO
- how does ECG miner decide how long a signal is? Currently approaching ~2000 samples (should be 1000)
- multiple image files for a single ECG are ignored currently
- ~~clean up `ECGClass.py` (random `__main__` here, commented out for now)~~ SS 10/04
- ~~Current hard coded refernce pulse height, can use grid lines to detect scale instead~~ done DW 03/04
- find a way to deal with column gaps in ECG (try: checking if two predicted ECGs overlap?)
- time() print statements to figure out why it's taking so long to run
- ~~overflow warning (:~~ done SS 03/04
- add option to return cleaned image for inspection before reconstruction step: currently fails on some images 

## General
- cross-validation for classification
- validation script for reconstruction
- would be nice to have a cv script that saves both cleaned and reconstructed images before classification
- Train model on both 100 Hz and 500 Hz data - seems to fail currently if predicting at a different frequency than training data