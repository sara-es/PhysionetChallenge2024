## Image cleaning
- Will need to standardize how we remove shadows, current method fails if grid is not red

## Reconstruction
- multiple image files for a single ECG are ignored currently
- find a way to deal with column gaps in ECG (missing pixels)
- gridsize detection (detected gridsize from `reconstruction.image_cleaning.clean_image` for the inputs we tested is picking up minor gridlines, we can work around this by ignoring modes below 30 pixels for now)
- reference pulse detection

## Classification
- currently, we make a list of file paths to pass into resnet, there is probably a better option?
 - update first submission 03/04: model uses file paths to .dat files when training, but can directly receive reconstructed signal when testing. May want to also train on reconstructed signals, or make other adjustments for robustness
- currently, demographics for classification are sex and age, is it worth adding height and weight?
 - note demographics are not present in hidden test set unless extracted from image - currently useless in inference.

## General
- Train model on both 100 Hz and 500 Hz data - seems to fail currently if predicting at a different frequency than training data