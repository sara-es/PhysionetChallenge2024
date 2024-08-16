## Image generation (image-kit)
- different layouts
**Low prio:**
- handwriting or approximation thereof
- add artefacts (e.g. tipp-ex) to generated images
- different gap sizes between leads in case of no rhythm strip

## Preprocessing/Image cleaning
- layout detection/validation: YOLO
- rotation testing

## YOLO
- generate yaml file with train/test/val paths after image generation

## Reconstruction
- classifier for real vs generated images
- cabrera detection 
- gaps between leads ?
- fine tune magic numbers on generated images in vectorize_signals.py: can pad signals based on "xgap" in generator
- header/text extraction for features 
- check dc offset

## Unet
- implementation of two models for real and generated images

## Classification
- training data is 2.5s per lead (non-concurrent)
- train resnet checkpoint
**Low prio:**
- ensemble resnet
- check model on both 100 Hz and 500 Hz data 

# submissions list
1. digitization as-is: assumes all generated images
2. classifier for real vs. generated images + yolo for row count

- SQI: replace with NaNs instead of 0s
- optimise reconstruction of generated images (gap from generator)
- optimise SQI: length of sliding window
- no cabrera option
- deterministic rhythm lead order
- for generated images: if signals are different lengths, pad with zeros instead of interpolating

## TINY TESTSET
- occlude non-plotted data