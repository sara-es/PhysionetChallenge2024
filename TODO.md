## Image generation (image-kit)
- different layouts
**Low prio:**
- handwriting or approximation thereof
- add artefacts (e.g. tipp-ex) to generated images
- different gap sizes between leads in case of no rhythm strip

## Preprocessing/Image cleaning
- image scaling
- layout detection/validation
- eval script to show/test both above

## Reconstruction
- classifier for real vs generated images
- cabrera detection 
- gaps between leads ?
- fine tune magic numbers on generated images in vectorize_signals.py: can pad signals based on "xgap" in generator
- header/text extraction for features 

## Unet
- scaling in training process
- domain adaptation
- re-train with black rectangles in image generation
- confidence measure per image

## Classification
- resnet missing feature flags + features for height/weight 
- training data is 2.5s per lead (non-concurrent)
- train resnet checkpoint
**Low prio:**
- ensemble resnet
- check model on both 100 Hz and 500 Hz data 