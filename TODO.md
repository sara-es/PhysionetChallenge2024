## Image generation (image-kit)
- different layouts
- vertical bars in ecg
**Low prio:**
- handwriting or approximation thereof
- add artefacts (e.g. tipp-ex) to generated images
- different gap sizes between leads in case of no rhythm strip
- different resolutions/image size?

## Preprocessing/Image cleaning
- image scaling
- layout detection/validation
- eval script to show/test both above

## YOLO
- generate yaml file with train/test/val paths after image generation

## Reconstruction
**Low prio:**
- signal quality metric for reconstructed signal: return nans if reconstruction has failed (+lead arithmetic for redundancy/error checking in reconstruction)
- check if using first pixel of each lead as a baseline improves SNR
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