## Image generation (image-kit)
- add artefacts (e.g. tipp-ex) to generated images
- handwriting or approximation thereof
- different layouts 
- different gap sizes between leads?
- different resolutions/image size?

## Preprocessing/Image cleaning
- image rotation
- image scaling
- layout detection/validation
- eval script to show/test both above

## YOLO
- generate yaml file with train/test/val paths after image generation

## Reconstruction
- fix issue with different sampling frequencies (postprocessor?)
- header/text extraction for features 
- lead arithmetic for redundancy/error checking in reconstruction

## Classification
- resnet missing feature flags + features for height/weight 
- ensemble resnet
- train resnet checkpoint
- check model on both 100 Hz and 500 Hz data 