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

## Reconstruction
- fix issue with different sampling frequencies (postprocessor?)
- classifier to judge if returned signal looks like an ecg: YOLO maybe?
- header/text extraction for features 

## Classification
- resnet missing feature flags + features for height/weight 
- ensemble resnet
- train resnet checkpoint
- check model on both 100 Hz and 500 Hz data 