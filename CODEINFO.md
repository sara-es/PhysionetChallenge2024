# Read me for information on file structure and code

```
├───classification        # code for the classification task
├───evaluation            # scripts to evaluate the performance of the models, including 5-fold cross-validation
├───models                # saved models, e.g. .sav and .pth
├───preprocessing         # code for preprocessing the data and images
├───reconstruction        # code for the reconstruction task
├───tests                 # unittests, etc.
├───tiny_testset          # 15 samples for fast testing
│   ├───records100           # 100hz data and images
│   ├───records100_hidden    # 100hz images with data occluded, for inference testing
│   ├───records500           # 500hz data and images
│   ├───records500_hidden    # 500hz images with data occluded, for inference testing
│   └───test_outputs         # send model outputs here
└───utils                 # constants, helper functions, etc.
```

## VERY IMPORTANT
Please create a function call to your model in `train_model.py` and `test_model.py` and make sure that both of these scripts run, without errors, before submitting a pull request. If you wish to test your model outside of the `train_model.py` and `test_model.py` scripts, please do so in a separate file in the `evaluation` or `tests` directory.
#### Please make sure that `train_model.py` and `test_model.py` run without errors before submitting a pull request.