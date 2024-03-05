import sys, os
from evaluation.five_fold_cv import five_fold_cv
from utils import default_models

def main():
    data_folder = os.path.join('tiny_testset', 'records100')
    five_fold_cv(data_folder=data_folder, multilabel_cv=True, n_splits=5, train_models=True)

if __name__ == '__main__':
    main()