
# Storing the parameters considering classification model (ResNet)
class Resnet_args:
    def __init__(self):
        self.batch_size = 64
        self.epochs = 50
        self.learning_rate = 0.003
        self.weight_decay = 0.00001
        self.in_channels = 12

        self.kfold = True
        self.n_splits = 5

