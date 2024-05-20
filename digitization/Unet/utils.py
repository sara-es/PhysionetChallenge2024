# Nicola Dinsdale 2020
# Useful functions for training the model
# Args: Class of useful values
# Early stopping: exactly that
# Load pretrained model: loads statedict into model
########################################################################################################################
import torch
import numpy as np

class Args:
    # Store lots of the parameters that we might need to train the model
    def __init__(self):
        self.batch_size = 8
        self.log_interval = 10
        self.learning_rate = 1e-4
        self.epochs = 2
        self.train_val_prop = 0.9
        self.patience = 5
        self.channels_first = True
        self.diff_model_flag = False
        self.alpha = 1
        self.ref_dist=None


class EarlyStopping:
    # Early stops the training if the validation loss doesnt improve after a given patience
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, epoch, optimizer, loss, PTH):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, optimizer, loss, PTH)
        elif score < self.best_score:
            self.counter += 1
            print('Early Stopping Counter: ', self.counter, '/', self.patience)
            if self .counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, optimizer, loss, PTH)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch, optimizer, loss, PTH):
            # Saves the model when the validation loss decreases
            if self.verbose:
                print('Validation loss decreased: ', self.val_loss_min, ' --> ',  val_loss, 'Saving model ...')
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss}, PTH)


