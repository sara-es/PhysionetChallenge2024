# Nicola Dinsdale 2024
# Classify whether the data is real or fake 
#####################################################################################################
# Import dependencies 
import numpy as np
from datasets import numpy_dataset
from utils import Args, EarlyStopping
from models.classifier import ResNet_adapt
import torch
import torch.nn as nn
import torch.optim as optim
from losses.ecg_losses import ComboLoss
from torch.autograd import Variable
from sklearn.utils import shuffle
from torch.utils.data import DataLoader
import sys
from sklearn.metrics import accuracy_score 

########################################################################################################################
# Values to set 
args = Args()       # This is just a class to pass values efficiently to the training loops
args.epochs = 1000
args.batch_size = 32
args.patience = 25  # For the early stopping
args.train_val_prop = 0.9
args.learning_rate = 0.5e-3

reduce_lr = True # Decay the learning rate --> currently just hard coded 

patchsize = 128 # Assumes square patches
augmentation = True # True or false 

max_samples = 1000 # reduce the max number of samples because more samples than I have RAM for, set to False if you have more RAM than me

LOAD_PATH_CLASSIFIER = None

PATH_UNET = '/home/bras3596/ECG/classifier_run1_128_aug'
CHK_PATH_UNET = '/home/bras3596/ECG/classifier_run1_128_aug_checkpoint'
LOSS_PATH = '/home/bras3596/ECG/classifier_losses_run1_128_aug'

X1_LOAD_PATH = '/home/bras3596/data/ECG/X_real_train_128.npy'
X2_LOAD_PATH = '/home/bras3596/data/ECG/X_train_128.npy'

cuda = torch.cuda.is_available()

########################################################################################################################
def train_normal(args, model, train_loader, optimizer, criterion, epoch):
    cuda = torch.cuda.is_available()
    total_loss = 0
    model.train()
    batches = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        target = target.type(torch.LongTensor)
        if cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)

        if list(data.size())[0] == args.batch_size :
            batches += 1

            # First update the encoder and regressor
            optimizer.zero_grad()
            x = model(data) 
            loss = criterion(x, target)
            loss.backward()
            optimizer.step()

            total_loss += loss

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
                           100. * (batch_idx+1) / len(train_loader), loss.item()), flush=True)
            del loss
    av_loss = total_loss / batches
    av_loss_copy = np.copy(av_loss.detach().cpu().numpy())

    del av_loss
    print('\nTraining set: Average loss: {:.4f}'.format(av_loss_copy,  flush=True))
    return av_loss_copy

def val_normal(args, model, val_loader, criterion):
    cuda = torch.cuda.is_available()
    model.eval()
    true_store = []
    pred_store = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            x = model(data)
            true_store.append(target.detach().cpu().numpy())
            pred_store.append(x.detach().cpu().numpy())

    true_store = np.array(true_store).squeeze()
    pred_store = np.array(pred_store).squeeze()
    pred_store = np.argmax(pred_store, axis=1)
    accuracy = accuracy_score(true_store, pred_store)
    print('Validation set: Accuracy: {:.4f}\n'.format(accuracy,  flush=True))
    return accuracy

########################################################################################################################

X0 = np.load(X1_LOAD_PATH)
X0 = np.transpose(X0, (0,3,1,2))
X0 = (X0 - X0.mean())/X0.std()

X1 = np.load(X2_LOAD_PATH)
X1 = np.transpose(X1, (0,3,1,2))
X1 = (X1 - X1.mean())/X1.std()

X0 = shuffle(X0)
X1 = shuffle(X1)

# Many fewer samples for X0 so just subsampling for balanced classifier 
X1 = X1[:len(X0)]
y0 = np.zeros((len(X0)))
y1 = np.ones((len(X1)))

print(X0.shape, y0.shape)
print(X1.shape, y1.shape)

X = np.append(X0, X1, axis=0)
y = np.append(y0, y1, axis=0)
X, y = shuffle(X, y)
print(X.shape, y.shape)

proportion = int(args.train_val_prop * len(X))
X_train, y_train = X[:proportion], y[:proportion]
X_val, y_val = X[proportion:], y[proportion:]

print('Training: ', X_train.shape, y_train.shape)
print('Validation: ', X_val.shape, y_val.shape)

print('Creating datasets and dataloaders')
train_dataset = numpy_dataset.numpy_dataset_classification(X_train, y_train, transform=augmentation)
val_dataset = numpy_dataset.numpy_dataset_classification(X_val, y_val, transform=augmentation)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

criterion = nn.CrossEntropyLoss()

# Load the model
model =  ResNet_adapt(embsize=2, weights='DEFAULT')

if cuda:
    model = model.cuda()
if LOAD_PATH_CLASSIFIER:
   print('Loading Weights')
   encoder_dict = model.state_dict()
   pretrained_dict = torch.load(LOAD_PATH_CLASSIFIER)['model_state_dict']
   pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
   print('weights loaded unet = ', len(pretrained_dict), '/', len(encoder_dict))
   model.load_state_dict(torch.load(LOAD_PATH_CLASSIFIER)['model_state_dict'])
   
optimizer = optim.AdamW(lr=args.learning_rate, params=model.parameters())
early_stopping = EarlyStopping(args.patience, verbose=False)

epoch_reached = 1
loss_store = []

for epoch in range(epoch_reached, args.epochs+1):
    print('Epoch ', epoch, '/', args.epochs, flush=True)
    loss = train_normal(args, model, train_dataloader, optimizer, criterion, epoch)
    accuracy = val_normal(args, model, val_dataloader, criterion)
    loss_store.append([loss, accuracy])
    np.save(LOSS_PATH, np.array(loss_store))

    # Decide whether the model should stop training or not
    early_stopping(-accuracy, model, epoch, optimizer, loss, CHK_PATH_UNET)

    if early_stopping.early_stop:
        loss_store = np.array(loss_store)
        np.save(LOSS_PATH, loss_store)
        sys.exit('Patience Reached - Early Stopping Activated')
        
    if reduce_lr:
        if early_stopping.counter == 5:
            print('Reducing learning rate')
            args.learning_rate = args.learning_rate/2
            optimizer = optim.AdamW(lr=args.learning_rate, params=model.parameters())
        if early_stopping.counter == 10:
            print('Reducing learning rate')
            args.learning_rate = args.learning_rate/2
            optimizer = optim.AdamW(lr=args.learning_rate, params=model.parameters())
        if early_stopping.counter == 15:
            print('Reducing learning rate')
            args.learning_rate = args.learning_rate/2
            optimizer = optim.AdamW(lr=args.learning_rate, params=model.parameters())
        if early_stopping.counter == 20:
            print('Reducing learning rate')
            args.learning_rate = args.learning_rate/2
            optimizer = optim.AdamW(lr=args.learning_rate, params=model.parameters())
                       
    if epoch == args.epochs:
        print('Finished Training', flush=True)
        print('Saving the model', flush=True)

        # Save the model in such a way that we can continue training later
        torch.save(model.state_dict(), PATH_UNET)
        loss_store = np.array(loss_store)
        np.save(LOSS_PATH, loss_store)

    torch.cuda.empty_cache()  # Clear memory cache
    
