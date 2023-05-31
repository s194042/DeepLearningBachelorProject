import shutil
import torch
import torch.nn as nn
import numpy as np
torch.manual_seed(100)
np.random.seed(100)
import Lossv2
import generateLossImages
import time

global counter 
counter = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def save_ckp(state, is_best, checkpoint_dir="./models/rest/", best_model_dir="./models/best/"):
    global counter 
    f_path = checkpoint_dir + str(counter) + '_checkpoint.pt'
    counter = counter + 1
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_dir + 'best_model.pt'
        shutil.copyfile(f_path, best_fpath)

def load_ckp(model, optimizer, checkpoint_fpath="./models/best/best_model.pt"):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch'], checkpoint['index'], checkpoint['min_lr'], checkpoint['max_lr'], checkpoint['steps'], checkpoint['step_size'], checkpoint['falling'], checkpoint['threshold']


load = True

if load:
    loss_fn = nn.L1Loss()
    model = Lossv2.Loss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.96)
    model, optimizer, start_epoch, start_index, min_lr, max_lr, steps, step_size, falling, threshold = load_ckp(model, optimizer)
else:
    loss_fn = nn.L1Loss()
    min_lr = 0.001
    max_lr = 0.003
    decay = 0.8
    steps = 300
    falling = True
    start_epoch = 0
    start_index = 0
    momentum = 0.94
    threshold = [0.16, 0.12, 0.09, 0.06, 0.04, 0.03, 0.02, 0.015, 0.01 -1]
    step_size = (max_lr-min_lr)/steps
    model = Lossv2.Loss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=max_lr, momentum=momentum)




printing = True


running_loss = 0.
last_loss = 0.
lis = []
# might have to delete when loading model
threshold_decay = 0.2
flags = [True for _ in threshold]
record = 1
epochs = 100

times = [0]*10
transforms = 108
for epoch in range(start_epoch, epochs):
    training = generateLossImages.MakeIter(generateLossImages.get_image_pairs_transforms_with_loss(cupy=True, start=start_index if epoch == start_epoch else 0)) # start=2*71*50*4))#start_index if epoch == start_epoch else 0)) 
    val = generateLossImages.get_image_pairs_transforms_with_loss("C:/Users/Rani/Desktop/ai_val/16")
    training_loader = torch.utils.data.DataLoader(training)
    for index, data in enumerate(training_loader): # loading data takes 25.72% of the time

        inputs, labels = data  #0.01%

        labels = labels.to("cuda") # 0.00%

        # Make predictions for this batch
        outputs = model(inputs) # 24.05%

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels) # 0.40%

        loss.backward() # 39.11%

        # Gather data and report
        running_loss += loss.item() # 10.23%

        if printing: # From here down only takes 0.47% of the time
            lis.append((f'{outputs.item():.3}',f'{labels.item():.3}'))
        if index % transforms == transforms-1:
            # Zero your gradients for every batch!
            # Adjust learning weights
            last_loss = running_loss / transforms # loss per batch
            if last_loss < threshold[0] and flags[0]:
                threshold = threshold[1:]
                flags = flags[1:]
                flag1 = False
                min_lr *= threshold_decay
                max_lr *= threshold_decay
                step_size = (max_lr-min_lr)/steps
                falling = True
                optimizer.param_groups[-1]['lr'] = max_lr
                if printing:
                    print(min_lr)
                    print(max_lr)
                    print(steps)
                    print(step_size)
                checkpoint = {
                    'epoch': epoch,
                    'index': index,
                    'min_lr': min_lr,
                    'max_lr': max_lr,
                    'steps': steps,
                    'step_size': step_size,
                    'falling': falling,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'threshold': threshold
                }
                save_ckp(checkpoint, True)
            elif falling:
                optimizer.param_groups[-1]['lr'] = optimizer.param_groups[-1]['lr'] - step_size
                if optimizer.param_groups[-1]['lr'] < min_lr:
                    falling = False
                    max_lr *= decay
                    min_lr *= decay
                    steps /= decay
                    step_size = (max_lr-min_lr)/steps

                    if printing:
                        print(min_lr)
                        print(max_lr)
                        print(steps)
                        print(step_size)
                    
                    
                    checkpoint = {
                        'epoch': epoch,
                        'index': index,
                        'min_lr': min_lr,
                        'max_lr': max_lr,
                        'steps': steps,
                        'step_size': step_size,
                        'falling': falling,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'threshold': threshold
                    }
                    save_ckp(checkpoint, False)

            else: 
                optimizer.param_groups[-1]['lr'] += step_size
                if optimizer.param_groups[-1]['lr'] > max_lr:
                    falling = True


            
            optimizer.step()
            optimizer.zero_grad()
            if printing:
                print(optimizer.param_groups[-1]['lr'])
                print('  batch {} loss: {}'.format(index + 1, last_loss))
                print(lis)
                lis = []
            running_loss = 0

            if last_loss < record:
                record = last_loss
                if last_loss < 0.08:
                    checkpoint = {
                    'epoch': epoch,
                    'index': index,
                    'min_lr': min_lr,
                    'max_lr': max_lr,
                    'steps': steps,
                    'step_size': step_size,
                    'falling': falling,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'threshold': threshold
                }
                    save_ckp(checkpoint, True)