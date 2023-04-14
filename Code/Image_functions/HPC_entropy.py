import shutil
import torch
import torch.nn as nn
import numpy as np
torch.manual_seed(100)
np.random.seed(100)
import compress_entropy
import generateCompressionImages
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

global counter 
counter = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def save_ckp(state, is_best, checkpoint_dir="./entropy_models/rest/", best_model_dir="./entropy_models/best/"):
    global counter 
    f_path = checkpoint_dir + str(counter) + '_checkpoint.pt'
    counter = counter + 1
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_dir + 'best_model.pt'
        shutil.copyfile(f_path, best_fpath)

#def load_ckp(model, optimizer, checkpoint_fpath="./models/rest/15_checkpoint.pt"):
#    checkpoint = torch.load(checkpoint_fpath)
#    model.load_state_dict(checkpoint['state_dict'])
#    optimizer.load_state_dict(checkpoint['optimizer'])
#    return model, optimizer, checkpoint['epoch'], checkpoint['index'], checkpoint['min_lr'], checkpoint['max_lr'], checkpoint['steps'], checkpoint['step_size'], checkpoint['falling'], checkpoint['startup']


loss_fn = nn.L1Loss(reduction='mean')  
startup = True
min_lr = 0.002
max_lr = 0.008
decay = 0.95
steps = 200  # might want to bump this up if the HPC is much faster than my computer
falling = True
start_epoch = 0
start_index = 0
momentum = 0.94
step_size = (max_lr-min_lr)/steps

printing = True
epochs = 100
batch_size= 6


model = compress_entropy.Compress().to(device).to(memory_format=torch.channels_last)
optimizer = torch.optim.SGD(model.parameters(), lr=max_lr, momentum=momentum)



torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.autograd.set_detect_anomaly(False, check_nan=False)
torch.autograd.profiler.profile(enabled=False)
torch.autograd.profiler.emit_nvtx(enabled=False)
torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(True)


scaler = GradScaler()
los = [0]*10
for epoch in range(start_epoch, epochs):
    if startup:
        training = generateCompressionImages.MakeIter(start_index=start_index if epoch == start_epoch else 0, startup = True)
        training_loader = torch.utils.data.DataLoader(training, batch_size=4, num_workers=2)
        min_lr *= batch_size**0.5
        max_lr *= batch_size**0.5
        step_size = (max_lr-min_lr)/steps
        optimizer.param_groups[-1]['lr'] = max_lr
    else:
        training = generateCompressionImages.MakeIter(start_index=start_index if epoch == start_epoch else 0, epoch=epoch, startup = False)
        training_loader = torch.utils.data.DataLoader(training, batch_size=batch_size, num_workers=2)


    for index, data in enumerate(training_loader):
        inputs, labels = data
        labels = torch.unsqueeze(labels, dim=-1)
        inputs.to(memory_format=torch.channels_last)
        labels.to(memory_format=torch.channels_last)

        with autocast():
            outputs = model(inputs)            
            loss = loss_fn(outputs, labels)

        scaler.scale(loss).backward()#loss.backward()

        if startup:
            los[index%10] = loss.item()
                   
            scaler.step(optimizer)#optimizer.step()
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        else:
            scaler.step(optimizer)#optimizer.step()
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if printing and index % 100 in [i for i in range(90,100)]:
                los[index%10] = loss.item()


        if falling:
            optimizer.param_groups[-1]['lr'] = optimizer.param_groups[-1]['lr'] - step_size
            if optimizer.param_groups[-1]['lr'] < min_lr:
                falling = False
                max_lr *= decay
                min_lr *= decay
                steps /= decay
                step_size = (max_lr-min_lr)/steps
                if printing:
                    print("Saving model !!")
                    print("Min lr")
                    print(min_lr)
                    print("Max lr")
                    print(max_lr)
                checkpoint = {'epoch': epoch, 'index': index, 'min_lr': min_lr, 'max_lr': max_lr, 'steps': steps, 'step_size': step_size, 'falling': falling, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'startup': startup}
                save_ckp(checkpoint, True)
                
                if startup and sum(los)/10 < 0.20 and max(los) < 0.25:
                    startup = False
                    max_lr /= 4
                    min_lr /= 4
                    steps *= 2
                    step_size = (max_lr-min_lr)/steps
                    falling = True
                    optimizer.param_groups[-1]['lr'] = max_lr
                    print("startup done ! !")
                    break
        else: 
            optimizer.param_groups[-1]['lr'] += step_size
            if optimizer.param_groups[-1]['lr'] > max_lr:
                falling = True

        if (startup and printing and index % 10  == 0):
            print("loss")
            print(los)
            print("lr")
            print(optimizer.param_groups[-1]['lr'])
        elif (not startup and printing and index % 100 == 0):
            print("loss")
            print(los)#loss.item())
            print("lr")
            print(optimizer.param_groups[-1]['lr'])