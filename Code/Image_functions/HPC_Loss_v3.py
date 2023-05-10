

if __name__ == '__main__':
    import shutil
    import torch
    import torch.nn as nn
    import numpy as np
    torch.manual_seed(100)
    np.random.seed(100)
    import Lossv2
    import generateLossImages

    global counter 
    counter = 18
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def save_ckp(state, is_best, checkpoint_dir="./models/rest/", best_model_dir="./models/best/"):
        global counter 
        f_path = checkpoint_dir + str(counter) + '_checkpoint.pt'
        counter = counter + 1
        torch.save(state, f_path)
        if is_best:
            best_fpath = best_model_dir + 'best_model.pt'
            shutil.copyfile(f_path, best_fpath)

    def load_ckp(model, optimizer, checkpoint_fpath="./models/rest/17_checkpoint.pt"):
        checkpoint = torch.load(checkpoint_fpath)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        return model, optimizer, checkpoint['epoch'], checkpoint['index'], checkpoint['min_lr'], checkpoint['max_lr'], checkpoint['steps'], checkpoint['step_size'], checkpoint['falling']



    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    loss_fn = nn.L1Loss()
    min_lr = 0.002
    max_lr = 0.008
    decay = 0.96
    steps = 500
    falling = True
    start_epoch = 0
    start_index = 0
    momentum = 0.94
    step_size = (max_lr-min_lr)/steps
    model = Lossv2.Loss(seperable=False, slim=True).to(device).to(memory_format=torch.channels_last)
    optimizer = torch.optim.SGD(model.parameters(), lr=max_lr, momentum=momentum)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    load = True
    if load:
        loss_fn = nn.L1Loss()
        model = Lossv2.Loss(seperable=False, slim=True).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=max_lr, momentum=momentum)
        model, optimizer, start_epoch, start_index, min_lr, max_lr, steps, step_size, falling = load_ckp(model, optimizer)


    from torch.cuda.amp import autocast
    printing = True
    epochs = 100
    batch_size= 14
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(False, check_nan=False)
    torch.autograd.profiler.profile(enabled=False)
    torch.autograd.profiler.emit_nvtx(enabled=False)


    from torch.utils.data import DataLoader
    torch.set_grad_enabled(True)

    startup = False
    for epoch in range(start_epoch, epochs):
        if startup:
            training = generateLossImages.MakeIter(start_index=start_index if epoch == start_epoch else 0, startup = True)
            training_loader = DataLoader(training, batch_size=batch_size, num_workers=0)
            min_lr /= batch_size**0.5  # normally one puts the learning rate up by sqrt(batchsize), since that keeps the variance constant, however since we use sum in L1Loss, the oposit is true
            max_lr /= batch_size**0.5
            step_size = (max_lr-min_lr)/steps
            optimizer.param_groups[-1]['lr'] = max_lr
            los = [0]*(112//batch_size)
        else:
            training = generateLossImages.MakeIter(start_index=start_index if epoch == start_epoch else 0, epoch=epoch, startup = False)
            training_loader = DataLoader(training, batch_size=batch_size, num_workers=0)

        for index, data in enumerate(training_loader):
            inputs, labels = data
            labels = torch.unsqueeze(labels, dim=-1)
            inputs.to(memory_format=torch.channels_last)

            with autocast():
                outputs = model(inputs)            
                loss = loss_fn(outputs, labels)

            scaler.scale(loss).backward()#loss.backward()
            if startup:
                los[index%(112//batch_size)] = loss.item()/batch_size
                if index % (112//batch_size) == (112//batch_size)-1:         
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if falling:
                optimizer.param_groups[-1]['lr'] = optimizer.param_groups[-1]['lr'] - step_size
                if optimizer.param_groups[-1]['lr'] < min_lr:
                    falling = False
                    max_lr *= decay
                    min_lr *= decay
                    steps /= decay
                    step_size = (max_lr-min_lr)/steps
                    if printing:
                        print("loss")
                        print(loss.item())
                        print("Saving model !!")
                        print("Min lr")
                        print(min_lr)
                        print("Max lr")
                        print(max_lr)
                    checkpoint = {'epoch': epoch, 'index': index, 'min_lr': min_lr, 'max_lr': max_lr, 'steps': steps, 'step_size': step_size, 'falling': falling, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()
                    }
                    save_ckp(checkpoint, True)
                    
                    if startup and sum(los)/len(los) < 0.05 and max(los) < 0.1:
                        startup = False
                        max_lr /= 3
                        min_lr /= 3
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

            if (startup and printing and index % len(los) == len(los)-1) or (not startup and printing and index % 200 == 0):
                print("loss")
                print(loss.item())
                print("lr")
                print(optimizer.param_groups[-1]['lr'])
                print("pred")
                print(labels.T)
                print(outputs.T)
