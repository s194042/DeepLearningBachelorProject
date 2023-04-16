import torch
import images



def get_images(path: str='C:/Users/Rani/Desktop/ai_training_immages/_0', epoch= 0, start: int = 0, end: int=6.645*50*4*106-1):
    imgs = images.get_all_imgs_with_everything(path, start, end)
    
    for idx, img in enumerate(imgs, start=epoch):
        im = torch.divide(torch.subtract(img, 128), 128)
        aim = torch.clone(im)
        yield im.permute(2,0,1), aim.permute(2,0,1)

    
def get_images_without_flips_and_cuts(path: str='C:/Users/Rani/Desktop/ai_training_immages/_0', epoch= 0, start: int = 0, end: int=6.645*50*4*106-1):
    imgs = images.get_all_images_without_anything(path)
    for idx, img in enumerate(imgs, start=epoch):
        im = torch.divide(torch.subtract(img, 128), 128)
        aim = torch.clone(im)
        yield im.permute(2,0,1), aim.permute(2,0,1)



class MakeIter(torch.utils.data.IterableDataset):
    #@torch.jit.script
    def __init__(self, path = 'C:/Users/Rani/Desktop/ai_training_immages/', start_index: int = 0, folder: str ="_0", epoch:int = 0, startup=False, **kwargs):
        super(MakeIter).__init__()
        self.start_index = start_index
        self.kwargs = kwargs
        self.folder = folder
        self.epoch = epoch
        self.startup = startup
        self.path = path

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info == None or worker_info.num_workers == 1:
            folder = self.folder
        elif worker_info.num_workers == 2:
            folder = self.folder + str(worker_info.id)
        else:
            folder = self.folder + str(worker_info.id + 1)
        if self.startup:
            return iter(get_images_without_flips_and_cuts(path = self.path + folder))
        else:
            return iter(get_images(path = self.path + folder, epoch=self.epoch))
    
    def __getitem__(self, index:int):
        return self.generator_func.__next__()
    
    def __len__(self):
        return 6000*50*4*11# not sure if there are 6000 because of bug in raw loading #645*50*4*106-1 #-1 just to be safe