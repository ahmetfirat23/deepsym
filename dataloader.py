import torch.utils.data as data
import torch
import numpy as np
from PIL import Image

class Dataset(data.Dataset):
    def __init__(self, images, actions, effects, transform=None):
        super(Dataset, self)
        self.image_size = images.size()[1]
        self.images = images.unsqueeze(1) # tensor as N,1,x,x
        self.transform = transform
        effects_mean = effects.mean()
        effects_std = effects.std()
        self.effects = (effects - effects_mean) / (effects_std + 1e-8)
        assert len(images) == len(actions) == len(effects) , "The number of images, actions, and effects must be the same."
        self.dataset = list(zip(images, actions, effects))
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.transform:
            item = (torch.squeeze(self.transform(item[0])), item[1], item[2])
        return item
    
    def get_batch(self, idx):
        batch = self.dataset[idx*self.batch_size:(idx+1)*self.batch_size]
        if self.transform:
            for i in range(len(batch)):
                batch[i] = (torch.squeeze(self.transform(batch[i][0])), batch[i][1], batch[i][2])
        return batch
    
    def get_image(self, idx):
        # print(self.dataset[idx][0].shape)
        if self.transform:
            return torch.squeeze(self.transform(self.dataset[idx][0]))
        return self.dataset[idx][0]
    
    def get_action(self, idx):
        return self.dataset[idx][1]
    
    def get_effect(self, idx):
        return self.dataset[idx][2]
    
    def show_example(self, idx):
        Image.fromarray((self.get_image(idx).numpy()*255).astype(np.uint8)).show()
        print("Image:")
        print(self.get_image(idx))
        print("Action:")
        print(self.get_action(idx))
        print("Effect:")
        print(self.get_effect(idx))


def load_data(images_path, actions_path, effects_path):
    images = torch.load(images_path)
    actions = torch.load(actions_path)
    effects = torch.load(effects_path)
    return images, actions, effects

def get_dataloader(images, actions, effects, batch_size, shuffle, transform=None):
    dataset = Dataset(images, actions, effects, transform=transform)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)