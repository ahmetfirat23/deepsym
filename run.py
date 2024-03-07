from autoencoder import Autoencoder
from dataloader import load_data, get_dataloader
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.optim import Adam
from utils import default_transform

# Load the data
batch_size = 128
epochs = 300
lr = 5e-5
images, actions, effects = load_data("data/img/obs_prev_z.pt", "data/img/action.pt", "data/img/delta_pix_1.pt")
train_size = int(len(images) * 0.8)
val_size = int(len(images) * 0.1)
test_size = len(images) - train_size - val_size
train_images = images[:train_size]
train_actions = actions[:train_size]
train_effects = effects[:train_size]
val_images = images[train_size:train_size + val_size]
val_actions = actions[train_size:train_size + val_size]
val_effects = effects[train_size:train_size + val_size]
test_images = images[train_size + val_size:]
test_actions = actions[train_size + val_size:]
test_effects = effects[train_size + val_size:]

transform = default_transform(size=42, affine=True, mean=0.279, std=0.0094)
dataloader = get_dataloader(images, actions, effects, batch_size, True, transform=None)
validation_dataloader = get_dataloader(val_images, val_actions, val_effects, batch_size, True)
test_dataloader = get_dataloader(test_images, test_actions, test_effects, batch_size, True)
autoencoder = Autoencoder("f1", "f1")
autoencoder = autoencoder.train_model(dataloader, validation_dataloader, epochs, lr)
autoencoder.save("autoencoder.pth")
autoencoder.load("autoencoder.pth")
# test_dataloader.dataset.show_example(0)
print(autoencoder.get_embedding(images[0].unsqueeze(0), actions[0].unsqueeze(0)))
# dataloader.dataset.show_example(1)
print(autoencoder.get_embedding(images[100].unsqueeze(0), actions[100].unsqueeze(0)))
# Calculate the loss on test
autoencoder.eval()
criterion = nn.MSELoss()
test_loss = 0
for batch in test_dataloader:
    x, action, effect = batch
    output = autoencoder(x, action)
    test_loss += criterion(output, effect)
test_loss /= len(test_dataloader)
print(f"Test loss: {test_loss:.4f}")