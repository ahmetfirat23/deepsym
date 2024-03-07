import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import MSELoss
from encoder import Encoder
from decoder import Decoder
from tqdm.auto import tqdm
import time

class Autoencoder(nn.Module):
    def __init__(self, encoder_type, decoder_type):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(encoder_type)
        self.decoder = Decoder(decoder_type)
        
    def forward(self, x, action):
        x = self.encoder(x, action)
        x = self.decoder(x)
        return x

    def train_model(self, dataloader, validation_dataloader, epochs, lr):
        best_val_loss = float("inf")
        self.train()
        optimizer = Adam(self.parameters(), lr=lr, amsgrad=True)
        criterion = MSELoss()
        for epoch in (pbar := tqdm(range(epochs))):
            running_loss = 0
            for batch in dataloader:
                x, action, effect = batch
                optimizer.zero_grad()
                output = self(x, action)
                loss = criterion(output, effect)
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
            pbar.set_description(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.4f}")

            if (epoch+1) % 20 == 0:
                self.eval()
                with torch.no_grad():
                    start_time = time.time()
                    val_loss = 0
                    for batch in validation_dataloader:
                        x, action, effect = batch
                        output = self(x, action)
                        val_loss += criterion(output, effect)
                    val_loss /= len(dataloader)
                    end_time = time.time()
                    print(f"Validation loss: {val_loss:.4f}, Time: {end_time-start_time:.2f}s")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.save(f"best_model_{epoch}_epoch.pth")
                    self.train()
        self.eval()
        return self

    def predict(self, x, action):
        self.eval()
        return self(x, action) # This is the effect symbol
    
    def get_embedding(self, x, action):
        self.eval()
        return self.encoder(x, action) # This is the object symbol
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
        return self
    
    
