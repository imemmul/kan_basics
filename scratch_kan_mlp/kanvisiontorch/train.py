import numpy as np
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder
from convkantorch import KANClassification
from loss import CrossEntropyLoss


def read_idx(filename):
    with open(filename, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), byteorder='big')
        num_dims = magic_number & 0xFF
        shape = tuple(int.from_bytes(f.read(4), byteorder='big') for _ in range(num_dims))
        
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(shape)

# Load training images
train_images_path = '/home/emir-machine/dev/datasets/train-images-idx3-ubyte'
train_labels = '/home/emir-machine/dev/datasets/train-labels-idx1-ubyte'
train_images = read_idx(train_images_path)
train_labels = read_idx(train_labels)
print(f'Training images shape: {train_images.shape}')

train_images = train_images.astype('float32') / 255.0

onehot_encoder = OneHotEncoder(sparse=False)
train_labels_one_hot = onehot_encoder.fit_transform(train_labels.reshape(-1, 1))


train_images = torch.tensor(train_images).unsqueeze(1)  # Add channel dimension
train_labels_one_hot = torch.tensor(train_labels_one_hot, dtype=torch.float32)

batch_size = 1
train_dataset = TensorDataset(train_images, train_labels_one_hot)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

train_images = torch.tensor(train_images)
train_labels = torch.tensor(train_labels, dtype=torch.long)

train_dataset = TensorDataset(train_images, train_labels)
torch.manual_seed(42)

train_dataset_subset = torch.utils.data.random_split(train_dataset, [10000, len(train_dataset)-10000])[0]


# TODO: fix batch size mismatch issue
batch_size = 1
train_loader = DataLoader(train_dataset_subset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = KANClassification(input_channel=1, height=28, width=28, device=device)
loss_fun = CrossEntropyLoss()
learning_rate = 0.01

epochs = 5

for epoch in range(epochs):
    epoch_loss = 0
    for i, batch in enumerate(train_loader):
        batch_images, batch_labels = batch
        batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
        # print(batch_images.shape, batch_labels.shape)
        preds = model.forward(batch_images)
        print(preds)
        loss = loss_fun(preds, batch_labels)
        epoch_loss += loss
        
        # Backward pass
        model.backward(loss_fun.dloss_dy.squeeze())
        
        model.zero_grad(which=['xin'])
        print(f"Batch {i}/{len(train_loader)}: Loss = {loss / batch_size}")
    epoch_loss /= len(train_loader)
    print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')
    model.update()
    model.zero_grad(which=['weights', 'bias'])    