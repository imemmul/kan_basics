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

test_images_path = '/home/emir-machine/dev/datasets/t10k-images-idx3-ubyte'
test_labels_path = '/home/emir-machine/dev/datasets/t10k-labels-idx1-ubyte'
test_images = read_idx(test_images_path)
test_labels = read_idx(test_labels_path)
test_images = test_images.astype('float32') / 255.0
test_images = torch.tensor(test_images).unsqueeze(1)
test_labels = torch.tensor(test_labels, dtype=torch.long)

test_dataset = TensorDataset(test_images, test_labels)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

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
print(train_labels_one_hot.shape)
batch_size = 1
train_dataset = TensorDataset(train_images, train_labels_one_hot)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

torch.manual_seed(472)

train_dataset_subset = torch.utils.data.random_split(train_dataset, [5000, len(train_dataset)-5000])[0]


# TODO: fix batch size mismatch issue
batch_size = 1
train_loader = DataLoader(train_dataset_subset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = KANClassification(input_channel=1, height=28, width=28, device=device)
loss_fun = CrossEntropyLoss()
learning_rate = 0.01

epochs = 10
corrects = 0
for epoch in range(epochs):
    epoch_loss = 0
    for i, batch in enumerate(train_loader):
        batch_images, batch_labels = batch
        batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
        # print(batch_images.shape, batch_labels.shape)
        preds = model.forward(batch_images)
        preds = preds.unsqueeze(0)
        loss = loss_fun(preds, batch_labels)
        epoch_loss += loss
        model.backward(loss_fun.dloss_dy.squeeze())
        corrects += (preds.argmax(dim=1) == batch_labels).sum().item()
        
        model.zero_grad(which=['xin'])
        print(f"Batch {i}/{len(train_loader)}: Loss = {loss / batch_size}")
    epoch_loss /= len(train_loader)
    print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')
    model.update()
    model.zero_grad(which=['weights', 'bias'])
    print(f"Train Accuracy: {corrects / 10000}")


model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_images, batch_labels in test_loader:
        batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
        
        # Forward pass
        outputs = model.forward(batch_images)
        # Calculate accuracy
        _, predicted = torch.max(outputs.unsqueeze(0), 1)
        print(predicted, batch_labels)
        correct += (predicted == batch_labels).sum().item()
        print(correct)

accuracy = 100 * correct / (len(test_loader))
print(f'Accuracy on test dataset: {accuracy:.2f}%')