import hydra
from models import KANBaseline, TinyVGGKAN, NNBaseline
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from kans import KAN, FastKANLayer, FastKAN
from torchvision import transforms
import torch
from torchvision.datasets import FashionMNIST, CIFAR10
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import os

def get_model(model_name):
    if model_name == "kanbaseline":
        return KANBaseline
    elif model_name == "tinyvggkan":
        return TinyVGGKAN
    else:
        return NNBaseline
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_basic(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            # images = images.view(images.size(0), -1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(test_loader.dataset)
    accuracy = 100 * correct / total
    print(f"Test Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

def save_model(model, cfg, epoch):
    os.makedirs(cfg.save_dir, exist_ok=True)
    save_path = os.path.join(cfg.save_dir, f"model_epoch_{epoch+1}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def load_dataset(dataset_path, dataset_name, batch_size):
    transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomCrop(32, padding=4),
        # transforms.ToTensor(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    if dataset_name == "fmnist":
        train_dataset = FashionMNIST(root=dataset_path, train=True, download=True, transform=transform)
        test_dataset = FashionMNIST(root=dataset_path, train=False, download=True, transform=transform)
    elif dataset_name == "cifar10":
        train_dataset = CIFAR10(root=dataset_path, train=True, download=True, transform=transform)
        test_dataset = CIFAR10(root=dataset_path, train=False, download=True, transform=transforms.ToTensor())
    else:
        raise ValueError("Unknown dataset") 
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    return train_dataloader, test_dataloader

def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name} grad mean: {param.grad.mean().item()}, std: {param.grad.std().item()}")
        else:
            print(f"{name} grad is None")

class LinearNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LinearNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x


# FIXME KAN performs worse than fastkan why ???
@hydra.main(config_path='./configs', config_name='fmnist.yaml')
def main(cfg):
    model_cls = get_model(cfg.model_name)
    model = model_cls(input_channel=1, hidden_channels=32, n_classes=10, height=28, width=28, device=cfg.device, use_fastkan=True) # 32 is too much
    print(f"Model: {cfg.model_name},\nParameters: {count_parameters(model)}")
    model.to(cfg.device)
    train_loader, test_loader = load_dataset(cfg.dataset_path, cfg.dataset_name, cfg.train_batch_size)
    loss_func = nn.CrossEntropyLoss(label_smoothing=cfg.loss.label_smoothing)
    if cfg.optim.type == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=cfg.optim.learning_rate,
                                      betas=(cfg.optim.adam_beta1, cfg.optim.adam_beta2),
                                      weight_decay=cfg.optim.adam_weight_decay,
                                      eps=cfg.optim.adam_epsilon)
    elif cfg.optim.type == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.optim.learning_rate, momentum=cfg.optim.momentum)
    elif cfg.optim.type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optim.learning_rate, betas=(cfg.optim.adam_beta1, cfg.optim.adam_beta2))
    if cfg.inference:
        
        n_model = model_cls(input_channel=1, hidden_channels=4, n_classes=10, height=28, width=28, device=cfg.device, use_fastkan=True)
        model.load_state_dict(torch.load(cfg.model_path))
        model.eval()
        pbar = tqdm(test_loader, desc="Inference")
        for images, labels in pbar:
            with torch.no_grad():
                output = model(images)
                n_output = n_model(images)
        model.conv_layers[0].save_attention_gif('trained_attention_maps.gif')
        n_model.conv_layers[0].save_attention_gif('untrained_attention_maps.gif')
    else:
        model.train()
        for epoch in range(cfg.epochs):
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            pbar = tqdm(train_loader, desc="Training")
            one_epoch_start = time.time()
            for images, labels in pbar:
                start = time.time()
                images, labels = images.to(cfg.device), labels.to(cfg.device)
                optimizer.zero_grad()
                # images = images.view(images.size(0), -1)
                outputs = model(images)
                loss = loss_func(outputs, labels)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                loss.backward()
                optimizer.step()

                # Calculate the number of correct predictions
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)

                pbar.set_postfix_str(f"Loss: {loss.item():.4f}")
                running_loss += loss.item() * images.size(0)
                end = time.time()

            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_accuracy = correct_predictions / total_predictions * 100
            one_epoch_end = time.time()

            print(f"Epoch {epoch+1} Training Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, Time: {one_epoch_end - one_epoch_start:.2f}s")

            if (epoch + 1) % cfg.save_interval == 0 or (epoch + 1) == cfg.epochs:
                save_model(model, cfg, epoch)
                if epoch % (cfg.epochs // 4) == 0:
                    evaluate_basic(model=model, test_loader=test_loader, criterion=loss_func, device=cfg.device)

if __name__ == "__main__":
    main()
    