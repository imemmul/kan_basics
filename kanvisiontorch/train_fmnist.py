import hydra
from convs import KANClassification
import torch.nn as nn
from kans import KAN
from torchvision import transforms
import torch
from torchvision.datasets import FashionMNIST, CIFAR10
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import os

def evaluate_basic(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
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

def load_dataset(dataset_path, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # train_dataset = FashionMNIST(root=dataset_path, train=True, download=True, transform=transform)
    # test_dataset = FashionMNIST(root=dataset_path, train=False, download=True, transform=transform)
    train_dataset = CIFAR10(root=dataset_path, train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root=dataset_path, train=False, download=True, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    return train_dataloader, test_dataloader

def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name} grad mean: {param.grad.mean().item()}, std: {param.grad.std().item()}")
        else:
            print(f"{name} grad is None")


@hydra.main(config_path='./configs', config_name='fmnist.yaml')
def main(cfg):
    # model = KAN(layers_shape=[784, 512, 256, 128, 64, 32, 16, 10], device=cfg.device)
    model = KANClassification(input_channel=3, n_classes=10, height=32, width=32, device=cfg.device)
    model.to(cfg.device)
    train_loader, test_loader = load_dataset(cfg.dataset_path, cfg.train_batch_size)
    loss_func = nn.CrossEntropyLoss(label_smoothing=cfg.loss.label_smoothing)
    optimizer_class = torch.optim.AdamW
    params_to_optimize = model.parameters()
    for name, param in model.named_parameters():
        print(f"{name} requires grad: {param.requires_grad}")
    optimizer = optimizer_class(
        params_to_optimize,
        lr=cfg.optim.learning_rate,
        betas=(cfg.optim.adam_beta1, cfg.optim.adam_beta2),
        weight_decay=cfg.optim.adam_weight_decay,
        eps=cfg.optim.adam_epsilon,
    )
    model.train()
    for epoch in range(cfg.epochs):
        running_loss = 0.0
        pbar = tqdm(train_loader, desc="Training")
        for images, labels in pbar:
            start = time.time()
            images, labels = images.to(cfg.device), labels.to(cfg.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            # check_gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            pbar.set_postfix_str(f"Loss: {loss.item():.4f}")
            running_loss += loss.item() * images.size(0)
            end = time.time()

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1} Training Loss: {epoch_loss:.4f}")

        if (epoch + 1) % cfg.save_interval == 0 or (epoch + 1) == cfg.epochs:
            save_model(model, cfg, epoch)
            if epoch % 2 == 0:
                evaluate_basic(model=model, test_loader=test_loader, criterion=loss_func, device=cfg.device)

if __name__ == "__main__":
    main()