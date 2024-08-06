# this file supposed to test running time of the conv kan
from models import KANBaseline, NNBaseline
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler, autocast

t = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))])

model = KANBaseline(1, 32, 10, 28, 28, device='cuda', use_fastkan=True).cuda()
# model = NNBaseline(1, 32, 10, 28, 28, device='cuda').to('cuda')

inputs = torch.randn(128, 1, 28, 28).cuda()

# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
#     with record_function("model_inference"):
#         model(inputs)

# print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=None))

target = torch.randint(0, 10, (128, 10,), dtype=torch.float32).to('cuda')

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_func = torch.nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()

def train_and_profile(model, data, target):
    model.train()
    optimizer.zero_grad()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with autocast():
            output = model(data)
            loss = loss_func(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total"))

# Start profiling
train_and_profile(model, inputs, target)