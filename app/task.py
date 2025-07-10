"""app: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


class Net(nn.Module):
    """Rede Neural Convolucional otimizada para Fashion MNIST.
    
    Arquitetura projetada especificamente para imagens 28x28 grayscale
    do dataset Fashion MNIST, com 10 classes de roupas/acessórios.
    """

    def __init__(self):
        super(Net, self).__init__()
        # Primeira camada convolucional: 1 canal de entrada (grayscale)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling e dropout para regularização
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Camadas totalmente conectadas
        # Após 3 poolings: 28 -> 14 -> 7 -> 3 (com padding), então 3*3*128 = 1152
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)  # 10 classes para Fashion MNIST

    def forward(self, x):
        # Primeira camada convolucional + ReLU + Pooling
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        
        # Segunda camada convolucional + ReLU + Pooling
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7
        
        # Terceira camada convolucional + ReLU + Pooling + Dropout
        x = self.pool(F.relu(self.conv3(x)))  # 7x7 -> 3x3
        x = self.dropout1(x)
        
        # Flatten para camadas totalmente conectadas
        x = x.view(x.size(0), -1)
        
        # Camadas totalmente conectadas com dropout
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Saída final (logits)
        
        return x


fds = None  # Cache FederatedDataset


def load_data(partition_id: int, num_partitions: int):
    """Load partition Fashion MNIST data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="zalando-datasets/fashion_mnist",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    
    # Normalização correta para Fashion MNIST (grayscale)
    # Usando estatísticas específicas do Fashion MNIST
    pytorch_transforms = Compose([
        ToTensor(),  # Converte para tensor e normaliza para [0,1]
        Normalize((0.2860,), (0.3530,))  # Média e desvio padrão para Fashion MNIST
    ])

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    # Otimizador Adam com learning rate ajustado para Fashion MNIST
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
    net.train()
    running_loss = 0.0
    
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"]
            labels = batch["label"]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
