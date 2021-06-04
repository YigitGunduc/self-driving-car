# Imports
import argparse
import torch
import torchvision.transforms as transforms  
from torch import optim  
from torch import nn   
from torch.utils.data import DataLoader  
from tqdm import tqdm
from models import ResNetMini
import dataset
from datasetsampler import ImbalancedDatasetSampler


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
in_channels = 3
num_classes = 9 
learning_rate = 0.002
BATCH_SIZE = 64
EPOCHS = 100 
model_name = 'ResNetMini'

#dataset
dataset  = dataset.SelfDrivingDataset(
        csv_file='annotations.csv',
        root_dir='data/',
        transform=transforms.ToTensor()
)


train_loader = DataLoader(dataset=dataset, sampler=ImbalancedDatasetSampler(dataset), batch_size=BATCH_SIZE, shuffle=False)


#model selection
model = ResNetMini(in_channels, num_classes)


#loss and optim
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


#accuracy function
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)


    model.train()

    return num_correct/num_samples


def fit(dataset, model, epochs, loss_fn, optimizer, device, save_at=10, see_acc=20):
    for epoch in range(1, epochs + 1):
        print(f'Epoch [{epoch}/{EPOCHS}]')
        for _, (data, targets) in enumerate(tqdm(train_loader)):
            data = data.to(device=device)
            targets = targets.to(device=device)

            # forward
            scores = model(data)
            loss = loss_fn(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

        if epoch % see_acc== 0:
            print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")

        if epoch % save_at == 0:
            torch.save(model, f'driveNet-{model_name}.pth')

if __name__ == '__main__':
    fit(train_loader, model, EPOCHS, loss_fn, optimizer, device, save_at=1, see_acc=20)


