import torch
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import make_grid
from imageOperations import show
from train import train
from net import Net

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

batch_size = 4

training_data_set = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transform,
)

train_data_loader = DataLoader(training_data_set, batch_size=batch_size, shuffle=True, num_workers=2)

test_data_set = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transform,
)

test_data_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=False, num_workers=2)

classes = datasets.FashionMNIST.classes

if __name__ == "__main__":
    random_data_iterator = iter(train_data_loader)
    images, labels = next(random_data_iterator)
    show(make_grid(images))

    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

if __name__ == "__main__": 
    train(train_data_loader)

if __name__ == "__main__":
    random_data_iterator = iter(test_data_loader)
    images, labels = next(random_data_iterator)
    show(make_grid(images))

    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    PATH = './fashionMNIST_net.pth'
    net = Net()
    net.load_state_dict(torch.load(PATH))

    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(batch_size)))