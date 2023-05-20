import torch
import torch.optim as optim
import torch.nn as nn

from net import Net

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def train(train_data_loader):
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(train_data_loader, 0):
            inputs, labels = data

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    PATH = './fashionMNIST_net.pth'
    torch.save(net.state_dict(), PATH)
