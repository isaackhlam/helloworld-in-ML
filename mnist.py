import torchvision
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable

transform = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(
                                     mean = 0.5,
                                     std = 0.5
                                )])

data_train = datasets.MNIST(root = "./data/",
                            transform = transform,
                            train = True,
                            download = True)

data_test = datasets.MNIST(root = "./data/",
                           transform = transform,
                           train = False)

data_loader_train = torch.utils.data.DataLoader(dataset = data_train,
                                                batch_size = 64,
                                                shuffle = True,
                                                )

data_loader_test = torch.utils.data.DataLoader(dataset = data_test,
                                                batch_size = 64,
                                                shuffle = True,
                                                )

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(1, 64, kernel_size = 3, stride = 1, padding = 1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(stride = 2, kernel_size = 2)
                )
        self.dense = torch.nn.Sequential(
                torch.nn.Linear(14 * 14 * 128, 1024),
                torch.nn.ReLU(),
                torch.nn.Dropout(p = 0.5),
                torch.nn.Linear(1024, 10))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 14 * 14 * 128)
        x = self.dense(x)
        return x

model = Model()
cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
epochs = 5

for epoch in range(epochs):
    running_loss = 0.0
    running_correct = 0
    print("Epoch {}/{}".format(epoch, epochs))
    print("="*10)
    for data in data_loader_train:
        x_train, y_train = data
        x_train, y_train = Variable(x_train), Variable(y_train)
        outputs = model(x_train)
        _, pred = torch.max(outputs.data, 1)
        optimizer.zero_grad()
        loss = cost(outputs, y_train)

        loss.backward()
        optimizer.step()
        running_loss += loss.data.item()
        running_correct += torch.sum(pred == y_train.data)
    
    testing_correct = 0
    for data in data_loader_test:
        x_test, y_test = data
        x_test, y_test = Variable(x_test), Variable(y_test)
        outputs = model(x_test)
        _, pred = torch.max(outputs.data, 1)
        testing_correct += torch.sum(pred == y_test.data)

    print("Loss is: {:.4f}, Train Acc is: {:.4f}%, Test Acc is: {:.4f}%\n".format(
        running_loss / len(data_train),
        100 * running_correct / len(data_train),
        100 * testing_correct / len(data_test)
    ))

