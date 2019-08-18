import numpy as np
import torch
from torch.utils.data import sampler
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn as nn


NUM_TRAIN = 59000

transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485],std=[0.229])
                            ])

train_data = datasets.FashionMNIST('F_MNIST_data/', download=True, train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, 
                                         sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

fashionMnist_val = datasets.FashionMNIST('F_MNIST_data/', train=True, download=True, transform=transform)
val_loader = torch.utils.data.DataLoader(fashionMnist_val, batch_size=64, 
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 60000)))

# Download and load the test data
test_data = datasets.FashionMNIST('F_MNIST_data/', download=True, train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)


labels_map = {0 : 'T-Shirt', 1 : 'Trouser', 2 : 'Pullover', 3 : 'Dress', 4 : 'Coat', 5 : 'Sandal', 6 : 'Shirt',
              7 : 'Sneaker', 8 : 'Bag', 9 : 'Ankle Boot'};
              
fig = plt.figure(figsize=(8,8));
columns = 4;
rows = 5;
for i in range(1, columns*rows +1):
    img_xy = np.random.randint(len(train_data));
    img = train_data[img_xy][0][0,:,:]
    fig.add_subplot(rows, columns, i)
    plt.title(labels_map[train_data[img_xy][1]])
    plt.axis('off')
    plt.imshow(img, cmap='gray')
plt.show()

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride =1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size = 2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size = 2, stride=2))
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
    
num_epochs = 15
model = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))

total_step = len(train_loader)
train_losses, val_losses, train_accs, val_accs = [],[],[],[]
for epoch in range(num_epochs):
    train_loss = 0
    val_loss = 0
    train_acc = 0 
    val_acc = 0
    for images,labels in train_loader:

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        train_acc += correct / total
    else:
        with torch.no_grad():
            model.eval()
            for images,labels in val_loader:
                val_out = model(images)
                loss = criterion(val_out, labels)
                val_loss += loss.item()
                
                total = labels.size(0)
                _, predicted = torch.max(val_out.data, 1)
                correct = (predicted == labels).sum().item()
                val_acc += correct / total
        model.train()
        
    print("Epoch: {}/{}.. ".format(epoch, num_epochs),
              "Training Loss: {:.3f}.. ".format(train_loss/len(train_loader)),
              "Validation Loss: {:.3f}.. ".format(val_loss/len(val_loader)),
              "Validation Accuracy: {:.3f}".format(val_acc/len(val_loader)*100))
    
    train_losses.append(train_loss/len(train_loader))
    val_losses.append(val_loss/len(val_loader))
    train_accs.append(train_acc/len(train_loader)*100)
    val_accs.append(val_acc/len(val_loader)*100)
        
plt.figure()           
plt.plot(train_losses,label = "Train losses")
plt.plot(val_losses, label = "Validation losses")
plt.legend()
plt.show()

plt.figure()
plt.plot(train_accs,label = "Train accuracy")
plt.plot(val_accs, label = "Validation accuracy")
plt.legend() 
plt.show()   
            
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
