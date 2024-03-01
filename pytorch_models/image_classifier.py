import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Bottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.relu(out)
        return out

class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels // 2, out_channels=in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, input):
        return input * self.attention(input)
    
class MultiScaleParallelConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleParallelConvBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            Attention(out_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            Attention(out_channels)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            Attention(out_channels)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            Attention(out_channels)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=4*out_channels, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, input):
        out1 = self.conv1(input)
        out2 = self.conv2(input)
        out3 = self.conv3(input)
        out4 = self.conv4(input)
        out = torch.cat((out1, out2, out3, out4), dim=1)
        out = self.conv5(out)
        return out

class Image_Classifier(nn.Module):
    input_size = 32
    conv1_out_channels = 64
    conv2_out_channels = 128
    conv3_out_channels = 128
    conv4_out_channels = 256
    output_size = 10
    def __init__(self):
        super(Image_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.conv1_out_channels, kernel_size=3, stride=2, padding=1)
        self.MultiScaleParallelConvBlock1 = MultiScaleParallelConvBlock(self.conv1_out_channels, self.conv1_out_channels)
        self.MultiScaleParallelConvBlock2 = MultiScaleParallelConvBlock(self.conv1_out_channels, self.conv2_out_channels)
        self.MultiScaleParallelConvBlock3 = MultiScaleParallelConvBlock(self.conv2_out_channels, self.conv3_out_channels)
        self.MultiScaleParallelConvBlock4 = MultiScaleParallelConvBlock(self.conv3_out_channels, self.conv4_out_channels)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.attention1 = Attention(self.conv4_out_channels)
        self.fc1 = nn.Linear(8*8*self.conv4_out_channels, 512)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, self.output_size)

    def forward(self, input):
        output = self.conv1(input)
        output = self.MultiScaleParallelConvBlock1(output)
        output = self.MultiScaleParallelConvBlock2(output)
        output = self.MultiScaleParallelConvBlock3(output)
        output = self.MultiScaleParallelConvBlock4(output)
        output = self.avgpool1(output)
        output = self.attention1(output)
        output = output.view(output.size(0), -1)
        output = self.dropout(output)
        output = self.fc1(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        return output

    
    def predict(self, input):
        input = input.to(device)
        output = self.forward(input)
        _, pred = torch.max(output, 1)
        return pred

    
    def predict_prob(self, input):
        input = input.to(device)
        output = self.forward(input)
        return output

    def load_model(model_path):
        model = Image_Classifier()
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        return model
    
    def save_model(model, model_path):
        torch.save(model.state_dict(), model_path)
        return model_path
    
    @staticmethod
    def load_data(batch_size, num_workers=2):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return trainloader, testloader
    
    def train_model(model, train_loader, test_loader, optimizer, criterion, epochs):
        for epoch in range(epochs):
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                if (i+1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, i+1, len(train_loader), loss.item()))
            correct = 0
            total = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
        return model
    
    def evaluate(model, test_loader):
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
        return correct, total
    
if __name__ == "__main__":
    # Hyper Parameters
    num_epochs = 20
    batch_size = 100
    learning_rate = 0.001
    model_path = './model/image_classifier.pth'
    # Load Data
    train_loader, test_loader = Image_Classifier.load_data(batch_size)
    # Create Model
    model = Image_Classifier()
    model.to(device)
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Train the Model
    model = Image_Classifier.train_model(model, train_loader, test_loader, optimizer, criterion, num_epochs)
    # Save the Model
    Image_Classifier.save_model(model, model_path)
    # Evaluate the Model
    Image_Classifier.evaluate(model, test_loader)
    # Load the Model
    model = Image_Classifier.load_model(model_path)
    model.eval()
    # Predict the Model
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    pred = model.predict(images)
    print(pred)
    # Predict the Probability
    pred_prob = model.predict_prob(images)
    print(pred_prob)