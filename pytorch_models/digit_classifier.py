import torch
import torch.nn as nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Digit_Classifier(nn.Module):
    input_size = 28
    def __init__(self):
        super(Digit_Classifier, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(7*7*32, 10)

    def forward(self, input):
        output = self.conv1(input)
        output = self.pool1(output)
        output = self.conv2(output)
        output = self.pool2(output)
        output = output.view(output.size(0), -1)
        output = self.dropout(output)
        output = self.fc1(output)
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
        model = Digit_Classifier()
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        return model
    
    def save_model(model, model_path):
        torch.save(model.state_dict(), model_path)
        return model_path
    
    def load_data(batch_size):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader
    
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
    num_epochs = 5
    batch_size = 100
    learning_rate = 0.001
    model_path = './model/digit_classifier.pth'
    # Load Data
    train_loader, test_loader = Digit_Classifier.load_data(batch_size)
    # Create Model
    model = Digit_Classifier()
    model.to(device)
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Train the Model
    model = Digit_Classifier.train_model(model, train_loader, test_loader, optimizer, criterion, num_epochs)
    # Save the Model
    Digit_Classifier.save_model(model, model_path)
    # Evaluate the Model
    Digit_Classifier.evaluate(model, test_loader)
    # Load the Model
    model = Digit_Classifier.load_model(model_path)
    model.eval()
    # Predict the Model
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    pred = model.predict(images)
    print(pred)
    # Predict the Probability
    pred_prob = model.predict_prob(images)
    print(pred_prob)