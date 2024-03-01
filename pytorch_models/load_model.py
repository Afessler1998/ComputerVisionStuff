import torch
import torchvision

from image_classifier import Image_Classifier

if __name__ == "__main__":
    model = Image_Classifier.load_model("model/image_classifier.pth")
    model.eval()
    _, test_loader = Image_Classifier.load_data(1)
    count = 0
    for data, target in test_loader:
        pred = model.predict(data)
        print("Predicted: ", pred.item())
        print("Actual: ", target.item())
        count += 1
        if count == 10:
            break