import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from tqdm import tqdm

from train_gnet import train_gnet_total

DEVICE = 'cuda'
DEVICE_IDS = [0, 1]
seed = 35


def train_test(model, loader, epoch, lr, clean_test_loader, bs, upload=False):
    params = {
        "lr": lr,
        "bs": bs,
        "epochs": epoch,
        "epsilon": 16,
    }
    cr = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=3e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=1e-6)
    for ep in range(epoch):

        model.train()
        running_loss, correct, samples = 0.0, 0, 0
        if not upload:
            item = tqdm(loader)
            item.set_description(f'Epoch [{ep + 1}/{epoch}]')
            for x, y in item:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                loss = cr(logits, y)

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                running_loss += loss.item() * y.shape[0]
                _, predicted = torch.max(logits, 1)
                correct += (predicted == y).sum().item()
                samples += y.shape[0]
                item.set_postfix({'loss': running_loss / samples, 'acc': correct / samples})
        else:
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                loss = cr(logits, y)

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                running_loss += loss.item() * y.shape[0]
                _, predicted = torch.max(logits, 1)
                correct += (predicted == y).sum().item()
                samples += y.shape[0]

        scheduler.step()

        print('Test acc: ' + str(test_model(model, clean_test_loader)))


def test_model(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def test_model_class(model, test_loader):
    model.eval()
    # prepare to count predictions for each class
    classes = ["apple", "orange", "pear", "pineapple", "lemon", "strawberry", "cherry", "mushroom", "carrot",
               "broccoli"]
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        if correct_count == 0:
            accuracy = 0
        else:
            accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                             accuracy))

    correct = 0

    total = 0

    with torch.no_grad():
        item = tqdm(test_loader)
        for images, labels in item:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            item.set_postfix({'acc': correct / total})

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


if __name__ == '__main__':
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    train_gnet_total(epochs=200, lr=1e-3, epsilon=16)
