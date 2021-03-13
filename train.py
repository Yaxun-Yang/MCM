import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import copy
import matplotlib.pyplot as plt
import numpy as np


def show_batch_images(sample_batch):
    labels_batch = sample_batch[1]
    images_batch = sample_batch[0]

    for i in range(4):
        label_ = labels_batch[i].item()
        image_ = np.transpose(images_batch[i], (1, 2, 0))
        ax = plt.subplot(1, 4, i + 1)
        ax.imshow(image_)
        ax.set_title(str(label_))
        ax.axis('off')
        plt.pause(0.01)


def main():
    data_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    train_dataset = datasets.ImageFolder(root="dataset\\train", transform=data_transform)
    train_data_loader = DataLoader(dataset=train_dataset,
                                   batch_size=4,
                                   shuffle=True,
                                   num_workers=4)

    val_dataset = datasets.ImageFolder(root='dataset\\test', transform=data_transform)
    val_data_loader = DataLoader(dataset=val_dataset,
                                 batch_size=4,
                                 shuffle=True,
                                 num_workers=4)

    model = models.squeezenet1_1(pretrained=True)
    model.classifier[1] = nn.Conv2d(in_channels=512, out_channels=2, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = 2
    # print(model)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.000045, momentum=0.9)
    loss_fc = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, 0.1)

    num_epoch = 1
    logfile_dir = "log/"

    acc_best_wts = model.state_dict()
    best_acc = 0

    for epoch in range(num_epoch):
        train_correct = 0
        train_total = 0

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        for i, sample_batch in enumerate(train_data_loader):
            inputs = sample_batch[0].to(device)
            labels = sample_batch[1].to(device)

            # 模型设置为train
            model.train()

            # forward
            outputs = model(inputs)

            # print(labels)
            # loss
            loss = loss_fc(outputs, labels)

            # forward update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            train_loss = loss.item()
            train_correct += (torch.max(outputs, 1)[1] == labels).sum().item()
            train_total += labels.size(0)

            # print('iter:{}'.format(i))

            if i % 10 == 9:
                for var_batch in val_data_loader:
                    inputs = var_batch[0].to(device)
                    labels = var_batch[1].to(device)

                    model.eval()
                    outputs = model(inputs)
                    loss = loss_fc(outputs, labels)
                    _, prediction = torch.max(outputs, 1)
                    val_correct += ((labels == prediction).sum()).item()
                    val_total += inputs.size(0)
                    val_loss = loss.item()

                val_acc = val_correct / val_total
                print('[{},{}] train_loss = {:.5f} train_acc = {:.5f} val_loss = {:.5f} val_acc = {:.5f}'.format(
                    epoch + 1, i + 1, train_loss, train_correct / train_total, val_loss,
                    val_correct / val_total))
                if val_acc > best_acc:
                    best_acc = val_acc
                    acc_best_wts = copy.deepcopy(model.state_dict())

                with open(logfile_dir + 'train_acc_4.txt', 'a') as f:
                    f.write(str(train_correct / train_total) + '\n')
                with open(logfile_dir + 'val_acc_4.txt', 'a') as f:
                    f.write(str(val_correct / val_total) + '\n')
        scheduler.step()

    torch.save(acc_best_wts, './models/model_4.pth')


def draw():
    train_acc_list = np.loadtxt('log/train_acc_4.txt', dtype=float)
    val_acc_list = np.loadtxt('log/val_acc_4.txt', dtype=float)
    l1, = plt.plot(range(len(train_acc_list)), train_acc_list, 'r-')
    l2, = plt.plot(range(len(val_acc_list)), val_acc_list, 'g--')
    plt.legend([l1, l2], ['train_accuracy', 'validation_accuracy'])
    plt.show()


def test():
    model = models.squeezenet1_1(pretrained=True)
    model.classifier[1] = nn.Conv2d(in_channels=512, out_channels=2, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = 2
    model.load_state_dict(torch.load('./models/model_4.pth', map_location='cpu'))
    model.eval()

    data_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_dataset = datasets.ImageFolder(root="dataset\\unverified", transform=data_transform)
    test_data_loader = DataLoader(dataset=test_dataset,
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=4)

    plt.figure(figsize=(7, 10))
    for i, sample in enumerate(test_data_loader):
        plt.subplot(7, 10, i+1)
        plt.axis("off")

        inputs, labels = sample[0], sample[1]
        outputs = model(inputs)
        _, prediction = torch.max(outputs, 1)
        likehood = outputs[0][prediction] / (outputs[0][prediction] + outputs[0][1-prediction])
        plt.imshow(np.transpose(inputs[0], (1, 2, 0)))
        if prediction == 0:
            plt.title("{:.2f}N ".format(float(likehood)))
        else:
            plt.title("{:.2f}Y ".format(float(likehood)))
    plt.suptitle("Unverified and Unprocessed Picture Judge")
    plt.show()


if __name__ == '__main__':
    test()
