from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import csv
import os

class VGG11(nn.Module):
    """VGG11 for CIFAR-10. Dropout is applied before ReLU as requested."""
    def __init__(self, num_classes=10, dropout_p=0.0):
        super().__init__()
        self.layers = self._make_layers(dropout_p)

    def _make_layers(self, dropout_p):
        layers = []

        def conv(in_c, out_c):
            return nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)

        layers += [conv(3, 64), nn.Dropout(p=dropout_p), nn.ReLU(inplace=True),
                   nn.MaxPool2d(kernel_size=2, stride=2)]

        layers += [conv(64, 128), nn.Dropout(p=dropout_p), nn.ReLU(inplace=True),
                   nn.MaxPool2d(kernel_size=2, stride=2)]

        layers += [conv(128, 256), nn.Dropout(p=dropout_p), nn.ReLU(inplace=True),
                   conv(256, 256), nn.Dropout(p=dropout_p), nn.ReLU(inplace=True),
                   nn.MaxPool2d(kernel_size=2, stride=2)]

        layers += [conv(256, 512), nn.Dropout(p=dropout_p), nn.ReLU(inplace=True),
                   conv(512, 512), nn.Dropout(p=dropout_p), nn.ReLU(inplace=True),
                   nn.MaxPool2d(kernel_size=2, stride=2)]

        layers += [conv(512, 512), nn.Dropout(p=dropout_p), nn.ReLU(inplace=True),
                   conv(512, 512), nn.Dropout(p=dropout_p), nn.ReLU(inplace=True),
                   nn.MaxPool2d(kernel_size=2, stride=2)]

        layers += [nn.Flatten(),
                   nn.Linear(512 * 1 * 1, 4096), nn.Dropout(p=dropout_p), nn.ReLU(inplace=True),
                   nn.Linear(4096, 4096), nn.Dropout(p=dropout_p), nn.ReLU(inplace=True),
                   nn.Linear(4096, 10)]

        return nn.ModuleList(layers)

    def forward(self, x):
        for mod in self.layers:
            x = mod(x)
        return x

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    count = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)
        count += data.size(0)
        if batch_idx % args.log_interval == 0:
            print('Current time: {:.4f}; Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                time.time(),
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    avg_loss = running_loss / count if count > 0 else 0.0
    return avg_loss

def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)

    test_loss /= total if total > 0 else 1.0
    acc = 100.0 * correct / total if total > 0 else 0.0

    print('Current time: {:.4f}; Test Epoch: {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        time.time(),
        epoch,
        test_loss, correct, total,
        acc))
    return test_loss, acc

def main():
    parser = argparse.ArgumentParser(description='PyTorch SVHN Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for testing (default: 1024)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--dropout_p', type=float, default=0.0,
                        help='dropout_p (default: 0.0)')
    parser.add_argument('--L2_reg', type=float, default=None,
                        help='L2_reg (default: None)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--data-path', type=str, default='../data', dest='data_path',
                        help='path where CIFAR-10 will be downloaded/stored (default: ../data)')
    parser.add_argument('--output-dir', type=str, default='./results', dest='output_dir',
                        help='directory to save CSV and model (default: ./results)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='save trained model after training')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': args.test_batch_size, 'shuffle': False}
    if use_cuda:
        cuda_kwargs = {'num_workers': 4, 'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2470, 0.2435, 0.2616)

    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])

    dataset_train = datasets.CIFAR10(root=args.data_path, train=True, download=True,
                                     transform=train_transforms)
    dataset_test = datasets.CIFAR10(root=args.data_path, train=False, download=True,
                                    transform=test_transforms)
    train_loader = torch.utils.data.DataLoader(dataset_train, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)

    model = VGG11(num_classes=10, dropout_p=args.dropout_p).to(device)

    weight_decay = float(args.L2_reg) if args.L2_reg is not None else 0.0
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=weight_decay)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_name = os.path.join(args.output_dir, f'vgg11_dropout{args.dropout_p}_L2{weight_decay}.csv')
    with open(csv_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch', 'train_loss', 'test_loss', 'test_accuracy', 'epoch_time_sec'])

    print(f'Starting training at: {time.time():.4f}; writing results to {csv_name}')
    for epoch in range(1, args.epochs + 1):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        train_loss = train(args, model, device, train_loader, optimizer, epoch)
        test_loss, test_acc = test(model, device, test_loader, epoch)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.time()
        epoch_time = t1 - t0

        with open(csv_name, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch, train_loss, test_loss, test_acc, epoch_time])

    if args.save_model:
        f_name = os.path.join(args.output_dir, f'trained_VGG11_dropout{args.dropout_p}_L2{weight_decay}.pt')
        torch.save(model.state_dict(), f_name)
        print(f'Saved model to: {f_name}')


if __name__ == '__main__':
    main()
