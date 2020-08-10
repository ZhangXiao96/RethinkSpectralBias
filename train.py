from lib.ModelWrapper import ModelWrapper
from tensorboardX import SummaryWriter
import torch
from torchvision import transforms, datasets
import numpy as np
import random
import sys
import os

args = sys.argv
data_name = args[1]     # 'svhn', 'cifar10', 'cifar100'
data_root = args[2]
model_name = args[3]    # 'resnet18', 'resnet34', 'vgg16', 'vgg13', 'vgg11'

# setting
lr = 1e-4
train_batch_size = 128
train_epoch = 1000
eval_batch_size = 250
label_noise = 0.10
delta_h = 0.5
nb_interpolation = 128

if data_name == 'cifar10':
    dataset = datasets.CIFAR10
    from archs.cifar10 import vgg, resnet
elif data_name == 'cifar100':
    dataset = datasets.CIFAR100
    from archs.cifar100 import vgg, resnet
elif data_name == 'svhn':
    dataset = datasets.SVHN
    from archs.svhn import vgg, resnet
else:
    raise Exception('No such dataset')

if model_name == 'vgg11':
    model = vgg.vgg11_bn()
elif model_name == 'vgg13':
    model = vgg.vgg13_bn()
elif model_name == 'vgg16':
    model = vgg.vgg16_bn()
elif model_name == 'resnet18':
    model = resnet.resnet18()
elif model_name == 'resnet34':
    model = resnet.resnet34()
else:
    raise Exception("No such model!")

train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor()])
eval_transform = transforms.Compose([transforms.ToTensor()])

# load data
if 'cifar' in data_name:
    train_data = dataset(data_root, train=True, transform=train_transform, download=True)
    train_targets = np.array(train_data.targets)
    data_size = len(train_targets)
    random_index = random.sample(range(data_size), int(data_size*label_noise))
    random_part = train_targets[random_index]
    np.random.shuffle(random_part)
    train_targets[random_index] = random_part
    train_data.targets = train_targets.tolist()

    noise_data = dataset(data_root, train=False, transform=eval_transform, download=True)
    noise_data.targets = random_part.tolist()
    noise_data.data = train_data.data[random_index]

    test_data = dataset(data_root, train=False, transform=eval_transform)

elif 'svhn' in data_name:
    train_data = dataset(data_root, split='train', transform=train_transform, download=True)
    train_targets = np.array(train_data.labels)
    data_size = len(train_targets)
    random_index = random.sample(range(data_size), int(data_size * label_noise))
    random_part = train_targets[random_index]
    np.random.shuffle(random_part)
    train_targets[random_index] = random_part
    train_data.labels = train_targets.tolist()

    noise_data = dataset(data_root, split='test', transform=eval_transform, download=True)
    noise_data.labels = random_part.tolist()
    noise_data.data = train_data.data[random_index]

    test_data = dataset(data_root, split='test', transform=eval_transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=0,
                                           drop_last=False)
noise_loader = torch.utils.data.DataLoader(noise_data, batch_size=eval_batch_size, shuffle=True, num_workers=0,
                                           drop_last=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=eval_batch_size, shuffle=True, num_workers=0,
                                          drop_last=False)

# build model
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
wrapper = ModelWrapper(model, optimizer, criterion, device)

# train the model
save_path = os.path.join('runs', data_name, "{}".format(model_name))
if not os.path.exists(save_path):
    os.makedirs(save_path)
np.savez(os.path.join(save_path, "label_noise.npz"), index=random_index, value=random_part)
writer = SummaryWriter(log_dir=os.path.join(save_path, "log"), flush_secs=30)

itr_index = 1
wrapper.train()

for id_epoch in range(train_epoch):
    # train loop
    train_loss = 0
    train_acc = 0
    train_size = 0
    for id_batch, (inputs, targets) in enumerate(train_loader):
        loss, acc, correct = wrapper.train_on_batch(inputs, targets)
        train_loss += loss
        train_acc += correct
        train_size += len(targets)
        print("epoch:{}/{}, batch:{}/{}, loss={}, acc={}".
              format(id_epoch+1, train_epoch, id_batch+1, len(train_loader), loss, acc))
        itr_index += 1
    train_loss /= id_batch
    train_acc /= train_size
    writer.add_scalar("train acc", train_acc, itr_index)
    writer.add_scalar("train loss", train_loss, itr_index)

    # eval
    wrapper.eval()
    test_loss, test_acc = wrapper.eval_all(test_loader)
    noise_loss, noise_acc = wrapper.eval_all(noise_loader)
    print("epoch:{}/{}, batch:{}/{}, testing...".format(id_epoch + 1, train_epoch, id_batch + 1, len(train_loader)))
    print("clean: loss={}, acc={}".format(test_loss, test_acc))
    print("noise: loss={}, acc={}".format(noise_loss, noise_acc))
    writer.add_scalar("test acc", test_acc, itr_index)
    writer.add_scalar("test loss", test_loss, itr_index)
    writer.add_scalar("noise acc", noise_acc, itr_index)
    writer.add_scalar("noise loss", noise_loss, itr_index)
    state = {
        'net': model.state_dict(),
        'optim': optimizer.state_dict(),
        'acc': test_acc,
        'epoch': id_epoch,
        'itr': itr_index
    }
    torch.save(state, os.path.join(save_path, "ckpt.pkl"))

    if id_epoch % 1 == 0:
        test_energy = wrapper.predict_line_fft(test_loader, delta_h, nb_interpolation)
        avg_test_energy = np.mean(test_energy[:500], axis=(0, 1))
        writer.add_scalars("test energy", {"{}".format(i): _ for i, _ in enumerate(avg_test_energy)}, itr_index)

        pert_energy = wrapper.predict_line_fft(noise_loader, delta_h, nb_interpolation)
        avg_pert_energy = np.mean(pert_energy[:500], axis=(0, 1))
        writer.add_scalars("pert energy", {"{}".format(i): _ for i, _ in enumerate(avg_pert_energy)}, itr_index)

        train_energy = wrapper.predict_line_fft(train_loader, delta_h, nb_interpolation)
        avg_train_energy = np.mean(train_energy[:500], axis=(0, 1))
        writer.add_scalars("train energy", {"{}".format(i): _ for i, _ in enumerate(avg_train_energy)}, itr_index)
    print()
    # return to train state.
    wrapper.train()
writer.close()
