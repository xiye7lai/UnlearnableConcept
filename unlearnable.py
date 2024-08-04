import argparse
import json
import os
from pathlib import Path

import clip
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from torch import nn
from torch.distributions import Beta
from torch.nn import init
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2, resnet18, resnet50, shufflenet_v2_x1_0
from torchvision.models.vision_transformer import _vision_transformer
from torchvision.utils import save_image

from main import train_test
from data_utils import DataFolderWithLabel
from generator import Generator
from model.vgg import VGG16


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
    return model


# Mixup
class MixupTransform:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.beta_distribution = Beta(alpha, alpha)

    def __call__(self, x):
        lam = self.beta_distribution.sample().item()
        index = torch.randint(len(x), (1,)).item()
        mixed_x = lam * x + (1 - lam) * x[index]
        return mixed_x


# Gaussian Smoothing
def gaussian_smoothing(x, sigma=1):
    return transforms.functional.gaussian_blur(x, kernel_size=23, sigma=sigma)


def train(device):
    if config['method'] == 'clean':
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            # transforms.CenterCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
    elif config['method'] == 'uc' or config['method'] == 'our':
        train_transform = transforms.Compose([
            # transforms.Resize(256),
            # transforms.RandomCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            lambda x: gaussian_smoothing(x, sigma=1.0),
            # MixupTransform(alpha=1.0),
            # transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
    elif config['method'] == 'sn' or config['method'] == 'tue':
        train_transform = transforms.Compose([
            transforms.Resize([32, 32]),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])

        test_transform = transforms.Compose([
            transforms.Resize([32, 32]),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])

    num_classes = config['num_classes']

    new_classes = None

    if config['label-agnostic']:
        num_classes = config['target_classes_num']

        clip_v, _ = clip.load('ViT-B/32', 'cpu')
        if config['dataset'] == 'flower':
            with open('../data/datasets/classification/kaggle-flower/cat_to_name.json', 'r') as f:
                cat_to_name = json.load(f)
            num = []
            q = []
            for key, value in cat_to_name.items():
                num.append(int(key))
                q.append(value)
            text_inputs = torch.cat([clip.tokenize("a photo of a " + c) for c in q])
            with torch.no_grad():
                text_features = clip_v.encode_text(text_inputs)
            classifier = KMeans(n_clusters=config['target_classes_num'], random_state=seed)
            pred_idx = classifier.fit_predict(text_features)
            new_classes = {num[i]: pred_idx[i] for i in range(len(q))}
        else:
            if config['dataset'] == 'sun':
                data = datasets.SUN397(root='../data/datasets/classification')
            elif config['dataset'] == 'car':
                data = datasets.StanfordCars(root='../data/datasets/classification')
            elif config['dataset'] == 'food':
                data = datasets.Food101(root='../data/datasets/classification')
            q_v = data.class_to_idx
            q = data.classes
            text_inputs = torch.cat([clip.tokenize("a photo of a " + c) for c in q])
            with torch.no_grad():
                text_features = clip_v.encode_text(text_inputs)
            classifier = KMeans(n_clusters=config['target_classes_num'], random_state=seed)
            pred_idx = classifier.fit_predict(text_features)
            new_classes = {q_v[q[i]]: int(pred_idx[i]) for i in range(len(q))}

    if config['label-agnostic']:
        target_transform = lambda label: new_classes[label]
    else:
        target_transform = None

    if config['method'] == 'clean':
        if config['dataset'] == 'flower':
            train_dataset = DataFolderWithLabel('../data/datasets/classification/kaggle-flower/train',
                                                transform=train_transform, new_classes=new_classes)
        if config['dataset'] == 'pet':
            train_dataset = DataFolderWithLabel('../data/datasets/classification/pets/train',
                                                transform=train_transform, new_classes=new_classes)
        if config['dataset'] == 'food':
            train_dataset = datasets.Food101('../data/datasets/classification', split='train',
                                             transform=train_transform, target_transform=target_transform)
        if config['dataset'] == 'car':
            train_dataset = datasets.StanfordCars('../data/datasets/classification', split='train',
                                                  transform=train_transform, target_transform=target_transform)
        if config['dataset'] == 'sun':
            train_dataset = datasets.SUN397('../data/datasets/classification', transform=train_transform,
                                            target_transform=target_transform)
            train_indices, test_indices = train_test_split(list(range(len(train_dataset))), test_size=0.2,
                                                           random_state=seed)
            train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    else:
        train_dataset = DataFolderWithLabel(root=f"outputs/{config['method']}_datasets/{config['dataset']}",
                                            transform=train_transform, new_classes=new_classes)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config['batch'], num_workers=8)

    if config['dataset'] == 'flower':
        test_dataset = DataFolderWithLabel('../data/datasets/classification/kaggle-flower/val',
                                           transform=test_transform, new_classes=new_classes)
    if config['dataset'] == 'pet':
        test_dataset = DataFolderWithLabel('../data/datasets/classification/pets/test',
                                           transform=test_transform, new_classes=new_classes)
    if config['dataset'] == 'food':
        test_dataset = datasets.Food101('../data/datasets/classification', split='test',
                                        transform=test_transform, target_transform=target_transform)
    if config['dataset'] == 'car':
        test_dataset = datasets.StanfordCars('../data/datasets/classification', split='test',
                                             transform=test_transform, target_transform=target_transform)
    if config['dataset'] == 'sun':
        test_dataset = datasets.SUN397('../data/datasets/classification', transform=test_transform,
                                       target_transform=target_transform)
        train_indices, test_indices = train_test_split(list(range(len(test_dataset))), test_size=0.2,
                                                       random_state=seed)
        test_dataset = torch.utils.data.Subset(test_dataset, test_indices)

    test_loader = DataLoader(test_dataset, batch_size=config['batch'], num_workers=8)

    if config['dataset'] == 'flower' and config['label-agnostic'] is False:
        num_classes = num_classes + 1

    # resnet,mobilenet,shufflenet
    if config['model'] == 'resnet':
        net = resnet18(num_classes=num_classes).to(device)
    if config['model'] == 'mobilenet':
        net = mobilenet_v2(num_classes=num_classes).to(device)
    if config['model'] == 'shufflenet':
        net = shufflenet_v2_x1_0(num_classes=num_classes).to(device)
    if config['model'] == 'vgg':
        # net = vgg16(num_classes=num_classes).to(device)
        net = VGG16(num_classes=num_classes).to(device)
    if config['model'] == 'vit':
        net = _vision_transformer(
            arch="vit_b_16",
            image_size=config['input_size'],
            patch_size=16,
            num_layers=4,
            num_heads=8,
            hidden_dim=256,
            mlp_dim=256,
            pretrained=False,
            progress=True,
            num_classes=num_classes,
            dropout=0.1
        ).to(device)
    if config['model'] == 'rn50':
        net = resnet50(num_classes=num_classes).to(device)

    train_test(net, train_loader, config['epoch'], config['lr'], test_loader, config['batch'],
               upload=config['upload'])


def generate():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])

    clip_v, preprocess = clip.load('ViT-B/32', 'cuda')

    num_classes = config['num_classes']

    if config['dataset'] == 'flower':
        num_classes = num_classes + 1
        train_dataset = DataFolderWithLabel('../data/datasets/classification/kaggle-flower/train',
                                            transform=transform)
    if config['dataset'] == 'pet':
        train_dataset = DataFolderWithLabel('../data/datasets/classification/pets/train',
                                            transform=transform)
    if config['dataset'] == 'food':
        train_dataset = datasets.Food101('../data/datasets/classification', split='train',
                                         transform=transform)
    if config['dataset'] == 'car':
        train_dataset = datasets.StanfordCars('../data/datasets/classification', split='train',
                                              transform=transform)
    if config['dataset'] == 'sun':
        train_dataset = datasets.SUN397('../data/datasets/classification', transform=transform)
        train_indices, test_indices = train_test_split(list(range(len(train_dataset))), test_size=0.2,
                                                       random_state=seed)
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)

    if config['dataset'] == 'cifar10':
        train_dataset = datasets.CIFAR10('../data/datasets/classification', train=True,
                                         transform=transform)

    if config['dataset'] == 'cifar100':
        train_dataset = datasets.CIFAR100('../data/datasets/classification', train=True,
                                          transform=transform)

    g_net = Generator(ex_dim=512)
    g_net.to('cuda')
    # g_net = torch.nn.DataParallel(g_net, device_ids=[0, 1])

    state_dict = \
        torch.load('outputs/imagenet_image_emb_512/g_net.pt', map_location=torch.device('cuda'))[
            'state_dict']

    g_net.load_state_dict(state_dict)
    g_net.eval()

    print(len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=8)

    count = [0 for _ in range(num_classes)]
    output_dir = f"outputs/our_datasets/{config['dataset']}"
    print(output_dir)
    for i in range(len(count)):
        Path(os.path.join(output_dir, str(i))).mkdir(parents=True, exist_ok=True)

    count = [0 for _ in range(num_classes)]
    for i in range(len(count)):
        Path(os.path.join(output_dir, str(i))).mkdir(parents=True, exist_ok=True)

    for images, ground_truth in train_loader:
        images, ground_truth = images.to('cuda'), ground_truth.to('cuda')
        with torch.no_grad():

            image_features = clip_v.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            delta_im = g_net(images, image_features)
            temp = torch.norm(delta_im.view(delta_im.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            delta_im = delta_im * 16 / temp
            images = torch.clamp(images + delta_im, 0, 1)

        ground_truth = ground_truth.tolist()

        for i in range(len(images)):
            gt = ground_truth[i]
            save_image(images[i], os.path.join(output_dir, str(gt), f'{count[gt]}.png'))
            count[gt] += 1


if __name__ == '__main__':
    seed = 35
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--stage', type=str, default='train')
    parser.add_argument('--dataset', type=str, default='pet')
    parser.add_argument('--method', type=str, default='our')
    parser.add_argument('--model', type=str, default='resnet')
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument("--set_batch", action='store_true')
    parser.add_argument('--batch', type=int, default=1000)
    parser.add_argument("--la", action='store_true')

    args = parser.parse_args()

    class_match = {'pet': 37, 'flower': 102, 'car': 196, 'food': 101, 'sun': 397, 'cifar10': 10, 'cifar100': 100}
    batch_match = {'pet': 64, 'flower': 64, 'car': 64, 'food': 256, 'sun': 256, 'cifar10': 256, 'cifar100': 256}
    config = {
        'stage': args.stage,  # generate,train
        'dataset': args.dataset,
        'method': args.method,  # sn,uc,tue,our,clean
        'model': args.model,  # resnet,mobilenet,shufflenet,vgg,vit,rn50
        'device': args.device,  # 0,1
        'label-agnostic': args.la,
        'input_size': args.input_size,
        'num_cluster': 10,
        'epoch': 200,
        'lr': 0.1,
        'upload': False,
        'g_epoch': 200,
        'target_classes_num': 20,
    }
    config['num_classes'] = class_match[config['dataset']]

    if args.set_batch:
        config['batch'] = args.batch
    else:
        config['batch'] = batch_match[config['dataset']]

    torch.cuda.set_device(config['device'])

    if config['stage'] == 'train':
        train('cuda')
    elif config['stage'] == 'generate':
        generate()
