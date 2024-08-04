import json
import os

import clip
import torch
from PIL import Image
from torch.utils.data import Dataset


def is_image_file(filename):
    IMG_EXTENSIONS = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
    ]
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class DataFolderWithLabel(Dataset):
    def __init__(self, root, transform=None, new_classes=None):
        self.labels = []
        self.images = []
        self.transform = transform

        for class_name in sorted(os.listdir(root)):
            label = int(class_name)
            if new_classes is not None:
                label = int(new_classes[label])
            for file_name in sorted(os.listdir(os.path.join(root, class_name))):
                if not is_image_file(file_name):
                    continue
                self.images.append(os.path.join(root, class_name, file_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        return image, label


class ImagenetDataset(Dataset):
    def __init__(self, root, transform=None):
        self.labels = []
        self.images = []
        self.text_targets = []
        self.transform = transform

        file_path = '../data/datasets/classification/imagenets/image.json'  # file path
        with open(file_path, 'r') as file:
            meta = json.load(file)

        for i in range(len(meta)):
            label = int(i)
            # label = temp[i]
            # self.text_targets.append(class_name.replace('_', ' '))
            self.text_targets.append(' '.join(word for word in meta[str(i)][1].replace('_', ' ').split()))
            for file_name in sorted(os.listdir(os.path.join(root, meta[str(i)][0]))):
                if not is_image_file(file_name):
                    continue
                self.images.append(os.path.join(root, meta[str(i)][0], file_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        return image, label


if __name__ == '__main__':
    clip_v, preprocess = clip.load('ViT-B/32', 'cpu')
    train_dataset = ImagenetDataset(root='../data/datasets/classification/imagenets/train', transform=preprocess)
    text_targets = train_dataset.text_targets
    text_inputs = torch.cat([clip.tokenize("a photo of a " + c) for c in text_targets])
    with torch.no_grad():
        text_features = clip_v.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    top_k_result = ()
    results = text_features @ text_features.T
    min_indices = torch.topk(results, k=1, dim=1,largest=False).indices

    # print(min_indices)
    # min_indices = min_indices[:,1:]
    print(min_indices)
    torch.save(min_indices, 'outputs/far_indices.pt')
