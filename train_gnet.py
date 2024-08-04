import clip
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data_utils import ImagenetDataset
from generator import Generator

DEVICE = 'cuda'


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, images, labels, text, text_features, results, far_results, clip, top_k):
        far_results = far_results.squeeze()
        image_features = clip.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        selected_features = text_features[far_results[labels]]
        true_similarity = torch.mean(image_features @ selected_features.T)

        results = results[:, 0:top_k]
        target_results = results[labels]
        target_selected_features = text_features[target_results[:, :top_k]]

        total_loss = 0.1 * torch.reciprocal(true_similarity)
        for i in range(top_k):
            total_loss = total_loss + torch.mean(image_features @ target_selected_features[:, i].T) / top_k
        return total_loss


def train_gnet_total(epochs, lr, epsilon):
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    params = {
        "lr": lr,
        "bs": 30,
        "epochs": epochs,
        "epsilon": epsilon,
        "type": 'emb+image',
        "g_size": '512+512*28*28'
    }

    model, preprocess = clip.load('ViT-B/32', DEVICE)

    train_dataset = ImagenetDataset(root='../data/datasets/classification/imagenets/train', transform=train_transform)
    text_targets = train_dataset.text_targets
    # text_inputs = torch.cat([clip.tokenize("a photo of a " + c + ", a type of pet") for c in text_targets])
    text_inputs = torch.cat([clip.tokenize("a photo of a " + c) for c in text_targets])
    text_inputs = text_inputs.to(DEVICE)

    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        # text_features = text_features.T

    train_loader = DataLoader(dataset=train_dataset, batch_size=params["bs"], shuffle=True, drop_last=False,
                              num_workers=4)

    g_net = Generator(ex_dim=512)
    g_net.to(DEVICE)
    # g_net = torch.nn.DataParallel(g_net, device_ids=[0, 1])

    # state_dict = \
    #     torch.load('outputs/imagenet_image_emb_512/generator.pt', map_location=torch.device('cuda'))[
    #         'state_dict']
    # g_net.load_state_dict(state_dict)

    # noise = torch.zeros((1, 3, 224, 224))
    # noise.uniform_(0, 1)
    # noise = noise.to(DEVICE)

    optimizer = torch.optim.Adam(g_net.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = CustomLoss()

    local_results = torch.load('outputs/indices.pt')
    far_results = torch.load('outputs/far_indices.pt')

    for epoch in range(epochs):
        g_net.train()

        total = tqdm(train_loader)

        running_loss = 0.0
        samples = 0

        for images, targets in train_loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            with torch.no_grad():
                images_features = model.encode_image(images)
                images_features /= images_features.norm(dim=-1, keepdim=True)
                # emb_noise = noise.repeat(images.shape[0], 1, 1, 1)

            delta_im = g_net(images, images_features.to(torch.float))
            temp = torch.norm(delta_im.view(delta_im.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            delta_im = delta_im * epsilon / temp
            images_adv = torch.clamp(images + delta_im, 0, 1)

            # text_inputs = text_inputs.to(DEVICE)
            # image_features /= image_features.norm(dim=-1, keepdim=True)
            # text_features /= text_features.norm(dim=-1, keepdim=True)
            model.eval()
            loss = criterion(images_adv, targets, text_inputs, text_features, local_results, far_results, model,
                             top_k=3)

            running_loss += loss.item() * targets.shape[0]
            samples += targets.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        total.set_description(f'Epoch [{epoch}/{epochs}]')
        total.set_postfix(loss=running_loss / samples)

        torch.save({'state_dict': g_net.state_dict()},
                   'outputs/emb_512/g_net.pt')
