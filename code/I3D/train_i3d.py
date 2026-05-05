import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
import videotransforms
import numpy as np

from configs import Config
from pytorch_i3d import InceptionI3d
from datasets.nslt_dataset import NSLT as Dataset

from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, default='rgb')
parser.add_argument('-save_model', type=str, default='checkpoints/')
parser.add_argument('-root', type=str, default='../../data/')
parser.add_argument('--num_class', type=int)

args = parser.parse_args()

torch.manual_seed(0)
np.random.seed(0)

def run(configs,
        mode='rgb',
        root = {'word': '../../data/'},
        train_split='preprocess/nslt_100.json',
        save_model='checkpoints/',
        weights=None):

    print("Device:", device)

    # transforms (giảm size cho RTX 2060)
    train_transforms = transforms.Compose([
        videotransforms.RandomCrop(224),
        videotransforms.RandomHorizontalFlip(),
    ])

    test_transforms = transforms.Compose([
        videotransforms.CenterCrop(224)
    ])

    dataset = Dataset(train_split, 'train', root, mode, train_transforms)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_dataset = Dataset(train_split, 'test', root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0
    )

    dataloaders = {'train': dataloader, 'test': val_dataloader}

    # model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('weights/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('weights/rgb_imagenet.pt'))

    num_classes = dataset.num_classes
    i3d.replace_logits(num_classes)

    if weights:
        i3d.load_state_dict(torch.load(weights))

    i3d.to(device)

    optimizer = optim.Adam(i3d.parameters(), lr=1e-4, weight_decay=1e-8)

    steps = 0
    epoch = 0
    best_val_score = 0

    configs.max_steps = 3000  # giảm để test nhanh

    while steps < configs.max_steps and epoch < 200:
        print(f"\nStep {steps}/{configs.max_steps}")
        print("-" * 10)

        epoch += 1

        for phase in ['train', 'test']:

            if phase == 'train':
                i3d.train()
            else:
                i3d.eval()

            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0

            confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

            for data in dataloaders[phase]:
                if data == -1:
                    continue

                inputs, labels, vid = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                t = inputs.size(2)

                with autocast():
                    per_frame_logits = i3d(inputs, pretrained=False)
                    per_frame_logits = F.interpolate(per_frame_logits, size=t, mode='linear')

                    loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                    cls_loss = F.binary_cross_entropy_with_logits(
                        torch.max(per_frame_logits, dim=2)[0],
                        torch.max(labels, dim=2)[0]
                    )

                    loss = 0.5 * loc_loss + 0.5 * cls_loss

                if phase == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    steps += 1

                tot_loss += loss.item()
                tot_loc_loss += loc_loss.item()
                tot_cls_loss += cls_loss.item()

                predictions = torch.max(per_frame_logits, dim=2)[0]
                gt = torch.max(labels, dim=2)[0]

                for i in range(predictions.shape[0]):
                    confusion_matrix[
                        torch.argmax(gt[i]).item(),
                        torch.argmax(predictions[i]).item()
                    ] += 1

                if steps % 50 == 0:
                    print("GPU:", torch.cuda.memory_allocated()/1024**3, "GB")

            acc = float(np.trace(confusion_matrix)) / np.sum(confusion_matrix)

            print(f"{phase.upper()} | Loss: {tot_loss:.4f} | Acc: {acc:.4f}")

            if phase == 'test':
                if acc > best_val_score:
                    best_val_score = acc
                    model_name = f"{save_model}/best_{steps}.pt"
                    torch.save(i3d.state_dict(), model_name)
                    print("Saved:", model_name)


if __name__ == '__main__':
    config_file = 'configfiles/asl2000.ini'
    configs = Config(config_file)

    run(configs=configs)