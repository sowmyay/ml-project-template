import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from tqdm import tqdm

# TODO 9: Import your dataset, model and transforms
from pytorch_template.src.models import DummyModel
from pytorch_template.src.transforms import DummyTransform
from pytorch_template.src.datasets import DummyDataset


def main(args):
    if torch.cuda.is_available():
        print("🐎 Running on GPU(s)", file=sys.stderr)
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        print("🐌 Running on CPU(s)", file=sys.stderr)
        device = torch.device("cpu")

    # TODO 10: Change from DummyModel to your model
    model = DummyModel()

    for params in model.parameters():
        params.requires_grad = False
    model.eval()
    model = model.to(device)
    model = nn.DataParallel(model)

    # TODO 11: Change from DummyTransform to your transforms or inbuilt pytorch transforms
    transforms = Compose([DummyTransform(),
                          DummyTransform()])

    # TODO 12: Change from DummyDataset to your dataset
    dataset = DummyDataset(args.dataset, transform=transforms)

    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    chkpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(chkpt["state_dict"])

    predict(model, device, loader)


def predict(model, device, data_loader):
    with torch.no_grad():
        for item in tqdm(data_loader, desc="predict", unit="batch"):
            item.to(device)
            output = model(item)
    return

