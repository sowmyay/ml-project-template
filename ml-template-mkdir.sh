#! /bin/bash 

echo "ML project named" $1
eval "mkdir -p $1/{$1/cli,bin,notebooks}"
eval "touch $1/$1/cli/{__init__,__main__,train,predict}.py"
eval "touch $1/$1/{__init__,models,transforms,datasets}.py"
eval "touch $1/{.flake8,.gitignore,.dockerignore,Makefile,Dockerfile.cpu,Dockerfile.gpu,README.md,requirements.in}"

cat <<END > "$1"/bin/$1
#!/bin/bash

python3 -m $1.cli "\$@"
END

cat <<END > "$1"/.flake8
[flake8]
max-line-length = 120

per-file-ignores =
  hubconf.py : E402, F401
END

cat <<END > "$1"/Dockerfile.cpu
FROM ubuntu:18.04

WORKDIR /usr/src/app

ENV LANG="C.UTF-8" LC_ALL="C.UTF-8" PATH="/opt/venv/bin:$PATH" PIP_NO_CACHE_DIR="false"

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \\
    python3 python3-pip python3-venv && \\
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python3 -m venv /opt/venv && \
    python3 -m pip install pip==19.3.1 pip-tools==4.2.0 setuptools==45.2.0

RUN echo "https://download.pytorch.org/whl/cpu/torch-1.4.0%2Bcpu-cp36-cp36m-linux_x86_64.whl       \\
          --hash=sha256:f20312fc168147e6b152e8a335bc916cc0b86a39d9a39c69d0223ffa99b72de4           \\
          \n                                                                                       \\
          https://download.pytorch.org/whl/cpu/torchvision-0.5.0%2Bcpu-cp36-cp36m-linux_x86_64.whl \\
          --hash=sha256:ae3686f57a56c71ee7cc001bb067b3ebb246f1d6abc8b9023f7d7d043a4f8b83           \\
          \n" >> requirements.txt

RUN python3 -m piptools sync

COPY . .

ENTRYPOINT ["/usr/src/app/bin/$1"]
CMD ["-h"]"
END

cat <<END > "$1"/Dockerfile.gpu
FROM nvidia/cuda:10.1-cudnn7-runtime

WORKDIR /usr/src/app

ENV LANG="C.UTF-8" LC_ALL="C.UTF-8" PATH="/opt/venv/bin:$PATH" PIP_NO_CACHE_DIR="false"

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \\
    python3 python3-pip python3-venv && \\
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python3 -m venv /opt/venv && \
    python3 -m pip install pip==19.3.1 pip-tools==4.2.0 setuptools==41.4.0

RUN echo "https://download.pytorch.org/whl/cu101/torch-1.4.0-cp36-cp36m-linux_x86_64.whl       \\
          --hash=sha256:8856f334aa9ecb742c1504bd2563d0ffb8dceb97149c8d72a04afa357f667dbc       \\
          \n                                                                                   \\
          https://download.pytorch.org/whl/cu101/torchvision-0.5.0-cp36-cp36m-linux_x86_64.whl \\
          --hash=sha256:df994daf1b04a183779538022fb4345b66364fccf66b33e1095ae8d4955a5406       \\
          \n" >> requirements.txt

RUN python3 -m piptools sync

COPY . .

ENTRYPOINT ["/usr/src/app/bin/$1"]
CMD ["-h"]
END

cat  <<END > "$1"/Makefile
dockerimage ?= $1
dockerfile ?= Dockerfile.cpu
srcdir ?= \$(shell pwd)
datadir ?= \$(shell pwd)

install:
	@docker build -t \$(dockerimage) -f \$(dockerfile) .

i: install


update:
	@docker build -t \$(dockerimage) -f \$(dockerfile) . --pull --no-cache

u: update


run:
	@docker run --ipc=host -it --rm -v \$(srcdir):/usr/src/app/ -v \$(datadir):/data --entrypoint=/bin/bash \$(dockerimage)

r: run

tensorboard:
	@docker run -it --rm -p 6006:6006 -v \$(datadir):/data tensorflow/tensorflow:2.0.1-py3 tensorboard --bind_all --logdir /data/runs

t: tensorboard

gpu:
	@docker run --runtime=nvidia --ipc=host -it --rm -v \$(srcdir):/usr/src/app/ -v \$(datadir):/data --entrypoint=/bin/bash \$(dockerimage)

g: gpu


publish:
	@docker image save \$(dockerimage) \
	  | pv -N "Publish \$(dockerimage) to \$(sshopts)" -s \$(shell docker image inspect \$(dockerimage) --format "{{.Size}}") \
	  | ssh \$(sshopts) "docker image load"

p: publish


.PHONY: install i run r update u gpu g publish p tensorboard t
END

cat <<END > "$1"/README.md
## $1

## Usage

## Development

We provide CPU and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) based GPU Dockerfiles for self-contained and reproducible environments.
Use the convenience Makefile to build the Docker image and then get into the container mounting a host directory to \`/data\` inside the container:

\`\`\`
make
make run datadir=/path/to/dataset
\`\`\`

By default we build and run the CPU Docker images; for GPUs run:

\`\`\`
make dockerfile=Dockerfile.gpu
make gpu datadir=/path/to/dataset
\`\`\`

## Preprocessing


## Model training


## Prediction

## References

## License
END

cat <<END > "$1"/"$1"/cli/__main__.py
import argparse
from pathlib import Path

import $1.cli.train


parser = argparse.ArgumentParser(prog="$1")

subcmd = parser.add_subparsers(dest="command")
subcmd.required = True

Formatter = argparse.ArgumentDefaultsHelpFormatter

train = subcmd.add_parser("train", help="trains super-resolution model", formatter_class=Formatter)
train.add_argument("--dataset", type=Path, help="path to directory for loading data from")
train.add_argument("--model", type=Path, required=True, help="file to save trained model to")
train.add_argument("--num-workers", type=int, default=0, help="number of parallel workers")
train.add_argument("--batch-size", type=int, default=64, help="number of items per batch")
train.add_argument("--checkpoint", type=str, required=False, help="path to a model checkpoint (to retrain)")
train.add_argument("--resume", type=bool, default=False, help="resume training or fine-tuning (if checkpoint)")
train.set_defaults(main=$1.cli.train.main)

args = parser.parse_args()
args.main(args)
END

cat <<END > "$1"/"$1"/datasets.py
import numpy as np
from torch.utils.data import Dataset


class DummyDataset(Dataset):
	def __init__(self, paths, transform=None):
		super().__init__()
		self.paths = paths
		self.transform = transform
	
	def __len__(self):
		return len(self.paths)

	def __getitem__(self, i):
		path = self.paths[i]
		item = np.load(path, allow_pickle=False)
		target = True
		if self.transform:
			item = self.tranform(item)
		return item, target
END

cat <<END > "$1"/"$1"/models.py
import torch.nn as nn


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Conv2d(24, 24, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(24, 12, kernel_size=3, padding=1))

    def forward(self, x):
        return self.model(x)
END

cat <<END > "$1"/"$1"/transforms.py


class DummyTransform:
	def __init__(self):
		super().__init__()

	def __call__(self, item):
		return item

END

cat <<END > "$1"/"$1"/cli/train.py
import sys

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.transforms import Compose
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm

from $1.models import DummyModel
from $1.datasets import DummyDataset
from $1.transforms import DummyTransform

train_counter = 0
valid_counter = 0


def main(args):
    if torch.cuda.is_available():
        print("🐎 Running on GPU(s)", file=sys.stderr)
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        print("🐌 Running on CPU(s)", file=sys.stderr)
        device = torch.device("cpu")

    writer = SummaryWriter()
    transforms = Compose([DummyTransform(),
                          DummyTransform()])

    dataset = DummyDataset(args.dataset, transform=transforms)

    num_train = int(len(dataset) * 0.9)
    num_val = len(dataset) - num_train

    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              num_workers=args.num_workers, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            num_workers=args.num_workers, shuffle=False)

    model = DummyModel()
    model = model.to(device)
    model = nn.DataParallel(model)

    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = nn.NLLLoss()

    resume = 0
    if args.checkpoint:
        chkpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(chkpt["state_dict"])

        if args.resume:
            optimizer.load_state_dict(chkpt["optimizer"])
            resume = chkpt["epoch"]

    if resume >= args.num_epochs:
        sys.exit("Error: Epoch {} set in {} already reached by the checkpoint provided".format(args.num_epochs,
                                                                                               args.model))

    for epoch in range(args.num_epochs):
        print("Epoch {}/{}".format(epoch, args.num_epochs - 1))
        print("-" * 10)

        train_loss = train_step(model, criterion=criterion, device=device, dataloader=train_loader,
                                dataset=train_dataset, optimizer=optimizer, writer=writer)

        print("train loss: {:.4f}".format(train_loss))

        val_loss = validation_step(model, criterion=criterion, device=device, dataloader=val_loader,
                                   dataset=val_dataset, writer=writer)

        print("val loss: {:.4f}".format(val_loss))

        checkpoint = args.model / "checkpoint-{:05d}-of-{:05d}.pth".format(epoch + 1, args.num_epochs)
        states = {"epoch": epoch + 1, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        torch.save(states, checkpoint)

    writer.close()


def train_step(model, criterion, device, dataloader, dataset, optimizer, writer):
    model.train()

    global train_counter
    running_loss = 0.0
    batch = 1
    for item, target in tqdm(dataloader, desc="train", unit="batch"):
        item.to(device)
        target.to(device)

        optimizer.zero_grad()

        loss = 0

        output = model(item)
        loss = criterion(output, target)
        loss.backward()

        optimizer.step()
        running_loss += loss.item() * item.size(0)

        if train_counter % 100 == 0:
            writer.add_scalar("Loss/train", running_loss / batch * item.size(0), train_counter)

        batch += 1
        train_counter += 1

    return running_loss / len(dataset)


def validation_step(model, criterion, device, dataloader, dataset, writer):
    model.eval()

    global valid_counter
    running_loss = 0.0
    batch = 1
    with torch.no_grad():
        for item, target in tqdm(dataloader, desc="val", unit="batch"):
            item.to(device)
            target.to(device)

            output = model(item)
            loss = criterion(output, target)
            running_loss += loss.item() * item.size(0)

            if valid_counter % 100 == 0:
                writer.add_scalar("Loss/eval", running_loss / batch * item.size(0), valid_counter)
            batch += 1
            valid_counter += 1

    return running_loss / len(dataset)

END

cat <<END > "$1"/"$1"/cli/predict.py
def main(args):
    if torch.cuda.is_available():
        print("🐎 Running on GPU(s)", file=sys.stderr)
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        print("🐌 Running on CPU(s)", file=sys.stderr)
        device = torch.device("cpu")

    model = DummyModel()

    for params in model.parameters():
        params.requires_grad = False
    model.eval()
    model = model.to(device)
    model = nn.DataParallel(model)

    transforms = Compose([DummyTransform(),
                          DummyTransform()])

    dataset = DummyDataset(args.dataset, transform=transforms)

    loader = DataLoader(dataset, batch_size=args.batch_size,
                              num_workers=args.num_workers, shuffle=False)

    chkpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(chkpt["state_dict"])

    predict(model, item)

def predict(model, device, dataloader, dataset):
    with torch.no_grad():
        for item in tqdm(dataloader, desc="predict", unit="batch"):
            item.to(device)

            output = model(item)

    return
END

cat <<END > "$1"/requirements.in
tqdm
scipy
tensorboard
END

