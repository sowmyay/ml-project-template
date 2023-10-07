#!/bin/bash

echo "Creating ML template for $1"
eval "mkdir -p $1/$1/{src,cli,notebooks}"
eval "touch $1/{.flake8,.gitignore,.dockerignore,Makefile,Dockerfile.cpu,Dockerfile.gpu,README.md,LICENSE.md,requirements.txt,jupyter.sh}"
eval "touch $1/$1/{__init__,__main__}.py"
eval "touch $1/$1/cli/{__init__,train,predict}.py"
eval "touch $1/$1/src/{__init__,models,transforms,datasets}.py"
chmod +x "$1/jupyter.sh"

cat <<END > "$1"/.flake8
[flake8]
max-line-length = 120

per-file-ignores =
  hubconf.py : E402, F401
END

cat <<END > "$1"/jupyter.sh
#!/bin/bash

jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root

END

cat <<END > "$1"/Dockerfile.cpu
FROM ubuntu:22.04

ENV LANG="C.UTF-8" LC_ALL="C.UTF-8" PATH="/home/python/.local/bin:$PATH" PIP_NO_CACHE_DIR="false"

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python-is-python3 curl ca-certificates vim && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 python && \
    useradd  --uid 1000 --gid python --shell /bin/bash --create-home python

USER 1000
RUN mkdir /home/python/app
WORKDIR /home/python/app

COPY --chown=python:python requirements.txt ./
RUN python3 -m pip install -r requirements.txt
RUN pip install torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

COPY --chown=python:python . .

END

cat <<END > "$1"/Dockerfile.gpu
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV LANG="C.UTF-8" LC_ALL="C.UTF-8" PATH="/home/python/.local/bin:$PATH" PIP_NO_CACHE_DIR="false"

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python-is-python3 curl ca-certificates vim && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 python && \
    useradd  --uid 1000 --gid python --shell /bin/bash --create-home python

USER 1000
RUN mkdir /home/python/app
WORKDIR /home/python/app

COPY --chown=python:python requirements.txt ./
RUN python3 -m pip install -r requirements.txt
RUN pip install torch==2.0.1+cu101 torchvision==0.15.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html

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
	@docker run --ipc=host -it --rm -v \$(srcdir):/home/python/app/ -v \$(datadir):/data --entrypoint=/bin/bash \$(dockerimage)

r: run

jupyter:
	@docker run --ipc=host -it --rm -p 8888:8888 -v \$(srcdir):/home/python/app/ -v \$(datadir):/data --entrypoint=/home/python/app/jupyter.sh \$(dockerimage)

j: jupyter

tensorboard:
	@docker run -it --rm -p 6006:6006 -v \$(datadir):/data tensorflow/tensorflow tensorboard --bind_all --logdir /data

t: tensorboard

gpu:
  @docker run --gpus all --ipc=host -it --rm -v \$(srcdir):/home/python/app/ -v \$(datadir):/data --entrypoint=/bin/bash \$(dockerimage)

g: gpu

jupyter-gpu:
	@docker run --gpus all --ipc=host -it --rm -p 8888:8888 -v \$(srcdir):/home/python/app/ -v \$(datadir):/data --entrypoint=/home/python/app/jupyter.sh \$(dockerimage)

jg: jupyter-gpu


publish:
	@docker image save \$(dockerimage) 	  | pv -N "Publish \$(dockerimage) to \$(sshopts)" -s \$(shell docker image inspect \$(dockerimage) --format "{{.Size}}") 	  | ssh \$(sshopts) "docker image load"

p: publish


.PHONY: install i run r update u gpu g publish p tensorboard t jupyter j jupyter-gpu jg
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
Inside your docker container, run -
\`\`\`bash
python3 -m $1 train --dataset /data/train \\
--model /data/models \\
--num-workers 12 \\
--batch-size 512 \\
--num-epochs 100
\`\`\`

## Prediction
Inside your docker container, run -
\`\`\`bash
python3 -m $1 predict --dataset /data/predict \\
--checkpoint /data/models/best-checkpoint.pth \\
--num-workers 12 \\
--batch-size 512
\`\`\`

## Jupyter notebook
To start jupyter environment run the following command from your terminal
\`\`\`bash
make jupyter
\`\`\`
or
\`\`\`bash
make jupyter-gpu
\`\`\`

## Monitor Tensorboard
To monitor the loss plots on tensorboard, run the following command from your terminal-
\`\`\`bash
make tensorboard datadir=/path/to/runs_directory
\`\`\`
Go to \`localhost:6006\` in your browser to monitor the tensorboard plots

## References

## License
Copyright © 2020 $2 $3

Distributed under the MIT License (MIT).
END

cat <<END > "$1"/LICENSE.md
MIT License

Copyright (c) 2023 $2 $3

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
END

cat <<END > "$1"/"$1"/__main__.py
import argparse
from pathlib import Path

import $1.cli.train
import $1.cli.predict

parser = argparse.ArgumentParser(prog="$1")

subcmd = parser.add_subparsers(dest="command")
subcmd.required = True

Formatter = argparse.ArgumentDefaultsHelpFormatter

train = subcmd.add_parser("train", help="train model", formatter_class=Formatter)
train.add_argument("--dataset", type=Path, help="path to directory for loading data from")
train.add_argument("--model", type=Path, required=True, help="file to save trained model to")
train.add_argument("--num-workers", type=int, default=0, help="number of parallel workers")
train.add_argument("--num-epochs", type=int, default=100, help="number of epochs to train for")
train.add_argument("--batch-size", type=int, default=64, help="number of items per batch")
train.add_argument("--checkpoint", type=str, required=False, help="path to a model checkpoint (to retrain)")
train.add_argument("--resume", type=bool, default=False, help="resume training or fine-tuning (if checkpoint)")
train.set_defaults(main=$1.cli.train.main)

predict = subcmd.add_parser("predict", help="predicts on model", formatter_class=Formatter)
predict.add_argument("--dataset", type=Path, help="path to directory for loading data from")
predict.add_argument("--checkpoint", type=str, required=False, help="path to a model checkpoint to predict with")
predict.add_argument("--num-workers", type=int, default=0, help="number of parallel workers")
predict.add_argument("--batch-size", type=int, default=64, help="number of items per batch")
predict.set_defaults(main=$1.cli.predict.main)

args = parser.parse_args()
args.main(args)

END

cat <<END > "$1"/"$1"/src/datasets.py
import numpy as np
from torch.utils.data import Dataset

# TODO 1: Create a your own custom Dataset here

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
			item = self.transform(item)
		return item, target

END

cat <<END > "$1"/"$1"/src/models.py
import torch.nn as nn

# TODO 2: Write your model architecture in a new model class here

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Conv2d(24, 24, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(24, 12, kernel_size=3, padding=1))

    def forward(self, x):
        return self.model(x)

END

cat <<END > "$1"/"$1"/src/transforms.py
# TODO 3: Create a your own custom transforms here

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

# TODO 4: Import your dataset, model and transforms
from $1.src.models import DummyModel
from $1.src.datasets import DummyDataset
from $1.src.transforms import DummyTransform

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

    # TODO 5: Change from DummyTransform to your transforms or inbuilt pytorch transforms
    transforms = Compose([DummyTransform(),
                          DummyTransform()])

    # TODO 6: Change from DummyDataset to your dataset
    dataset = DummyDataset(args.dataset, transform=transforms)

    num_train = int(len(dataset) * 0.9)
    num_val = len(dataset) - num_train

    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              num_workers=args.num_workers, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            num_workers=args.num_workers, shuffle=False)

    # TODO 7: Change from DummyModel to your model
    model = DummyModel()
    model = model.to(device)
    model = nn.DataParallel(model)

    # TODO 8: Change optimizer and loss functions as needed
    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = nn.NLLLoss()

    best_loss = float("inf")
    resume = 0
    if args.checkpoint:
        chkpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(chkpt["state_dict"])

        if args.resume:
            optimizer.load_state_dict(chkpt["optimizer"])
            resume = chkpt["epoch"]
            best_loss = chkpt["val_loss"]

    if resume >= args.num_epochs:
        sys.exit("Error: Epoch {} set in {} already reached by the checkpoint provided".format(args.num_epochs,
                                                                                               args.model))

    for epoch in range(args.num_epochs):
        print("Epoch {}/{}".format(epoch, args.num_epochs - 1))
        print("-" * 10)

        train_loss = train_step(model, criterion=criterion, device=device, data_loader=train_loader,
                                dataset=train_dataset, optimizer=optimizer, writer=writer)

        print("train loss: {:.4f}".format(train_loss))

        val_loss = validation_step(model, criterion=criterion, device=device, data_loader=val_loader,
                                   dataset=val_dataset, writer=writer)

        print("val loss: {:.4f}".format(val_loss))

        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint = args.model / "beat-checkpoint.pth"
            states = {"epoch": epoch + 1,
                      "state_dict": model.state_dict(),
                      "optimizer": optimizer.state_dict(),
                      "val_loss": val_loss,
                      "train_loss": train_loss}
            torch.save(states, checkpoint)

    writer.close()


def train_step(model, criterion, device, data_loader, dataset, optimizer, writer):
    model.train()

    global train_counter
    running_loss = 0.0
    epoch_counter = 0
    for item, target in tqdm(data_loader, desc="train", unit="batch"):
        item.to(device)
        target.to(device)

        optimizer.zero_grad()

        output = model(item)
        loss = criterion(output, target)
        loss.backward()

        optimizer.step()
        running_loss += loss.item() * item.size(0)

        epoch_counter += item.size(0)

        if train_counter % 100 == 0:
            writer.add_scalar("Loss/train", running_loss / epoch_counter, train_counter)

        train_counter += 1

    return running_loss / len(dataset)


def validation_step(model, criterion, device, data_loader, dataset, writer):
    model.eval()

    global valid_counter
    running_loss = 0.0
    epoch_counter = 0
    with torch.no_grad():
        for item, target in tqdm(data_loader, desc="val", unit="batch"):
            item.to(device)
            target.to(device)

            output = model(item)
            loss = criterion(output, target)
            running_loss += loss.item() * item.size(0)

            epoch_counter += item.size(0)
            if valid_counter % 100 == 0:
                writer.add_scalar("Loss/eval", running_loss / epoch_counter, valid_counter)
            valid_counter += 1

    return running_loss / len(dataset)

END

cat <<END > "$1"/"$1"/cli/predict.py
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from tqdm import tqdm

# TODO 9: Import your dataset, model and transforms
from $1.src.models import DummyModel
from $1.src.transforms import DummyTransform
from $1.src.datasets import DummyDataset


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

END

cat <<END > "$1"/requirements.txt
# TODO 0: Update frameworks as needed for your project. Torch and torchvision are included in the dockerfiles
einops==0.5.0
jupyter
matplotlib==3.7.1
numpy>=1.22.4
pandas==2.0.3
Pillow>=9.3.0
pandas==2.0.3
scikit-image==0.21.0
scikit-learn==1.3.0
tensorboard==2.14.0
tqdm==4.65.0
END

echo "Voila!"
