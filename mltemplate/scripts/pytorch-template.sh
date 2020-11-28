#!/bin/bash

echo "Creating ML template for " $1
eval "mkdir -p $1/{$1/cli,notebooks}"
eval "touch $1/$1/cli/{__init__,__main__,train,predict}.py"
eval "touch $1/$1/{__init__,models,transforms,datasets}.py"
eval "touch $1/{.flake8,.gitignore,.dockerignore,Makefile,Dockerfile.cpu,Dockerfile.gpu,README.md,LICENSE.md,pyproject.toml}"

cat <<END > "$1"/.flake8
[flake8]
max-line-length = 120

per-file-ignores =
  hubconf.py : E402, F401
END

cat <<END > "$1"/Dockerfile.cpu
FROM ubuntu:20.04

ENV LANG="C.UTF-8" LC_ALL="C.UTF-8" PATH="/home/python/.poetry/bin:/home/python/.local/bin:\$PATH" PIP_NO_CACHE_DIR="false"

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \\
    python3 python3-pip python3-venv python-is-python3 curl ca-certificates vim && \\
    rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 python && \\
    useradd  --uid 1000 --gid python --shell /bin/bash --create-home python

USER 1000
RUN mkdir /home/python/app
WORKDIR /home/python/app

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/d2fd581c9a856a5c4e60a25acb95d06d2a963cf2/get-poetry.py | python - --version 1.0.10
RUN poetry config virtualenvs.create false

COPY --chown=python:python pyproject.toml poetry.lock ./
RUN poetry install --no-interaction --no-root --ansi

RUN python3 -m pip install torch==1.7.0+cpu torchvision==0.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

COPY --chown=python:python . .
RUN poetry install --no-interaction --ansi

END

cat <<END > "$1"/Dockerfile.gpu
FROM nvidia/cuda:11.0-runtime-ubuntu20.04

ENV LANG="C.UTF-8" LC_ALL="C.UTF-8" PATH="/home/python/.poetry/bin:/home/python/.local/bin:\$PATH" PIP_NO_CACHE_DIR="false"

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \\
    python3 python3-pip python3-venv python-is-python3 curl ca-certificates vim && \\
    rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 python && \\
    useradd  --uid 1000 --gid python --shell /bin/bash --create-home python

USER 1000
RUN mkdir /home/python/app
WORKDIR /home/python/app

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/d2fd581c9a856a5c4e60a25acb95d06d2a963cf2/get-poetry.py | python - --version 1.0.10
RUN poetry config virtualenvs.create false

COPY --chown=python:python pyproject.toml poetry.lock ./
RUN poetry install --no-interaction --no-root --ansi

RUN pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

COPY --chown=python:python . .
RUN poetry install --no-interaction --ansi

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

tensorboard:
	@docker run -it --rm -p 6006:6006 -v \$(datadir):/data tensorflow/tensorflow:2.0.1-py3 tensorboard --bind_all --logdir /data/runs

t: tensorboard

gpu:
  @docker run --gpus all --ipc=host -it --rm -v \$(srcdir):/home/python/app/ -v \$(datadir):/data --entrypoint=/bin/bash \$(dockerimage)

g: gpu


publish:
	@docker image save \$(dockerimage) 	  | pv -N "Publish \$(dockerimage) to \$(sshopts)" -s \$(shell docker image inspect \$(dockerimage) --format "{{.Size}}") 	  | ssh \$(sshopts) "docker image load"

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
Inside your docker container, run -
\`\`\`bash
poetry run $1 train --dataset /data/train \\
--model /data/models \\
--num-workers 12 \\
--batch-size 512 \\
--num-epochs 100
\`\`\`

## Prediction
Inside your docker container, run -
\`\`\`bash
poetry run $1 predict --dataset /data/predict \\
--checkpoint /data/models/best-checkpoint.pth \\
--num-workers 12 \\
--batch-size 512
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

Copyright (c) 2020 $2 $3

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

cat <<END > "$1"/"$1"/cli/__main__.py
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

cat <<END > "$1"/"$1"/datasets.py
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

cat <<END > "$1"/"$1"/models.py
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

cat <<END > "$1"/"$1"/transforms.py
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
from $1.models import DummyModel
from $1.transforms import DummyTransform
from $1.datasets import DummyDataset


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

cat <<END > "$1"/pyproject.toml
[tool.poetry]
name = "$1"
version = "0.1.0"
description = ""
authors = ["$2 $3 <$4>"]

[tool.poetry.dependencies]
python = "^3.7"
tqdm = "^4.53.0"
pandas = "^1.1.4"
numpy = "^1.19.4"
sklearn = "^0.0"
tensorboard = "^2.4.0"
einops = "^0.3.0"

[tool.poetry.dev-dependencies]

[tool.poetry.scripts]
$1 = "$1.cli.__main__:main"

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"

END

cat <<END > "$1"/poetry.lock
[[package]]
name = "absl-py"
version = "0.11.0"
description = "Abseil Python Common Libraries, see https://github.com/abseil/abseil-py."
category = "main"
optional = false
python-versions = "*"

[package.dependencies]
six = "*"

[[package]]
name = "cachetools"
version = "4.1.1"
description = "Extensible memoizing collections and decorators"
category = "main"
optional = false
python-versions = "~=3.5"

[[package]]
name = "certifi"
version = "2020.11.8"
description = "Python package for providing Mozilla's CA Bundle."
category = "main"
optional = false
python-versions = "*"

[[package]]
name = "chardet"
version = "3.0.4"
description = "Universal encoding detector for Python 2 and 3"
category = "main"
optional = false
python-versions = "*"

[[package]]
name = "einops"
version = "0.3.0"
description = "A new flavour of deep learning operations"
category = "main"
optional = false
python-versions = "*"

[[package]]
name = "google-auth"
version = "1.23.0"
description = "Google Authentication Library"
category = "main"
optional = false
python-versions = ">=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*"

[package.dependencies]
cachetools = ">=2.0.0,<5.0"
pyasn1-modules = ">=0.2.1"
rsa = {version = ">=3.1.4,<5", markers = "python_version >= \"3.5\""}
six = ">=1.9.0"

[package.extras]
aiohttp = ["aiohttp (>=3.6.2,<4.0.0dev)"]

[[package]]
name = "google-auth-oauthlib"
version = "0.4.2"
description = "Google Authentication Library"
category = "main"
optional = false
python-versions = ">=3.6"

[package.dependencies]
google-auth = "*"
requests-oauthlib = ">=0.7.0"

[package.extras]
tool = ["click"]

[[package]]
name = "grpcio"
version = "1.33.2"
description = "HTTP/2-based RPC framework"
category = "main"
optional = false
python-versions = "*"

[package.dependencies]
six = ">=1.5.2"

[package.extras]
protobuf = ["grpcio-tools (>=1.33.2)"]

[[package]]
name = "idna"
version = "2.10"
description = "Internationalized Domain Names in Applications (IDNA)"
category = "main"
optional = false
python-versions = ">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*"

[[package]]
name = "importlib-metadata"
version = "3.1.0"
description = "Read metadata from Python packages"
category = "main"
optional = false
python-versions = ">=3.6"

[package.dependencies]
zipp = ">=0.5"

[package.extras]
docs = ["sphinx", "rst.linker"]
testing = ["packaging", "pep517", "unittest2", "importlib-resources (>=1.3)"]

[[package]]
name = "joblib"
version = "0.17.0"
description = "Lightweight pipelining: using Python functions as pipeline jobs."
category = "main"
optional = false
python-versions = ">=3.6"

[[package]]
name = "markdown"
version = "3.3.3"
description = "Python implementation of Markdown."
category = "main"
optional = false
python-versions = ">=3.6"

[package.dependencies]
importlib-metadata = {version = "*", markers = "python_version < \"3.8\""}

[package.extras]
testing = ["coverage", "pyyaml"]

[[package]]
name = "numpy"
version = "1.19.4"
description = "NumPy is the fundamental package for array computing with Python."
category = "main"
optional = false
python-versions = ">=3.6"

[[package]]
name = "oauthlib"
version = "3.1.0"
description = "A generic, spec-compliant, thorough implementation of the OAuth request-signing logic"
category = "main"
optional = false
python-versions = ">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*"

[package.extras]
rsa = ["cryptography"]
signals = ["blinker"]
signedtoken = ["cryptography", "pyjwt (>=1.0.0)"]

[[package]]
name = "pandas"
version = "1.1.4"
description = "Powerful data structures for data analysis, time series, and statistics"
category = "main"
optional = false
python-versions = ">=3.6.1"

[package.dependencies]
numpy = ">=1.15.4"
python-dateutil = ">=2.7.3"
pytz = ">=2017.2"

[package.extras]
test = ["pytest (>=4.0.2)", "pytest-xdist", "hypothesis (>=3.58)"]

[[package]]
name = "protobuf"
version = "3.14.0"
description = "Protocol Buffers"
category = "main"
optional = false
python-versions = "*"

[package.dependencies]
six = ">=1.9"

[[package]]
name = "pyasn1"
version = "0.4.8"
description = "ASN.1 types and codecs"
category = "main"
optional = false
python-versions = "*"

[[package]]
name = "pyasn1-modules"
version = "0.2.8"
description = "A collection of ASN.1-based protocols modules."
category = "main"
optional = false
python-versions = "*"

[package.dependencies]
pyasn1 = ">=0.4.6,<0.5.0"

[[package]]
name = "python-dateutil"
version = "2.8.1"
description = "Extensions to the standard Python datetime module"
category = "main"
optional = false
python-versions = "!=3.0.*,!=3.1.*,!=3.2.*,>=2.7"

[package.dependencies]
six = ">=1.5"

[[package]]
name = "pytz"
version = "2020.4"
description = "World timezone definitions, modern and historical"
category = "main"
optional = false
python-versions = "*"

[[package]]
name = "requests"
version = "2.25.0"
description = "Python HTTP for Humans."
category = "main"
optional = false
python-versions = ">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*"

[package.dependencies]
certifi = ">=2017.4.17"
chardet = ">=3.0.2,<4"
idna = ">=2.5,<3"
urllib3 = ">=1.21.1,<1.27"

[package.extras]
security = ["pyOpenSSL (>=0.14)", "cryptography (>=1.3.4)"]
socks = ["PySocks (>=1.5.6,!=1.5.7)", "win-inet-pton"]

[[package]]
name = "requests-oauthlib"
version = "1.3.0"
description = "OAuthlib authentication support for Requests."
category = "main"
optional = false
python-versions = ">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*"

[package.dependencies]
oauthlib = ">=3.0.0"
requests = ">=2.0.0"

[package.extras]
rsa = ["oauthlib[signedtoken] (>=3.0.0)"]

[[package]]
name = "rsa"
version = "4.6"
description = "Pure-Python RSA implementation"
category = "main"
optional = false
python-versions = ">=3.5, <4"

[package.dependencies]
pyasn1 = ">=0.1.3"

[[package]]
name = "scikit-learn"
version = "0.23.2"
description = "A set of python modules for machine learning and data mining"
category = "main"
optional = false
python-versions = ">=3.6"

[package.dependencies]
joblib = ">=0.11"
numpy = ">=1.13.3"
scipy = ">=0.19.1"
threadpoolctl = ">=2.0.0"

[package.extras]
alldeps = ["numpy (>=1.13.3)", "scipy (>=0.19.1)"]

[[package]]
name = "scipy"
version = "1.5.4"
description = "SciPy: Scientific Library for Python"
category = "main"
optional = false
python-versions = ">=3.6"

[package.dependencies]
numpy = ">=1.14.5"

[[package]]
name = "six"
version = "1.15.0"
description = "Python 2 and 3 compatibility utilities"
category = "main"
optional = false
python-versions = ">=2.7, !=3.0.*, !=3.1.*, !=3.2.*"

[[package]]
name = "sklearn"
version = "0.0"
description = "A set of python modules for machine learning and data mining"
category = "main"
optional = false
python-versions = "*"

[package.dependencies]
scikit-learn = "*"

[[package]]
name = "tensorboard"
version = "2.4.0"
description = "TensorBoard lets you watch Tensors Flow"
category = "main"
optional = false
python-versions = ">= 2.7, != 3.0.*, != 3.1.*"

[package.dependencies]
absl-py = ">=0.4"
google-auth = ">=1.6.3,<2"
google-auth-oauthlib = ">=0.4.1,<0.5"
grpcio = ">=1.24.3"
markdown = ">=2.6.8"
numpy = ">=1.12.0"
protobuf = ">=3.6.0"
requests = ">=2.21.0,<3"
six = ">=1.10.0"
tensorboard-plugin-wit = ">=1.6.0"
werkzeug = ">=0.11.15"

[[package]]
name = "tensorboard-plugin-wit"
version = "1.7.0"
description = "What-If Tool TensorBoard plugin."
category = "main"
optional = false
python-versions = "*"

[[package]]
name = "threadpoolctl"
version = "2.1.0"
description = "threadpoolctl"
category = "main"
optional = false
python-versions = ">=3.5"

[[package]]
name = "tqdm"
version = "4.53.0"
description = "Fast, Extensible Progress Meter"
category = "main"
optional = false
python-versions = "!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,>=2.7"

[package.extras]
dev = ["py-make (>=0.1.0)", "twine", "argopt", "pydoc-markdown", "wheel"]

[[package]]
name = "urllib3"
version = "1.26.2"
description = "HTTP library with thread-safe connection pooling, file post, and more."
category = "main"
optional = false
python-versions = ">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, <4"

[package.extras]
brotli = ["brotlipy (>=0.6.0)"]
secure = ["pyOpenSSL (>=0.14)", "cryptography (>=1.3.4)", "idna (>=2.0.0)", "certifi", "ipaddress"]
socks = ["PySocks (>=1.5.6,!=1.5.7,<2.0)"]

[[package]]
name = "werkzeug"
version = "1.0.1"
description = "The comprehensive WSGI web application library."
category = "main"
optional = false
python-versions = ">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*"

[package.extras]
dev = ["pytest", "pytest-timeout", "coverage", "tox", "sphinx", "pallets-sphinx-themes", "sphinx-issues"]
watchdog = ["watchdog"]

[[package]]
name = "zipp"
version = "3.4.0"
description = "Backport of pathlib-compatible object wrapper for zip files"
category = "main"
optional = false
python-versions = ">=3.6"

[package.extras]
docs = ["sphinx", "jaraco.packaging (>=3.2)", "rst.linker (>=1.9)"]
testing = ["pytest (>=3.5,!=3.7.3)", "pytest-checkdocs (>=1.2.3)", "pytest-flake8", "pytest-cov", "jaraco.test (>=3.2.0)", "jaraco.itertools", "func-timeout", "pytest-black (>=0.3.7)", "pytest-mypy"]

[metadata]
lock-version = "1.1"
python-versions = "^3.7"
content-hash = "85fe3554277695c13f09f5c8702b12d44f87e810ef4c768b9dc18f882f09a36d"

[metadata.files]
absl-py = [
    {file = "absl-py-0.11.0.tar.gz", hash = "sha256:673cccb88d810e5627d0c1c818158485d106f65a583880e2f730c997399bcfa7"},
    {file = "absl_py-0.11.0-py3-none-any.whl", hash = "sha256:b3d9eb5119ff6e0a0125f6dabf2f9fae02f8acae7be70576002fac27235611c5"},
]
cachetools = [
    {file = "cachetools-4.1.1-py3-none-any.whl", hash = "sha256:513d4ff98dd27f85743a8dc0e92f55ddb1b49e060c2d5961512855cda2c01a98"},
    {file = "cachetools-4.1.1.tar.gz", hash = "sha256:bbaa39c3dede00175df2dc2b03d0cf18dd2d32a7de7beb68072d13043c9edb20"},
]
certifi = [
    {file = "certifi-2020.11.8-py2.py3-none-any.whl", hash = "sha256:1f422849db327d534e3d0c5f02a263458c3955ec0aae4ff09b95f195c59f4edd"},
    {file = "certifi-2020.11.8.tar.gz", hash = "sha256:f05def092c44fbf25834a51509ef6e631dc19765ab8a57b4e7ab85531f0a9cf4"},
]
chardet = [
    {file = "chardet-3.0.4-py2.py3-none-any.whl", hash = "sha256:fc323ffcaeaed0e0a02bf4d117757b98aed530d9ed4531e3e15460124c106691"},
    {file = "chardet-3.0.4.tar.gz", hash = "sha256:84ab92ed1c4d4f16916e05906b6b75a6c0fb5db821cc65e70cbd64a3e2a5eaae"},
]
einops = [
    {file = "einops-0.3.0-py2.py3-none-any.whl", hash = "sha256:a91c6190ceff7d513d74ca9fd701dfa6a1ffcdd98ea0ced14350197c07f75c73"},
    {file = "einops-0.3.0.tar.gz", hash = "sha256:a3b0935a4556f012cd5fa1851373f63366890a3f6698d117afea55fd2a40c1fc"},
]
google-auth = [
    {file = "google-auth-1.23.0.tar.gz", hash = "sha256:5176db85f1e7e837a646cd9cede72c3c404ccf2e3373d9ee14b2db88febad440"},
    {file = "google_auth-1.23.0-py2.py3-none-any.whl", hash = "sha256:b728625ff5dfce8f9e56a499c8a4eb51443a67f20f6d28b67d5774c310ec4b6b"},
]
google-auth-oauthlib = [
    {file = "google-auth-oauthlib-0.4.2.tar.gz", hash = "sha256:65b65bc39ad8cab15039b35e5898455d3d66296d0584d96fe0e79d67d04c51d9"},
    {file = "google_auth_oauthlib-0.4.2-py2.py3-none-any.whl", hash = "sha256:d4d98c831ea21d574699978827490a41b94f05d565c617fe1b420e88f1fc8d8d"},
]
grpcio = [
    {file = "grpcio-1.33.2-cp27-cp27m-macosx_10_9_x86_64.whl", hash = "sha256:c5030be8a60fb18de1fc8d93d130d57e4296c02f229200df814f6578da00429e"},
    {file = "grpcio-1.33.2-cp27-cp27m-manylinux2010_i686.whl", hash = "sha256:5b21d3de520a699cb631cfd3a773a57debeb36b131be366bf832153405cc5404"},
    {file = "grpcio-1.33.2-cp27-cp27m-manylinux2010_x86_64.whl", hash = "sha256:b412f43c99ca72769306293ba83811b241d41b62ca8f358e47e0fdaf7b6fbbd7"},
    {file = "grpcio-1.33.2-cp27-cp27m-win32.whl", hash = "sha256:703da25278ee7318acb766be1c6d3b67d392920d002b2d0304e7f3431b74f6c1"},
    {file = "grpcio-1.33.2-cp27-cp27m-win_amd64.whl", hash = "sha256:2f2eabfd514af8945ee415083a0f849eea6cb3af444999453bb6666fadc10f54"},
    {file = "grpcio-1.33.2-cp27-cp27mu-linux_armv7l.whl", hash = "sha256:d51ddfb3d481a6a3439db09d4b08447fb9f6b60d862ab301238f37bea8f60a6d"},
    {file = "grpcio-1.33.2-cp27-cp27mu-manylinux2010_i686.whl", hash = "sha256:407b4d869ce5c6a20af5b96bb885e3ecaf383e3fb008375919eb26cf8f10d9cd"},
    {file = "grpcio-1.33.2-cp27-cp27mu-manylinux2010_x86_64.whl", hash = "sha256:abaf30d18874310d4439a23a0afb6e4b5709c4266966401de7c4ae345cc810ee"},
    {file = "grpcio-1.33.2-cp35-cp35m-linux_armv7l.whl", hash = "sha256:f2673c51e8535401c68806d331faba614bcff3ee16373481158a2e74f510b7f6"},
    {file = "grpcio-1.33.2-cp35-cp35m-macosx_10_7_intel.whl", hash = "sha256:65b06fa2db2edd1b779f9b256e270f7a58d60e40121660d8b5fd6e8b88f122ed"},
    {file = "grpcio-1.33.2-cp35-cp35m-manylinux2010_i686.whl", hash = "sha256:514b4a6790d6597fc95608f49f2f13fe38329b2058538095f0502b734b98ffd2"},
    {file = "grpcio-1.33.2-cp35-cp35m-manylinux2010_x86_64.whl", hash = "sha256:4cef3eb2df338abd9b6164427ede961d351c6bf39b4a01448a65f9e795f56575"},
    {file = "grpcio-1.33.2-cp35-cp35m-manylinux2014_i686.whl", hash = "sha256:3ac453387add933b6cfbc67cc8635f91ff9895299130fc612c3c4b904e91d82a"},
    {file = "grpcio-1.33.2-cp35-cp35m-manylinux2014_x86_64.whl", hash = "sha256:7d292dabf7ded9c062357f8207e20e94095a397d487ffd25aa213a2c3dff0ab4"},
    {file = "grpcio-1.33.2-cp35-cp35m-win32.whl", hash = "sha256:0aeed3558a0eec0b31700af6072f1c90e8fd5701427849e76bc469554a14b4f5"},
    {file = "grpcio-1.33.2-cp35-cp35m-win_amd64.whl", hash = "sha256:88f2a102cbc67e91f42b4323cec13348bf6255b25f80426088079872bd4f3c5c"},
    {file = "grpcio-1.33.2-cp36-cp36m-linux_armv7l.whl", hash = "sha256:affbb739fde390710190e3540acc9f3e65df25bd192cc0aa554f368288ee0ea2"},
    {file = "grpcio-1.33.2-cp36-cp36m-macosx_10_9_x86_64.whl", hash = "sha256:ffec0b854d2ed6ee98776c7168c778cdd18503642a68d36c00ba0f96d4ccff7c"},
    {file = "grpcio-1.33.2-cp36-cp36m-manylinux2010_i686.whl", hash = "sha256:7744468ee48be3265db798f27e66e118c324d7831a34fd39d5775bcd5a70a2c4"},
    {file = "grpcio-1.33.2-cp36-cp36m-manylinux2010_x86_64.whl", hash = "sha256:6a1b5b7e47600edcaeaa42983b1c19e7a5892c6b98bcde32ae2aa509a99e0436"},
    {file = "grpcio-1.33.2-cp36-cp36m-manylinux2014_i686.whl", hash = "sha256:289671cfe441069f617bf23c41b1fa07053a31ff64de918d1016ac73adda2f73"},
    {file = "grpcio-1.33.2-cp36-cp36m-manylinux2014_x86_64.whl", hash = "sha256:a8c84db387907e8d800c383e4c92f39996343adedf635ae5206a684f94df8311"},
    {file = "grpcio-1.33.2-cp36-cp36m-win32.whl", hash = "sha256:4bb771c4c2411196b778871b519c7e12e87f3fa72b0517b22f952c64ead07958"},
    {file = "grpcio-1.33.2-cp36-cp36m-win_amd64.whl", hash = "sha256:b581ddb8df619402c377c81f186ad7f5e2726ad9f8d57047144b352f83f37522"},
    {file = "grpcio-1.33.2-cp37-cp37m-macosx_10_9_x86_64.whl", hash = "sha256:02a4a637a774382d6ac8e65c0a7af4f7f4b9704c980a0a9f4f7bbc1e97c5b733"},
    {file = "grpcio-1.33.2-cp37-cp37m-manylinux2010_i686.whl", hash = "sha256:592656b10528aa327058d2007f7ab175dc9eb3754b289e24cac36e09129a2f6b"},
    {file = "grpcio-1.33.2-cp37-cp37m-manylinux2010_x86_64.whl", hash = "sha256:c89510381cbf8c8317e14e747a8b53988ad226f0ed240824064a9297b65f921d"},
    {file = "grpcio-1.33.2-cp37-cp37m-manylinux2014_i686.whl", hash = "sha256:7fda62846ef8d86caf06bd1ecfddcae2c7e59479a4ee28808120e170064d36cc"},
    {file = "grpcio-1.33.2-cp37-cp37m-manylinux2014_x86_64.whl", hash = "sha256:d386630af995fd4de225d550b6806507ca09f5a650f227fddb29299335cda55e"},
    {file = "grpcio-1.33.2-cp37-cp37m-win32.whl", hash = "sha256:bf7de9e847d2d14a0efcd48b290ee181fdbffb2ae54dfa2ec2a935a093730bac"},
    {file = "grpcio-1.33.2-cp37-cp37m-win_amd64.whl", hash = "sha256:7c1ea6ea6daa82031af6eb5b7d1ab56b1193840389ea7cf46d80e98636f8aff5"},
    {file = "grpcio-1.33.2-cp38-cp38-macosx_10_9_x86_64.whl", hash = "sha256:85e56ab125b35b1373205b3802f58119e70ffedfe0d7e2821999126058f7c44f"},
    {file = "grpcio-1.33.2-cp38-cp38-manylinux2010_i686.whl", hash = "sha256:0cebba3907441d5c620f7b491a780ed155140fbd590da0886ecfb1df6ad947b9"},
    {file = "grpcio-1.33.2-cp38-cp38-manylinux2010_x86_64.whl", hash = "sha256:52143467237bfa77331ed1979dc3e203a1c12511ee37b3ddd9ff41b05804fb10"},
    {file = "grpcio-1.33.2-cp38-cp38-manylinux2014_i686.whl", hash = "sha256:8cf67b8493bff50fa12b4bc30ab40ce1f1f216eb54145962b525852959b0ab3d"},
    {file = "grpcio-1.33.2-cp38-cp38-manylinux2014_x86_64.whl", hash = "sha256:fa78bd55ec652d4a88ba254c8dae623c9992e2ce647bd17ba1a37ca2b7b42222"},
    {file = "grpcio-1.33.2-cp38-cp38-win32.whl", hash = "sha256:143b4fe72c01000fc0667bf62ace402a6518939b3511b3c2bec04d44b1d7591c"},
    {file = "grpcio-1.33.2-cp38-cp38-win_amd64.whl", hash = "sha256:08b6a58c8a83e71af5650f8f879fe14b7b84dce0c4969f3817b42c72989dacf0"},
    {file = "grpcio-1.33.2-cp39-cp39-macosx_10_9_x86_64.whl", hash = "sha256:56e2a985efdba8e2282e856470b684e83a3cadd920f04fcd360b4b826ced0dd3"},
    {file = "grpcio-1.33.2-cp39-cp39-manylinux2010_i686.whl", hash = "sha256:62ce7e86f11e8c4ff772e63c282fb5a7904274258be0034adf37aa679cf96ba0"},
    {file = "grpcio-1.33.2-cp39-cp39-manylinux2010_x86_64.whl", hash = "sha256:7f727b8b6d9f92fcab19dbc62ec956d8352c6767b97b8ab18754b2dfa84d784f"},
    {file = "grpcio-1.33.2-cp39-cp39-manylinux2014_i686.whl", hash = "sha256:2d5124284f9d29e4f06f674a12ebeb23fc16ce0f96f78a80a6036930642ae5ab"},
    {file = "grpcio-1.33.2-cp39-cp39-manylinux2014_x86_64.whl", hash = "sha256:eff55d318a114742ed2a06972f5daacfe3d5ad0c0c0d9146bcaf10acb427e6be"},
    {file = "grpcio-1.33.2-cp39-cp39-win32.whl", hash = "sha256:dd47fac2878f6102efa211461eb6fa0a6dd7b4899cd1ade6cdcb9fa9748363eb"},
    {file = "grpcio-1.33.2-cp39-cp39-win_amd64.whl", hash = "sha256:89add4f4cda9546f61cb8a6988bc5b22101dd8ca4af610dff6f28105d1f78695"},
    {file = "grpcio-1.33.2.tar.gz", hash = "sha256:21265511880056d19ce4f809ce3fbe2a3fa98ec1fc7167dbdf30a80d3276202e"},
]
idna = [
    {file = "idna-2.10-py2.py3-none-any.whl", hash = "sha256:b97d804b1e9b523befed77c48dacec60e6dcb0b5391d57af6a65a312a90648c0"},
    {file = "idna-2.10.tar.gz", hash = "sha256:b307872f855b18632ce0c21c5e45be78c0ea7ae4c15c828c20788b26921eb3f6"},
]
importlib-metadata = [
    {file = "importlib_metadata-3.1.0-py2.py3-none-any.whl", hash = "sha256:590690d61efdd716ff82c39ca9a9d4209252adfe288a4b5721181050acbd4175"},
    {file = "importlib_metadata-3.1.0.tar.gz", hash = "sha256:d9b8a46a0885337627a6430db287176970fff18ad421becec1d64cfc763c2099"},
]
joblib = [
    {file = "joblib-0.17.0-py3-none-any.whl", hash = "sha256:698c311779f347cf6b7e6b8a39bb682277b8ee4aba8cf9507bc0cf4cd4737b72"},
    {file = "joblib-0.17.0.tar.gz", hash = "sha256:9e284edd6be6b71883a63c9b7f124738a3c16195513ad940eae7e3438de885d5"},
]
markdown = [
    {file = "Markdown-3.3.3-py3-none-any.whl", hash = "sha256:c109c15b7dc20a9ac454c9e6025927d44460b85bd039da028d85e2b6d0bcc328"},
    {file = "Markdown-3.3.3.tar.gz", hash = "sha256:5d9f2b5ca24bc4c7a390d22323ca4bad200368612b5aaa7796babf971d2b2f18"},
]
numpy = [
    {file = "numpy-1.19.4-cp36-cp36m-macosx_10_9_x86_64.whl", hash = "sha256:e9b30d4bd69498fc0c3fe9db5f62fffbb06b8eb9321f92cc970f2969be5e3949"},
    {file = "numpy-1.19.4-cp36-cp36m-manylinux1_i686.whl", hash = "sha256:fedbd128668ead37f33917820b704784aff695e0019309ad446a6d0b065b57e4"},
    {file = "numpy-1.19.4-cp36-cp36m-manylinux1_x86_64.whl", hash = "sha256:8ece138c3a16db8c1ad38f52eb32be6086cc72f403150a79336eb2045723a1ad"},
    {file = "numpy-1.19.4-cp36-cp36m-manylinux2010_i686.whl", hash = "sha256:64324f64f90a9e4ef732be0928be853eee378fd6a01be21a0a8469c4f2682c83"},
    {file = "numpy-1.19.4-cp36-cp36m-manylinux2010_x86_64.whl", hash = "sha256:ad6f2ff5b1989a4899bf89800a671d71b1612e5ff40866d1f4d8bcf48d4e5764"},
    {file = "numpy-1.19.4-cp36-cp36m-manylinux2014_aarch64.whl", hash = "sha256:d6c7bb82883680e168b55b49c70af29b84b84abb161cbac2800e8fcb6f2109b6"},
    {file = "numpy-1.19.4-cp36-cp36m-win32.whl", hash = "sha256:13d166f77d6dc02c0a73c1101dd87fdf01339febec1030bd810dcd53fff3b0f1"},
    {file = "numpy-1.19.4-cp36-cp36m-win_amd64.whl", hash = "sha256:448ebb1b3bf64c0267d6b09a7cba26b5ae61b6d2dbabff7c91b660c7eccf2bdb"},
    {file = "numpy-1.19.4-cp37-cp37m-macosx_10_9_x86_64.whl", hash = "sha256:27d3f3b9e3406579a8af3a9f262f5339005dd25e0ecf3cf1559ff8a49ed5cbf2"},
    {file = "numpy-1.19.4-cp37-cp37m-manylinux1_i686.whl", hash = "sha256:16c1b388cc31a9baa06d91a19366fb99ddbe1c7b205293ed072211ee5bac1ed2"},
    {file = "numpy-1.19.4-cp37-cp37m-manylinux1_x86_64.whl", hash = "sha256:e5b6ed0f0b42317050c88022349d994fe72bfe35f5908617512cd8c8ef9da2a9"},
    {file = "numpy-1.19.4-cp37-cp37m-manylinux2010_i686.whl", hash = "sha256:18bed2bcb39e3f758296584337966e68d2d5ba6aab7e038688ad53c8f889f757"},
    {file = "numpy-1.19.4-cp37-cp37m-manylinux2010_x86_64.whl", hash = "sha256:fe45becb4c2f72a0907c1d0246ea6449fe7a9e2293bb0e11c4e9a32bb0930a15"},
    {file = "numpy-1.19.4-cp37-cp37m-manylinux2014_aarch64.whl", hash = "sha256:6d7593a705d662be5bfe24111af14763016765f43cb6923ed86223f965f52387"},
    {file = "numpy-1.19.4-cp37-cp37m-win32.whl", hash = "sha256:6ae6c680f3ebf1cf7ad1d7748868b39d9f900836df774c453c11c5440bc15b36"},
    {file = "numpy-1.19.4-cp37-cp37m-win_amd64.whl", hash = "sha256:9eeb7d1d04b117ac0d38719915ae169aa6b61fca227b0b7d198d43728f0c879c"},
    {file = "numpy-1.19.4-cp38-cp38-macosx_10_9_x86_64.whl", hash = "sha256:cb1017eec5257e9ac6209ac172058c430e834d5d2bc21961dceeb79d111e5909"},
    {file = "numpy-1.19.4-cp38-cp38-manylinux1_i686.whl", hash = "sha256:edb01671b3caae1ca00881686003d16c2209e07b7ef8b7639f1867852b948f7c"},
    {file = "numpy-1.19.4-cp38-cp38-manylinux1_x86_64.whl", hash = "sha256:f29454410db6ef8126c83bd3c968d143304633d45dc57b51252afbd79d700893"},
    {file = "numpy-1.19.4-cp38-cp38-manylinux2010_i686.whl", hash = "sha256:ec149b90019852266fec2341ce1db513b843e496d5a8e8cdb5ced1923a92faab"},
    {file = "numpy-1.19.4-cp38-cp38-manylinux2010_x86_64.whl", hash = "sha256:1aeef46a13e51931c0b1cf8ae1168b4a55ecd282e6688fdb0a948cc5a1d5afb9"},
    {file = "numpy-1.19.4-cp38-cp38-manylinux2014_aarch64.whl", hash = "sha256:08308c38e44cc926bdfce99498b21eec1f848d24c302519e64203a8da99a97db"},
    {file = "numpy-1.19.4-cp38-cp38-win32.whl", hash = "sha256:5734bdc0342aba9dfc6f04920988140fb41234db42381cf7ccba64169f9fe7ac"},
    {file = "numpy-1.19.4-cp38-cp38-win_amd64.whl", hash = "sha256:09c12096d843b90eafd01ea1b3307e78ddd47a55855ad402b157b6c4862197ce"},
    {file = "numpy-1.19.4-cp39-cp39-macosx_10_9_x86_64.whl", hash = "sha256:e452dc66e08a4ce642a961f134814258a082832c78c90351b75c41ad16f79f63"},
    {file = "numpy-1.19.4-cp39-cp39-manylinux1_i686.whl", hash = "sha256:a5d897c14513590a85774180be713f692df6fa8ecf6483e561a6d47309566f37"},
    {file = "numpy-1.19.4-cp39-cp39-manylinux1_x86_64.whl", hash = "sha256:a09f98011236a419ee3f49cedc9ef27d7a1651df07810ae430a6b06576e0b414"},
    {file = "numpy-1.19.4-cp39-cp39-manylinux2010_i686.whl", hash = "sha256:50e86c076611212ca62e5a59f518edafe0c0730f7d9195fec718da1a5c2bb1fc"},
    {file = "numpy-1.19.4-cp39-cp39-manylinux2010_x86_64.whl", hash = "sha256:f0d3929fe88ee1c155129ecd82f981b8856c5d97bcb0d5f23e9b4242e79d1de3"},
    {file = "numpy-1.19.4-cp39-cp39-manylinux2014_aarch64.whl", hash = "sha256:c42c4b73121caf0ed6cd795512c9c09c52a7287b04d105d112068c1736d7c753"},
    {file = "numpy-1.19.4-cp39-cp39-win32.whl", hash = "sha256:8cac8790a6b1ddf88640a9267ee67b1aee7a57dfa2d2dd33999d080bc8ee3a0f"},
    {file = "numpy-1.19.4-cp39-cp39-win_amd64.whl", hash = "sha256:4377e10b874e653fe96985c05feed2225c912e328c8a26541f7fc600fb9c637b"},
    {file = "numpy-1.19.4-pp36-pypy36_pp73-manylinux2010_x86_64.whl", hash = "sha256:2a2740aa9733d2e5b2dfb33639d98a64c3b0f24765fed86b0fd2aec07f6a0a08"},
    {file = "numpy-1.19.4.zip", hash = "sha256:141ec3a3300ab89c7f2b0775289954d193cc8edb621ea05f99db9cb181530512"},
]
oauthlib = [
    {file = "oauthlib-3.1.0-py2.py3-none-any.whl", hash = "sha256:df884cd6cbe20e32633f1db1072e9356f53638e4361bef4e8b03c9127c9328ea"},
    {file = "oauthlib-3.1.0.tar.gz", hash = "sha256:bee41cc35fcca6e988463cacc3bcb8a96224f470ca547e697b604cc697b2f889"},
]
pandas = [
    {file = "pandas-1.1.4-cp36-cp36m-macosx_10_9_x86_64.whl", hash = "sha256:e2b8557fe6d0a18db4d61c028c6af61bfed44ef90e419ed6fadbdc079eba141e"},
    {file = "pandas-1.1.4-cp36-cp36m-manylinux1_i686.whl", hash = "sha256:3aa8e10768c730cc1b610aca688f588831fa70b65a26cb549fbb9f35049a05e0"},
    {file = "pandas-1.1.4-cp36-cp36m-manylinux1_x86_64.whl", hash = "sha256:185cf8c8f38b169dbf7001e1a88c511f653fbb9dfa3e048f5e19c38049e991dc"},
    {file = "pandas-1.1.4-cp36-cp36m-manylinux2014_aarch64.whl", hash = "sha256:0d9a38a59242a2f6298fff45d09768b78b6eb0c52af5919ea9e45965d7ba56d9"},
    {file = "pandas-1.1.4-cp36-cp36m-win32.whl", hash = "sha256:8b4c2055ebd6e497e5ecc06efa5b8aa76f59d15233356eb10dad22a03b757805"},
    {file = "pandas-1.1.4-cp36-cp36m-win_amd64.whl", hash = "sha256:5dac3aeaac5feb1016e94bde851eb2012d1733a222b8afa788202b836c97dad5"},
    {file = "pandas-1.1.4-cp37-cp37m-macosx_10_9_x86_64.whl", hash = "sha256:6d2b5b58e7df46b2c010ec78d7fb9ab20abf1d306d0614d3432e7478993fbdb0"},
    {file = "pandas-1.1.4-cp37-cp37m-manylinux1_i686.whl", hash = "sha256:c681e8fcc47a767bf868341d8f0d76923733cbdcabd6ec3a3560695c69f14a1e"},
    {file = "pandas-1.1.4-cp37-cp37m-manylinux1_x86_64.whl", hash = "sha256:c5a3597880a7a29a31ebd39b73b2c824316ae63a05c3c8a5ce2aea3fc68afe35"},
    {file = "pandas-1.1.4-cp37-cp37m-manylinux2014_aarch64.whl", hash = "sha256:6613c7815ee0b20222178ad32ec144061cb07e6a746970c9160af1ebe3ad43b4"},
    {file = "pandas-1.1.4-cp37-cp37m-win32.whl", hash = "sha256:43cea38cbcadb900829858884f49745eb1f42f92609d368cabcc674b03e90efc"},
    {file = "pandas-1.1.4-cp37-cp37m-win_amd64.whl", hash = "sha256:5378f58172bd63d8c16dd5d008d7dcdd55bf803fcdbe7da2dcb65dbbf322f05b"},
    {file = "pandas-1.1.4-cp38-cp38-macosx_10_9_x86_64.whl", hash = "sha256:a7d2547b601ecc9a53fd41561de49a43d2231728ad65c7713d6b616cd02ddbed"},
    {file = "pandas-1.1.4-cp38-cp38-manylinux1_i686.whl", hash = "sha256:41746d520f2b50409dffdba29a15c42caa7babae15616bcf80800d8cfcae3d3e"},
    {file = "pandas-1.1.4-cp38-cp38-manylinux1_x86_64.whl", hash = "sha256:a15653480e5b92ee376f8458197a58cca89a6e95d12cccb4c2d933df5cecc63f"},
    {file = "pandas-1.1.4-cp38-cp38-manylinux2014_aarch64.whl", hash = "sha256:5fdb2a61e477ce58d3f1fdf2470ee142d9f0dde4969032edaf0b8f1a9dafeaa2"},
    {file = "pandas-1.1.4-cp38-cp38-win32.whl", hash = "sha256:8a5d7e57b9df2c0a9a202840b2881bb1f7a648eba12dd2d919ac07a33a36a97f"},
    {file = "pandas-1.1.4-cp38-cp38-win_amd64.whl", hash = "sha256:54404abb1cd3f89d01f1fb5350607815326790efb4789be60508f458cdd5ccbf"},
    {file = "pandas-1.1.4-cp39-cp39-macosx_10_9_x86_64.whl", hash = "sha256:112c5ba0f9ea0f60b2cc38c25f87ca1d5ca10f71efbee8e0f1bee9cf584ed5d5"},
    {file = "pandas-1.1.4-cp39-cp39-manylinux1_i686.whl", hash = "sha256:cf135a08f306ebbcfea6da8bf775217613917be23e5074c69215b91e180caab4"},
    {file = "pandas-1.1.4-cp39-cp39-manylinux1_x86_64.whl", hash = "sha256:b1f8111635700de7ac350b639e7e452b06fc541a328cf6193cf8fc638804bab8"},
    {file = "pandas-1.1.4-cp39-cp39-win32.whl", hash = "sha256:09e0503758ad61afe81c9069505f8cb8c1e36ea8cc1e6826a95823ef5b327daf"},
    {file = "pandas-1.1.4-cp39-cp39-win_amd64.whl", hash = "sha256:0a11a6290ef3667575cbd4785a1b62d658c25a2fd70a5adedba32e156a8f1773"},
    {file = "pandas-1.1.4.tar.gz", hash = "sha256:a979d0404b135c63954dea79e6246c45dd45371a88631cdbb4877d844e6de3b6"},
]
protobuf = [
    {file = "protobuf-3.14.0-cp27-cp27m-macosx_10_9_x86_64.whl", hash = "sha256:629b03fd3caae7f815b0c66b41273f6b1900a579e2ccb41ef4493a4f5fb84f3a"},
    {file = "protobuf-3.14.0-cp27-cp27mu-manylinux1_x86_64.whl", hash = "sha256:5b7a637212cc9b2bcf85dd828b1178d19efdf74dbfe1ddf8cd1b8e01fdaaa7f5"},
    {file = "protobuf-3.14.0-cp35-cp35m-macosx_10_9_intel.whl", hash = "sha256:43b554b9e73a07ba84ed6cf25db0ff88b1e06be610b37656e292e3cbb5437472"},
    {file = "protobuf-3.14.0-cp35-cp35m-manylinux1_x86_64.whl", hash = "sha256:5e9806a43232a1fa0c9cf5da8dc06f6910d53e4390be1fa06f06454d888a9142"},
    {file = "protobuf-3.14.0-cp35-cp35m-win32.whl", hash = "sha256:1c51fda1bbc9634246e7be6016d860be01747354ed7015ebe38acf4452f470d2"},
    {file = "protobuf-3.14.0-cp35-cp35m-win_amd64.whl", hash = "sha256:4b74301b30513b1a7494d3055d95c714b560fbb630d8fb9956b6f27992c9f980"},
    {file = "protobuf-3.14.0-cp36-cp36m-macosx_10_9_x86_64.whl", hash = "sha256:86a75477addde4918e9a1904e5c6af8d7b691f2a3f65587d73b16100fbe4c3b2"},
    {file = "protobuf-3.14.0-cp36-cp36m-manylinux1_x86_64.whl", hash = "sha256:ecc33531a213eee22ad60e0e2aaea6c8ba0021f0cce35dbf0ab03dee6e2a23a1"},
    {file = "protobuf-3.14.0-cp36-cp36m-win32.whl", hash = "sha256:72230ed56f026dd664c21d73c5db73ebba50d924d7ba6b7c0d81a121e390406e"},
    {file = "protobuf-3.14.0-cp36-cp36m-win_amd64.whl", hash = "sha256:0fc96785262042e4863b3f3b5c429d4636f10d90061e1840fce1baaf59b1a836"},
    {file = "protobuf-3.14.0-cp37-cp37m-macosx_10_9_x86_64.whl", hash = "sha256:4e75105c9dfe13719b7293f75bd53033108f4ba03d44e71db0ec2a0e8401eafd"},
    {file = "protobuf-3.14.0-cp37-cp37m-manylinux1_x86_64.whl", hash = "sha256:2a7e2fe101a7ace75e9327b9c946d247749e564a267b0515cf41dfe450b69bac"},
    {file = "protobuf-3.14.0-cp37-cp37m-win32.whl", hash = "sha256:b0d5d35faeb07e22a1ddf8dce620860c8fe145426c02d1a0ae2688c6e8ede36d"},
    {file = "protobuf-3.14.0-cp37-cp37m-win_amd64.whl", hash = "sha256:8971c421dbd7aad930c9bd2694122f332350b6ccb5202a8b7b06f3f1a5c41ed5"},
    {file = "protobuf-3.14.0-cp38-cp38-macosx_10_9_x86_64.whl", hash = "sha256:9616f0b65a30851e62f1713336c931fcd32c057202b7ff2cfbfca0fc7d5e3043"},
    {file = "protobuf-3.14.0-cp38-cp38-manylinux1_x86_64.whl", hash = "sha256:22bcd2e284b3b1d969c12e84dc9b9a71701ec82d8ce975fdda19712e1cfd4e00"},
    {file = "protobuf-3.14.0-py2.py3-none-any.whl", hash = "sha256:0e247612fadda953047f53301a7b0407cb0c3cb4ae25a6fde661597a04039b3c"},
    {file = "protobuf-3.14.0.tar.gz", hash = "sha256:1d63eb389347293d8915fb47bee0951c7b5dab522a4a60118b9a18f33e21f8ce"},
]
pyasn1 = [
    {file = "pyasn1-0.4.8-py2.4.egg", hash = "sha256:fec3e9d8e36808a28efb59b489e4528c10ad0f480e57dcc32b4de5c9d8c9fdf3"},
    {file = "pyasn1-0.4.8-py2.5.egg", hash = "sha256:0458773cfe65b153891ac249bcf1b5f8f320b7c2ce462151f8fa74de8934becf"},
    {file = "pyasn1-0.4.8-py2.6.egg", hash = "sha256:5c9414dcfede6e441f7e8f81b43b34e834731003427e5b09e4e00e3172a10f00"},
    {file = "pyasn1-0.4.8-py2.7.egg", hash = "sha256:6e7545f1a61025a4e58bb336952c5061697da694db1cae97b116e9c46abcf7c8"},
    {file = "pyasn1-0.4.8-py2.py3-none-any.whl", hash = "sha256:39c7e2ec30515947ff4e87fb6f456dfc6e84857d34be479c9d4a4ba4bf46aa5d"},
    {file = "pyasn1-0.4.8-py3.1.egg", hash = "sha256:78fa6da68ed2727915c4767bb386ab32cdba863caa7dbe473eaae45f9959da86"},
    {file = "pyasn1-0.4.8-py3.2.egg", hash = "sha256:08c3c53b75eaa48d71cf8c710312316392ed40899cb34710d092e96745a358b7"},
    {file = "pyasn1-0.4.8-py3.3.egg", hash = "sha256:03840c999ba71680a131cfaee6fab142e1ed9bbd9c693e285cc6aca0d555e576"},
    {file = "pyasn1-0.4.8-py3.4.egg", hash = "sha256:7ab8a544af125fb704feadb008c99a88805126fb525280b2270bb25cc1d78a12"},
    {file = "pyasn1-0.4.8-py3.5.egg", hash = "sha256:e89bf84b5437b532b0803ba5c9a5e054d21fec423a89952a74f87fa2c9b7bce2"},
    {file = "pyasn1-0.4.8-py3.6.egg", hash = "sha256:014c0e9976956a08139dc0712ae195324a75e142284d5f87f1a87ee1b068a359"},
    {file = "pyasn1-0.4.8-py3.7.egg", hash = "sha256:99fcc3c8d804d1bc6d9a099921e39d827026409a58f2a720dcdb89374ea0c776"},
    {file = "pyasn1-0.4.8.tar.gz", hash = "sha256:aef77c9fb94a3ac588e87841208bdec464471d9871bd5050a287cc9a475cd0ba"},
]
pyasn1-modules = [
    {file = "pyasn1-modules-0.2.8.tar.gz", hash = "sha256:905f84c712230b2c592c19470d3ca8d552de726050d1d1716282a1f6146be65e"},
    {file = "pyasn1_modules-0.2.8-py2.4.egg", hash = "sha256:0fe1b68d1e486a1ed5473f1302bd991c1611d319bba158e98b106ff86e1d7199"},
    {file = "pyasn1_modules-0.2.8-py2.5.egg", hash = "sha256:fe0644d9ab041506b62782e92b06b8c68cca799e1a9636ec398675459e031405"},
    {file = "pyasn1_modules-0.2.8-py2.6.egg", hash = "sha256:a99324196732f53093a84c4369c996713eb8c89d360a496b599fb1a9c47fc3eb"},
    {file = "pyasn1_modules-0.2.8-py2.7.egg", hash = "sha256:0845a5582f6a02bb3e1bde9ecfc4bfcae6ec3210dd270522fee602365430c3f8"},
    {file = "pyasn1_modules-0.2.8-py2.py3-none-any.whl", hash = "sha256:a50b808ffeb97cb3601dd25981f6b016cbb3d31fbf57a8b8a87428e6158d0c74"},
    {file = "pyasn1_modules-0.2.8-py3.1.egg", hash = "sha256:f39edd8c4ecaa4556e989147ebf219227e2cd2e8a43c7e7fcb1f1c18c5fd6a3d"},
    {file = "pyasn1_modules-0.2.8-py3.2.egg", hash = "sha256:b80486a6c77252ea3a3e9b1e360bc9cf28eaac41263d173c032581ad2f20fe45"},
    {file = "pyasn1_modules-0.2.8-py3.3.egg", hash = "sha256:65cebbaffc913f4fe9e4808735c95ea22d7a7775646ab690518c056784bc21b4"},
    {file = "pyasn1_modules-0.2.8-py3.4.egg", hash = "sha256:15b7c67fabc7fc240d87fb9aabf999cf82311a6d6fb2c70d00d3d0604878c811"},
    {file = "pyasn1_modules-0.2.8-py3.5.egg", hash = "sha256:426edb7a5e8879f1ec54a1864f16b882c2837bfd06eee62f2c982315ee2473ed"},
    {file = "pyasn1_modules-0.2.8-py3.6.egg", hash = "sha256:cbac4bc38d117f2a49aeedec4407d23e8866ea4ac27ff2cf7fb3e5b570df19e0"},
    {file = "pyasn1_modules-0.2.8-py3.7.egg", hash = "sha256:c29a5e5cc7a3f05926aff34e097e84f8589cd790ce0ed41b67aed6857b26aafd"},
]
python-dateutil = [
    {file = "python-dateutil-2.8.1.tar.gz", hash = "sha256:73ebfe9dbf22e832286dafa60473e4cd239f8592f699aa5adaf10050e6e1823c"},
    {file = "python_dateutil-2.8.1-py2.py3-none-any.whl", hash = "sha256:75bb3f31ea686f1197762692a9ee6a7550b59fc6ca3a1f4b5d7e32fb98e2da2a"},
]
pytz = [
    {file = "pytz-2020.4-py2.py3-none-any.whl", hash = "sha256:5c55e189b682d420be27c6995ba6edce0c0a77dd67bfbe2ae6607134d5851ffd"},
    {file = "pytz-2020.4.tar.gz", hash = "sha256:3e6b7dd2d1e0a59084bcee14a17af60c5c562cdc16d828e8eba2e683d3a7e268"},
]
requests = [
    {file = "requests-2.25.0-py2.py3-none-any.whl", hash = "sha256:e786fa28d8c9154e6a4de5d46a1d921b8749f8b74e28bde23768e5e16eece998"},
    {file = "requests-2.25.0.tar.gz", hash = "sha256:7f1a0b932f4a60a1a65caa4263921bb7d9ee911957e0ae4a23a6dd08185ad5f8"},
]
requests-oauthlib = [
    {file = "requests-oauthlib-1.3.0.tar.gz", hash = "sha256:b4261601a71fd721a8bd6d7aa1cc1d6a8a93b4a9f5e96626f8e4d91e8beeaa6a"},
    {file = "requests_oauthlib-1.3.0-py2.py3-none-any.whl", hash = "sha256:7f71572defaecd16372f9006f33c2ec8c077c3cfa6f5911a9a90202beb513f3d"},
    {file = "requests_oauthlib-1.3.0-py3.7.egg", hash = "sha256:fa6c47b933f01060936d87ae9327fead68768b69c6c9ea2109c48be30f2d4dbc"},
]
rsa = [
    {file = "rsa-4.6-py3-none-any.whl", hash = "sha256:6166864e23d6b5195a5cfed6cd9fed0fe774e226d8f854fcb23b7bbef0350233"},
    {file = "rsa-4.6.tar.gz", hash = "sha256:109ea5a66744dd859bf16fe904b8d8b627adafb9408753161e766a92e7d681fa"},
]
scikit-learn = [
    {file = "scikit-learn-0.23.2.tar.gz", hash = "sha256:20766f515e6cd6f954554387dfae705d93c7b544ec0e6c6a5d8e006f6f7ef480"},
    {file = "scikit_learn-0.23.2-cp36-cp36m-macosx_10_9_x86_64.whl", hash = "sha256:98508723f44c61896a4e15894b2016762a55555fbf09365a0bb1870ecbd442de"},
    {file = "scikit_learn-0.23.2-cp36-cp36m-manylinux1_i686.whl", hash = "sha256:a64817b050efd50f9abcfd311870073e500ae11b299683a519fbb52d85e08d25"},
    {file = "scikit_learn-0.23.2-cp36-cp36m-manylinux1_x86_64.whl", hash = "sha256:daf276c465c38ef736a79bd79fc80a249f746bcbcae50c40945428f7ece074f8"},
    {file = "scikit_learn-0.23.2-cp36-cp36m-win32.whl", hash = "sha256:cb3e76380312e1f86abd20340ab1d5b3cc46a26f6593d3c33c9ea3e4c7134028"},
    {file = "scikit_learn-0.23.2-cp36-cp36m-win_amd64.whl", hash = "sha256:0a127cc70990d4c15b1019680bfedc7fec6c23d14d3719fdf9b64b22d37cdeca"},
    {file = "scikit_learn-0.23.2-cp37-cp37m-macosx_10_9_x86_64.whl", hash = "sha256:2aa95c2f17d2f80534156215c87bee72b6aa314a7f8b8fe92a2d71f47280570d"},
    {file = "scikit_learn-0.23.2-cp37-cp37m-manylinux1_i686.whl", hash = "sha256:6c28a1d00aae7c3c9568f61aafeaad813f0f01c729bee4fd9479e2132b215c1d"},
    {file = "scikit_learn-0.23.2-cp37-cp37m-manylinux1_x86_64.whl", hash = "sha256:da8e7c302003dd765d92a5616678e591f347460ac7b53e53d667be7dfe6d1b10"},
    {file = "scikit_learn-0.23.2-cp37-cp37m-win32.whl", hash = "sha256:d9a1ce5f099f29c7c33181cc4386660e0ba891b21a60dc036bf369e3a3ee3aec"},
    {file = "scikit_learn-0.23.2-cp37-cp37m-win_amd64.whl", hash = "sha256:914ac2b45a058d3f1338d7736200f7f3b094857758895f8667be8a81ff443b5b"},
    {file = "scikit_learn-0.23.2-cp38-cp38-macosx_10_9_x86_64.whl", hash = "sha256:7671bbeddd7f4f9a6968f3b5442dac5f22bf1ba06709ef888cc9132ad354a9ab"},
    {file = "scikit_learn-0.23.2-cp38-cp38-manylinux1_i686.whl", hash = "sha256:d0dcaa54263307075cb93d0bee3ceb02821093b1b3d25f66021987d305d01dce"},
    {file = "scikit_learn-0.23.2-cp38-cp38-manylinux1_x86_64.whl", hash = "sha256:5ce7a8021c9defc2b75620571b350acc4a7d9763c25b7593621ef50f3bd019a2"},
    {file = "scikit_learn-0.23.2-cp38-cp38-win32.whl", hash = "sha256:0d39748e7c9669ba648acf40fb3ce96b8a07b240db6888563a7cb76e05e0d9cc"},
    {file = "scikit_learn-0.23.2-cp38-cp38-win_amd64.whl", hash = "sha256:1b8a391de95f6285a2f9adffb7db0892718950954b7149a70c783dc848f104ea"},
]
scipy = [
    {file = "scipy-1.5.4-cp36-cp36m-macosx_10_9_x86_64.whl", hash = "sha256:4f12d13ffbc16e988fa40809cbbd7a8b45bc05ff6ea0ba8e3e41f6f4db3a9e47"},
    {file = "scipy-1.5.4-cp36-cp36m-manylinux1_i686.whl", hash = "sha256:a254b98dbcc744c723a838c03b74a8a34c0558c9ac5c86d5561703362231107d"},
    {file = "scipy-1.5.4-cp36-cp36m-manylinux1_x86_64.whl", hash = "sha256:368c0f69f93186309e1b4beb8e26d51dd6f5010b79264c0f1e9ca00cd92ea8c9"},
    {file = "scipy-1.5.4-cp36-cp36m-manylinux2014_aarch64.whl", hash = "sha256:4598cf03136067000855d6b44d7a1f4f46994164bcd450fb2c3d481afc25dd06"},
    {file = "scipy-1.5.4-cp36-cp36m-win32.whl", hash = "sha256:e98d49a5717369d8241d6cf33ecb0ca72deee392414118198a8e5b4c35c56340"},
    {file = "scipy-1.5.4-cp36-cp36m-win_amd64.whl", hash = "sha256:65923bc3809524e46fb7eb4d6346552cbb6a1ffc41be748535aa502a2e3d3389"},
    {file = "scipy-1.5.4-cp37-cp37m-macosx_10_9_x86_64.whl", hash = "sha256:9ad4fcddcbf5dc67619379782e6aeef41218a79e17979aaed01ed099876c0e62"},
    {file = "scipy-1.5.4-cp37-cp37m-manylinux1_i686.whl", hash = "sha256:f87b39f4d69cf7d7529d7b1098cb712033b17ea7714aed831b95628f483fd012"},
    {file = "scipy-1.5.4-cp37-cp37m-manylinux1_x86_64.whl", hash = "sha256:25b241034215247481f53355e05f9e25462682b13bd9191359075682adcd9554"},
    {file = "scipy-1.5.4-cp37-cp37m-manylinux2014_aarch64.whl", hash = "sha256:fa789583fc94a7689b45834453fec095245c7e69c58561dc159b5d5277057e4c"},
    {file = "scipy-1.5.4-cp37-cp37m-win32.whl", hash = "sha256:d6d25c41a009e3c6b7e757338948d0076ee1dd1770d1c09ec131f11946883c54"},
    {file = "scipy-1.5.4-cp37-cp37m-win_amd64.whl", hash = "sha256:2c872de0c69ed20fb1a9b9cf6f77298b04a26f0b8720a5457be08be254366c6e"},
    {file = "scipy-1.5.4-cp38-cp38-macosx_10_9_x86_64.whl", hash = "sha256:e360cb2299028d0b0d0f65a5c5e51fc16a335f1603aa2357c25766c8dab56938"},
    {file = "scipy-1.5.4-cp38-cp38-manylinux1_i686.whl", hash = "sha256:3397c129b479846d7eaa18f999369a24322d008fac0782e7828fa567358c36ce"},
    {file = "scipy-1.5.4-cp38-cp38-manylinux1_x86_64.whl", hash = "sha256:168c45c0c32e23f613db7c9e4e780bc61982d71dcd406ead746c7c7c2f2004ce"},
    {file = "scipy-1.5.4-cp38-cp38-manylinux2014_aarch64.whl", hash = "sha256:213bc59191da2f479984ad4ec39406bf949a99aba70e9237b916ce7547b6ef42"},
    {file = "scipy-1.5.4-cp38-cp38-win32.whl", hash = "sha256:634568a3018bc16a83cda28d4f7aed0d803dd5618facb36e977e53b2df868443"},
    {file = "scipy-1.5.4-cp38-cp38-win_amd64.whl", hash = "sha256:b03c4338d6d3d299e8ca494194c0ae4f611548da59e3c038813f1a43976cb437"},
    {file = "scipy-1.5.4-cp39-cp39-macosx_10_9_x86_64.whl", hash = "sha256:3d5db5d815370c28d938cf9b0809dade4acf7aba57eaf7ef733bfedc9b2474c4"},
    {file = "scipy-1.5.4-cp39-cp39-manylinux1_i686.whl", hash = "sha256:6b0ceb23560f46dd236a8ad4378fc40bad1783e997604ba845e131d6c680963e"},
    {file = "scipy-1.5.4-cp39-cp39-manylinux1_x86_64.whl", hash = "sha256:ed572470af2438b526ea574ff8f05e7f39b44ac37f712105e57fc4d53a6fb660"},
    {file = "scipy-1.5.4-cp39-cp39-manylinux2014_aarch64.whl", hash = "sha256:8c8d6ca19c8497344b810b0b0344f8375af5f6bb9c98bd42e33f747417ab3f57"},
    {file = "scipy-1.5.4-cp39-cp39-win32.whl", hash = "sha256:d84cadd7d7998433334c99fa55bcba0d8b4aeff0edb123b2a1dfcface538e474"},
    {file = "scipy-1.5.4-cp39-cp39-win_amd64.whl", hash = "sha256:cc1f78ebc982cd0602c9a7615d878396bec94908db67d4ecddca864d049112f2"},
    {file = "scipy-1.5.4.tar.gz", hash = "sha256:4a453d5e5689de62e5d38edf40af3f17560bfd63c9c5bd228c18c1f99afa155b"},
]
six = [
    {file = "six-1.15.0-py2.py3-none-any.whl", hash = "sha256:8b74bedcbbbaca38ff6d7491d76f2b06b3592611af620f8426e82dddb04a5ced"},
    {file = "six-1.15.0.tar.gz", hash = "sha256:30639c035cdb23534cd4aa2dd52c3bf48f06e5f4a941509c8bafd8ce11080259"},
]
sklearn = [
    {file = "sklearn-0.0.tar.gz", hash = "sha256:e23001573aa194b834122d2b9562459bf5ae494a2d59ca6b8aa22c85a44c0e31"},
]
tensorboard = [
    {file = "tensorboard-2.4.0-py3-none-any.whl", hash = "sha256:cde0c663a85609441cb4d624e7255fd8e2b6b1d679645095aac8a234a2812738"},
]
tensorboard-plugin-wit = [
    {file = "tensorboard_plugin_wit-1.7.0-py3-none-any.whl", hash = "sha256:ee775f04821185c90d9a0e9c56970ee43d7c41403beb6629385b39517129685b"},
]
threadpoolctl = [
    {file = "threadpoolctl-2.1.0-py3-none-any.whl", hash = "sha256:38b74ca20ff3bb42caca8b00055111d74159ee95c4370882bbff2b93d24da725"},
    {file = "threadpoolctl-2.1.0.tar.gz", hash = "sha256:ddc57c96a38beb63db45d6c159b5ab07b6bced12c45a1f07b2b92f272aebfa6b"},
]
tqdm = [
    {file = "tqdm-4.53.0-py2.py3-none-any.whl", hash = "sha256:5ff3f5232b19fa4c5531641e480b7fad4598819f708a32eb815e6ea41c5fa313"},
    {file = "tqdm-4.53.0.tar.gz", hash = "sha256:3d3f1470d26642e88bd3f73353cb6ff4c51ef7d5d7efef763238f4bc1f7e4e81"},
]
urllib3 = [
    {file = "urllib3-1.26.2-py2.py3-none-any.whl", hash = "sha256:d8ff90d979214d7b4f8ce956e80f4028fc6860e4431f731ea4a8c08f23f99473"},
    {file = "urllib3-1.26.2.tar.gz", hash = "sha256:19188f96923873c92ccb987120ec4acaa12f0461fa9ce5d3d0772bc965a39e08"},
]
werkzeug = [
    {file = "Werkzeug-1.0.1-py2.py3-none-any.whl", hash = "sha256:2de2a5db0baeae7b2d2664949077c2ac63fbd16d98da0ff71837f7d1dea3fd43"},
    {file = "Werkzeug-1.0.1.tar.gz", hash = "sha256:6c80b1e5ad3665290ea39320b91e1be1e0d5f60652b964a3070216de83d2e47c"},
]
zipp = [
    {file = "zipp-3.4.0-py3-none-any.whl", hash = "sha256:102c24ef8f171fd729d46599845e95c7ab894a4cf45f5de11a44cc7444fb1108"},
    {file = "zipp-3.4.0.tar.gz", hash = "sha256:ed5eee1974372595f9e416cc7bbeeb12335201d8081ca8a0743c954d4446e5cb"},
]
END
echo "Voila!"
