import sys

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.transforms import Compose
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm

from test.models import DummyModel
from test.datasets import DummyDataset
from test.transforms import DummyTransform

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

