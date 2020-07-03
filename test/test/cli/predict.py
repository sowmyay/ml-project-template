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
