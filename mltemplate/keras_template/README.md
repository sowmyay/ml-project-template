## pytorch_template

## Usage

## Development

We provide CPU and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) based GPU Dockerfiles for self-contained and reproducible environments.
Use the convenience Makefile to build the Docker image and then get into the container mounting a host directory to `/data` inside the container:

```
make
make run datadir=/path/to/dataset
```

By default we build and run the CPU Docker images; for GPUs run:

```
make dockerfile=Dockerfile.gpu
make gpu datadir=/path/to/dataset
```

## Preprocessing


## Model training
Inside your docker container, run -
```bash
python3 -m keras_template train --dataset /data/train \
--model /data/models \
--num-workers 12 \
--batch-size 512 \
--num-epochs 100
```

## Prediction
Inside your docker container, run -
```bash
python3 -m keras_template predict --dataset /data/predict \
--checkpoint /data/models/best-checkpoint.pth \
--num-workers 12 \
--batch-size 512
```

## Jupyter notebook
To start jupyter environment run the following command from your terminal
```bash
make jupyter
```
or
```bash
make jupyter-gpu
```

## Monitor Tensorboard
To monitor the loss plots on tensorboard, run the following command from your terminal-
```bash
make tensorboard datadir=/path/to/runs_directory
```
Go to `localhost:6006` in your browser to monitor the tensorboard plots

## References

## License
Copyright © 2020  

Distributed under the MIT License (MIT).
