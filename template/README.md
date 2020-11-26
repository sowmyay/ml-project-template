## template

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
```bash
poetry run template train --dataset /data/train \
--model /data/models \
--num-workers 12 \
--batch-size 512 \
--num-epochs 100
```

## Prediction
```bash
poetry run template predict --dataset /data/predict \
--checkpoint /data/models/best-checkpoint.pth \
--num-workers 12 \
--batch-size 512
```

## References

## License
