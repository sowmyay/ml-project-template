## test

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
make gpu
```

## Preprocessing


## Model training


## Prediction

## References

## License
