dockerimage ?= pytorch_template
dockerfile ?= Dockerfile.cpu
srcdir ?= $(shell pwd)
datadir ?= $(shell pwd)

install:
	@docker build -t $(dockerimage) -f $(dockerfile) .

i: install


update:
	@docker build -t $(dockerimage) -f $(dockerfile) . --pull --no-cache

u: update


run:
	@docker run --ipc=host -it --rm -v $(srcdir):/home/python/app/ -v $(datadir):/data --entrypoint=/bin/bash $(dockerimage)

r: run

jupyter:
	@docker run --ipc=host -it --rm -p 8888:8888 -v $(srcdir):/home/python/app/ -v $(datadir):/data --entrypoint=/home/python/app/jupyter.sh $(dockerimage)

j: jupyter

tensorboard:
	@docker run -it --rm -p 6006:6006 -v $(datadir):/data tensorflow/tensorflow tensorboard --bind_all --logdir /data

t: tensorboard

gpu:
	@docker run --gpus all --ipc=host -it --rm -v $(srcdir):/home/python/app/ -v $(datadir):/data --entrypoint=/bin/bash $(dockerimage)

g: gpu

jupyter-gpu:
	@docker run --gpus all --ipc=host -it --rm -p 8888:8888 -v $(srcdir):/home/python/app/ -v $(datadir):/data --entrypoint=/home/python/app/jupyter.sh $(dockerimage)

jg: jupyter-gpu

publish:
	@docker image save $(dockerimage) 	  | pv -N "Publish $(dockerimage) to $(sshopts)" -s $(shell docker image inspect $(dockerimage) --format "{{.Size}}") 	  | ssh $(sshopts) "docker image load"

p: publish

.PHONY: install i run r update u gpu g publish p tensorboard t jupyter j jupyter-gpu jg

