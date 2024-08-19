SHELL := /bin/bash
# (C)2024
# Version 1.5
# Written by Joe Cincotta
#
export TORCH_CUDA_ARCH_LIST="12.1"
export MMCV_WITH_OPS=1
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda

help: ## This help
	@echo "BEES0006 Project"
	@echo -e "$$(grep -hE '^\S+:.*##' $(MAKEFILE_LIST) | sed -e 's/:.*##\s*/:/' -e 's/^\(.\+\):\(.*\)/\\x1b[36m\1\\x1b[m:\2/' | column -c2 -t -s :)"

setup: ## run the CONDA environment setup
	bin/env2.sh bees0006
	@echo "Finished"
	@date +"%F %T"

update: ## Update all dependencies [Conda and NPM]
	bin/env2.sh -u bees0006
	@echo "Finished"
	@date +"%F %T"

reset: ## Hard reset all git repos and reinstall Conda environment
	#	git reset --hard HEAD;git pull;git checkout master
	#	git pull
	bin/env2.sh -delete bees0006
	@echo "Finished"
	@date +"%F %T"

jupyter: ## Start Jupyter
	docker build -t bees0006 -f Dockerfile .
	docker run -it --privileged --gpus all --rm -p 8888:8888 -p 8008:8008 -v ./notebooks:/notebooks -v ./projects:/projects --name bees0006-instance bees0006 /api/jupyter.sh bees0006 /notebooks

connect: ## Connect to CUDA Container
	docker exec -it bees0006-instance bash 

build-rgbtseg: ## Build RGBTSeg
	docker build --no-cache --build-arg CONDA_YAML=bees0006-rgbtseg.yml  -t bees0006-rgbtseg -f Dockerfile .

build-cut: ## Build CUT
	docker build --no-cache --build-arg CONDA_YAML=bees0006-cut.yml  -t bees0006-cut -f Dockerfile .

build-tensorboard: ## Build Tensorboard
	docker build --no-cache --build-arg CONDA_YAML=bees0006-tensorboard.yml  -t bees0006-tensorboard -f Dockerfile .

jupyter-rgbtseg: ## Start RGBTSeg Jupyter Environment
	docker run -it --privileged --gpus all --ipc host --rm -p 8888:8888 -v ./notebooks:/notebooks -v ./projects:/projects --name bees0006-rgbtseg-instance bees0006-rgbtseg bash -c "/api/build-ops.sh bees0006-rgbtseg;/api/jupyter.sh bees0006-rgbtseg /projects/CRM_RGBTSeg"

train-rgbtseg: ## Train Jupyter Test Envrionment
	docker run -it --privileged --gpus all --ipc host --rm -p 8888:8888 -v ./notebooks:/notebooks -v ./projects:/projects --name bees0006-rgbtseg-instance bees0006-rgbtseg bash -c "/api/build-ops.sh bees0006-rgbtseg;/api/run_ipynb.sh bees0006-rgbtseg /projects/CRM_RGBTSeg bees0006_rgbtseg.ipynb"

connect-rgbtseg: ## Connect to RGBTSeg Container
	docker exec -it bees0006-rgbtseg-instance bash

tensorboard-rgbtseg: ## Start RGBTSeg tensorboard 
	docker run -it --rm -p 8008:8008 -v ./notebooks:/notebooks -v ./projects:/projects --name bees0006-rgbtseg-tensorboard-instance bees0006-tensorboard bash -c "/api/tensorboard.sh bees0006-tensorboard /projects/CRM_RGBTSeg/checkpoints"

bash-cut: ## BASH CUT
	docker run -it --privileged --gpus all --rm -v ./notebooks:/notebooks -v ./projects:/projects --name bees0006-cut-instance bees0006-cut bash

jupyter-cut: ## Start Jupyter Test Environment
	docker run -it --privileged --gpus all --rm -p 8888:8888 -v ./notebooks:/notebooks -v ./projects:/projects --name bees0006-cut-instance bees0006-cut /api/jupyter.sh bees0006-cut /projects/cut

connect-cut: ## Connect to CUT Container
	docker exec -it bees0006-cut-instance bash


