WORKSPACE_DIR := /workspace
GPUDRIVE_DIR := $(WORKSPACE_DIR)/src/envs/gpudrive

# Function that builds the gpudrive code
gpudrive_build:
	$(MAKE) -C ${GPUDRIVE_DIR} gpudrive_build WORKSPACE_DIR=${GPUDRIVE_DIR}

# Function that runs the docker run command
docker_run:
	docker run --gpus all -it --rm --shm-size=20G \
		-v ${PWD}:/workspace \
		-v ${CARLA_ROOT}:/carla \
		--network host \
		mairlsim-dev:latest \
		/bin/bash

# Function that builds the docker image
docker_build:
	DOCKER_BUILDKIT=1 docker build --build-arg USE_CUDA=true --tag mairlsim-dev:latest .
