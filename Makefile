GPUDRIVE_DIR := "./src/envs/gpudrive"

# Function that builds the gpudrive code
gpudrive_build:
	find ${GPUDRIVE_DIR}/external/* -type d -exec git config --global --add safe.directory {} \;
	uv lock
	uv sync --frozen
	cd ${GPUDRIVE_DIR} && uv run python ./build_gpudrive.py

# Function that runs the docker run command
docker_run:
	docker run --gpus all -it --rm --shm-size=20G \
		-v ${PWD}:/workspace \
		mairlsim-dev:latest \
		/bin/bash 

# Function that builds the docker image
docker_build:
	DOCKER_BUILDKIT=1 docker build --build-arg USE_CUDA=true --tag mairlsim-dev:latest .