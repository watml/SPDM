# File contains a shell command to creat a docker container using the provided docker 
# image <named> with some set gpu paramters and mounted folders.
#


######################################################################################
# Command to build docker image from Dockerfile                                      #
######################################################################################
docker build -t SPDM:latest /SPDM/docker/Dockerfile 


######################################################################################
# Command to create container from existing image with gpu access and mounted drives #
######################################################################################
docker container run --gpus device=all  --shm-size 24GB --restart=always -it -d  -v /home/SPDM/:/home/SPDM -v /home/datasets:/home/datasets -v /home/checkpoints:/home/checkpoints SPDM /bin/bash

docker container run --gpus device=all  --shm-size 24GB --restart=always -it -d  -v /home/SPDM/:/home/SPDM -v /home/datasets:/home/datasets -v /home/checkpoints:/home/checkpoints SPDM /bin/bash
