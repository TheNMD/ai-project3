# On server
Data: cd /data/data_WF/NhaBe
Code: cd /data/DanHoangThu

# Docker
## Create image
docker build -t dht-image -f Dockerfile .
## Run image (Mount NhaBe and Image folders from host to container)
docker run -v /data/DanHoangThu/image:/app/image -v /data/DanHoangThu/image:/app/model --shm-size=16g --memory=16g --gpus '"device=0"' --name dht-cont dht-image
## Remove image
docker rmi -f dht-image

## Start container
docker start dht-cont
docker exec -it dht-cont /bin/bash 
## Copy from host to container
docker cp 1.gen_meta.py dht-cont:/app
## Copy from container to host
docker cp dht-cont:/app/result .
## Stop container
docker stop dht-cont
## Remove container
docker rm -f dht-cont

# Hyerparameter
## VIT
## For model
model_name = "vit"
option = "pretrained"
checkpoint = False

## For optimizer
learning_rate = 1e-4
optimizer = "adam"

## For callbacks
patience = 5
min_delta = 1e-3

## For training loop
batch_size = 64
num_epochs = 20
epoch_ratio = 0.5