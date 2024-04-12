# On server
Data: cd /data/data_WF/NhaBe
Code: cd /data/DanHoangThu

# Docker
## Create image
docker build -t dht-image-base -f Dockerfile_base .
docker build -t dht-image -f Dockerfile .
## Run image (Mount NhaBe and Image folders from host to container)
docker run -v /data/data_WF/NhaBe:/app/data -v /data/DanHoangThu/image:/app/image --name dht-cont dht-image

docker run -v /data/DanHoangThu/image:/app/image --shm-size=16g --gpus '"device=0"' --name dht-cont dht-image
docker run -v /data/DanHoangThu/image:/app/image --shm-size=16g --gpus '"device=1"' --name dht-cont1 dht-image1
## Remove image
docker rmi -f dht-image

## Start container
docker start dht-cont
docker exec -it dht-cont /bin/bash 
## Copy from host to container
docker cp 1.gen_meta.py dht-cont:/app
## Copy from container to host
docker cp dht-cont:/app/metadata.csv .
docker cp dht-cont:/app/metadata_temp.csv .
docker cp dht-cont:/app/metadata_lite.csv .
docker cp dht-cont:/app/metadata .
docker cp dht-cont:/app/model .
docker cp dht-cont:/app/result .
## Find container id
docker ps -aqf "ancestor=dht-image"
## Stop container
docker stop dht-cont
## Remove container
docker rm -f dht-cont

# Out server
exit

# Other
Total RAW files: 259999
Total images files: 251507
Total labeled images files: 250167

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
batch_size = 16
num_epochs = 30
epoch_ratio = 0.5