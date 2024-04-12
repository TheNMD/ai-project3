# On server
Data: cd /data/data_WF/NhaBe
Code: cd /data/DanHoangThu

# Docker
## Create image
docker build -t dht-image-base -f Dockerfile_base .
docker build -t dht-image -f Dockerfile .
## Run image (Mount NhaBe and Image folders from host to container)
docker run -v /data/data_WF/NhaBe:/app/data -v /data/DanHoangThu/image:/app/image --name dht-cont dht-image

docker run -v /data/DanHoangThu/result:/app/result -v /data/DanHoangThu/image:/app/image --shm-size=16g --gpus '"device=2"' --name dht-cont dht-image

docker run -v /data/DanHoangThu/result:/app/result -v /data/DanHoangThu/image:/app/image --shm-size=16g --gpus '"device=3"' --name dht-cont1 dht-image1
## Remove image
docker rmi -f dht-image

## Start container
docker start dht-cont
# Access container
docker exec -it dht-cont /bin/bash
# Access container's terminal output
docker logs -f dht-cont
## Copy from host to container
docker cp 1.gen_meta.py dht-cont:/app
## Copy from container to host
docker cp dht-cont:/app/metadata.csv .
docker cp dht-cont:/app/metadata_temp.csv .
docker cp dht-cont:/app/metadata_lite.csv .
docker cp dht-cont:/app/metadata .
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

# Models
## On server
vit-base
swinv2-tiny
effnetv2-m
convnext-s

## On Colab
vit-base
swinv2-tiny
effnetv2-s
convnext-s
