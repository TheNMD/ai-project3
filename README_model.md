# On server
Data: cd /data/data_WF/NhaBe
Code: cd /data/DanHoangThu

# Docker
## Create image
docker build -t dht-image -f Dockerfile .
## Run image (Mount NhaBe and Image folders from host to container)
docker run -v /data/DanHoangThu/image:/app/image --shm-size=16g --gpus '"device=0"' --name dht-cont dht-image
docker run -v /data/DanHoangThu/image:/app/image --shm-size=16g --gpus '"device=1"' --name dht-cont1 dht-image1
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
docker cp dht-cont:/app/result/checkpoint/. result/checkpoint
## Stop container
docker stop dht-cont
## Remove container
docker rm -f dht-cont