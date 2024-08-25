# On server
- **Data**: cd /data/data_WF/NhaBe
- **Code**: cd /data/DanHoangThu

# On server
- **Data**: cd /data/data_WF/NhaBe
- **Code**: cd /data/DanHoangThu

# Docker
## Create image
- **Create base image**: 
docker build -t dht-image-base -f Dockerfile_base .

- **Create image**: 
docker build -t dht-image -f Dockerfile .

docker build -t dht-image1 -f Dockerfile .

## Run image
docker run -v /data/data_WF/NhaBe:/app/data -v /data/DanHoangThu/image:/app/image -v /data/DanHoangThu/result:/app/result -d --shm-size=16g --name dht-cont dht-image

docker run -v /data/DanHoangThu/image:/app/image -v /data/DanHoangThu/result:/app/result -d --shm-size=16g --gpus '"device=6"' --name dht-cont dht-image

docker run -v /data/DanHoangThu/image:/app/image -v /data/DanHoangThu/result:/app/result -d --shm-size=16g --gpus '"device=4"' --name dht-cont1 dht-image1
## Remove image
- docker rmi -f dht-image
## Start container
- docker start dht-cont
## Access container
- docker exec -it dht-cont /bin/bash
- docker logs -f dht-cont
## Copy from container to host
- docker cp dht-cont:/app/metadata.csv .
- docker cp dht-cont:/app/errors.log .
## Stop container
- docker stop dht-cont
## Remove container
- docker rm -f dht-cont

# Out server
- exit

# Other
- **Total RAW files**: 259999
- **Total images files**: 251479