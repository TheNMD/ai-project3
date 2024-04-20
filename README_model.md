# On server
Data: cd /data/data_WF/NhaBe
Code: cd /data/DanHoangThu

# Docker
## Create image
docker build -t dht-image-base -f Dockerfile_base .

docker build -t dht-image -f Dockerfile .
## Run image (Mount NhaBe and Image folders from host to container)
docker run -v /data/DanHoangThu/result:/app/result -v /data/DanHoangThu/image:/app/image --shm-size=16g --gpus '"device=1"' --name dht-cont dht-image
## Remove image
docker rmi -f dht-image

## Start container
docker start dht-cont
## Access container
docker exec -it dht-cont /bin/bash
## Access container's terminal output
docker logs -f dht-cont
## Stop container
docker stop dht-cont
## Remove container
docker rm -f dht-cont

# Out server
exit

# Other
Total RAW files: 259999
Total images files: 251486

Total labeled images files (0):     250167
Total labeled images files (7200):  250167
Total labeled images files (21600): 250167
Total labeled images files (43200): 250167