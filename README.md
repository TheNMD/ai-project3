# On server
Data: cd /data/data_WF/NhaBe
Code: cd /data/DanHoangThu

# Docker
## Create image
docker build -t dht-image .
## Run image (Mount NhaBe folder from host to container)
docker run -v /data/data_WF/NhaBe:/app/data --name dht-cont dht-image
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
docker cp -rn dht-cont:/app/image .
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
Total non-error images files: 251487
