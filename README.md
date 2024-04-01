# Docker
# On server
Data: cd /data/data_WF/NhaBe
Code: cd /data/DanHoangThu

# Docker
## Create image
docker build -t dht-image .
## Remove image
docker rmi -f dht-model
## Run image (Mount NhaBe folder from host to container)
docker run -v /data/data_WF/NhaBe:/app/data --name dht-cont dht-image
## Find container id
docker ps -aqf "ancestor=dht-image"
## Stop container
docker stop dht-cont
## Remove container
docker rm -f dht-cont

# Out server
logout