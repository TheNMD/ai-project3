# Docker
# On server
Data: cd /data/data_WF/NhaBe
Code: cd /data/DanHoangThu

## Create image
docker build -t dataric-server .
## Remove image
docker rmi -f dataric-server
## Run image (Mount NhaBe folder from host to container)
docker run -v /data/data_WF/NhaBe:/app/data --name srv dataric-server
## Find container id
docker ps -aqf "ancestor=dataric-server"
## Stop container
docker stop -f srv
## Remove container
docker rm -f srv

# Out server
logout