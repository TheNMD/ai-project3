# On server
Data: cd /data/data_WF/NhaBe
Code: cd /data/DanHoangThu

# Docker
## Create image
docker build -t dataric-server .
## Remove image
docker rmi dataric-server
## Run image
docker run --name srv dataric-server
## Find container id
docker ps -aqf "ancestor=dataric-server"
## Stop container
docker stop <container_id>
## Remove container
docker rm -f srv

# Out server
logout