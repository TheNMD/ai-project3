FROM dht-image-base

WORKDIR /app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

RUN apt-get update && apt-get install -y iputils-ping
RUN apt-get update && apt-get install -y dnsutils

# Make some required folders inside the container
RUN mkdir data
RUN mkdir image
RUN mkdir result

RUN chmod +x docker-entrypoint.sh

EXPOSE 5000

# CMD ["python3", "waiting.py"]
ENTRYPOINT ["sh", "docker-entrypoint.sh"]
