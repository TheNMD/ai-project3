FROM dht-image-base

WORKDIR /app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Make some required folders inside the container
RUN mkdir data
RUN mkdir image

# Update pip and install packages
RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

RUN chmod +x docker-entrypoint.sh

EXPOSE 5000

# CMD ["ls"]

ENTRYPOINT ["sh", "docker-entrypoint.sh"]
