FROM dht-image-base

WORKDIR /app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Make some required folders inside the container
RUN mkdir data
RUN mkdir image
RUN mkdir result

# Install packages
RUN pip install torch-summary

RUN chmod +x docker-entrypoint.sh

EXPOSE 5000

# CMD ["python3", "testing3.py"]
ENTRYPOINT ["sh", "docker-entrypoint.sh"]
