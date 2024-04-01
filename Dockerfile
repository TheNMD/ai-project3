FROM python:3.10-slim

WORKDIR /app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Make some required folders inside the container
RUN mkdir data

RUN chmod +x entrypoint.sh

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

# CMD ["ls"]

ENTRYPOINT ["sh", "./entrypoint.sh"]