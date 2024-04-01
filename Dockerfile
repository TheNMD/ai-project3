FROM python:3.10-slim

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Make some required folders inside the container
RUN mkdir data
RUN mkdir metadata
RUN mkdir image
RUN mkdir result

RUN chmod +x entrypoint.sh

# RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

ENTRYPOINT ["sh", "./entrypoint.sh"]