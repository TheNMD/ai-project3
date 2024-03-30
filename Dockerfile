FROM python:3.10-slim
# set the working directory in the container
WORKDIR /usr/src/app
# cp the current directory contents into the container at /usr/src/app
COPY . .

# RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
# CMD ["python3", "./1.gen_meta.py"]
CMD ["python3", "testing.py"]
