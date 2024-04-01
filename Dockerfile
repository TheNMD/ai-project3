FROM python:3.10-slim

WORKDIR /data/DanHoangThu
# de cai path dan toi image la ../data_WF/NhaBe

# cp the current directory contents into the container at /usr/src/app
COPY . .

RUN chmod +x entrypoint.sh

# RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

ENTRYPOINT ["./entrypoint.sh" ]