FROM python:3.10
ADD requirements.txt /tmp/requirements.txt
RUN pip install -r --no-cache-dir /tmp/requirements.txt &&\
    rm /tmp/requirements.txt
ADD . /app
WORKDIR /app
ENTRYPOINT ["python", "main.py"]