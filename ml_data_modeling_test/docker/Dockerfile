FROM python:3.6.7-slim-stretch as install_container

COPY . /source
WORKDIR /source

RUN apt-get update && \
    apt-get install gcc git -y && \
    pip install --no-cache-dir -U pip setuptools && \
    pip install --no-cache-dir . && \
    pip install --no-cache-dir .

FROM python:3.6.7-slim-stretch as prod_container

COPY --from = install_container /usr/local /usr/local


CMD ["./run.sh"]

