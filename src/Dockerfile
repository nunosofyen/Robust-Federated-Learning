FROM docker.io/tensorflow/tensorflow:latest-gpu 
RUN apt update
RUN dpkg --add-architecture i386 && apt update
RUN apt-get -y install \
    git \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    sudo

COPY ./requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir
RUN git clone https://github.com/Trusted-AI/adversarial-robustness-toolbox.git
RUN pip install ./adversarial-robustness-toolbox
