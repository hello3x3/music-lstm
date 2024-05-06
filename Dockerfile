FROM ubuntu:22.04

LABEL maintainer="shujiuhe <shujiuhe@outlook.com>"

SHELL [ "/bin/bash", "--login", "-c" ]

ENV DEBIAN_FRONTEND noninteractive

USER root

# ==========================================================================
# dev: custom
# ==========================================================================

COPY --chown=root:root ./datasets/musicfonts/Arachno.sf2 /root/Arachno.sf2

COPY --chown=root:root ./src/mid2wav.py /root/mid2wav.py

RUN sed -i 's@http://archive.ubuntu.com/ubuntu/@http://mirrors.aliyun.com/ubuntu/@g' /etc/apt/sources.list && \
    sed -i 's@http://security.ubuntu.com/ubuntu/@http://mirrors.aliyun.com/ubuntu/@g' /etc/apt/sources.list && \
    apt-get clean && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install fluidsynth -y

RUN apt-get install python3-pip -y && \
    python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip && \
    pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 install tqdm
