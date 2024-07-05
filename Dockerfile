FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

MAINTAINER YeZhengMao <yezhengmaolove@gmail.com>

ARG PYTHON_VERSION=3.12
ARG http_proxy
ARG https_proxy

RUN apt-get update \
    && apt-get install -y \
    locales \
    build-essential \
    git \
    git-lfs \
    vim \
    openssh-server \
    cmake \
    pkg-config \
    zlib1g-dev libncurses5-dev \
    libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget \
    liblzma-dev libsqlite3-dev libbz2-dev

ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen

RUN mkdir /run/sshd \
    && echo 'root:mlora@123' | chpasswd \
    && echo "PermitRootLogin yes" >> /etc/ssh/sshd_config \
    && echo "AuthorizedKeysFile .ssh/authorized_keys" >> /etc/ssh/sshd_config \
    && echo "RSAAuthentication yes" >> /etc/ssh/sshd_config \
    && echo "PubkeyAuthentication yes" >> /etc/ssh/sshd_config \
    && echo "StrictModes no" >> /etc/ssh/sshd_config

ENV PYENV_ROOT=/root/.pyenv
ENV PATH "$PYENV_ROOT/bin/:$PATH"

RUN /usr/bin/echo -e '#!/bin/bash\n/usr/sbin/sshd -D' | tee /opt/start.sh \
    && /usr/bin/echo -e '#!/bin/bash\neval "$(pyenv init -)"\neval "$(pyenv virtualenv-init -)"\npython mlora_server.py --base_model $BASE_MODEL --root $STORAGE_DIR' | tee /opt/deploy.sh \
    && chmod +x /opt/start.sh \
    && chmod +x /opt/deploy.sh \
    && /usr/bin/echo -e 'export PYENV_ROOT=/root/.pyenv' >> ~/.bashrc \
    && /usr/bin/echo -e 'export PATH=/root/.pyenv/bin:$PATH' >> ~/.bashrc \
    && /usr/bin/echo -e 'eval "$(pyenv init -)"' >> ~/.bashrc \
    && /usr/bin/echo -e 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc \
    && git clone https://github.com/pyenv/pyenv.git /root/.pyenv \
    && git clone https://github.com/pyenv/pyenv-virtualenv.git /root/.pyenv/plugins/pyenv-virtualenv \
    && cd /root/.pyenv && src/configure && make -C src \
    && eval "$(pyenv init -)" \
    && eval "$(pyenv virtualenv-init -)"

RUN . ~/.bashrc \
    && pyenv install $PYTHON_VERSION \
    && pyenv global $PYTHON_VERSION \
    && git clone https://github.com/TUDB-Labs/mLoRA /mLoRA \
    && cd /mLoRA \
    && pyenv virtualenv $PYTHON_VERSION mlora \
    && pyenv local mlora \
    && pip install .[ci_test,test,debug,deploy]

WORKDIR /mLoRA

CMD /bin/bash /opt/start.sh
