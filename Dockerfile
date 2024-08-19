# FROM condaforge/mambaforge:latest
# taken from https://github.com/conda-forge/miniforge-images/blob/master/ubuntu/Dockerfile
FROM 11.6.1-cudnn8-devel-ubuntu20.04
ARG MINIFORGE_NAME=Miniforge3
ARG MINIFORGE_VERSION=23.3.1-1
ARG TARGETPLATFORM
ARG CONDA_YAML=bees0006.yml
ENV CONDA_DIR=/opt/conda
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=${CONDA_DIR}/bin:${PATH}
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV CPATH=/usr/local/cuda/include:${CPATH}
RUN apt-get update > /dev/null && \
    apt-get install --no-install-recommends --yes \
        wget bzip2 ca-certificates \
        git \
        tini \
        > /dev/null && \
    apt-get clean && \
    wget --no-hsts --quiet https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/${MINIFORGE_NAME}-${MINIFORGE_VERSION}-Linux-$(uname -m).sh -O /tmp/miniforge.sh && \
    /bin/bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh && \
    conda clean --tarballs --index-cache --packages --yes && \
    find ${CONDA_DIR} -follow -type f -name '*.a' -delete && \
    find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete && \
    conda clean --force-pkgs-dirs --all --yes  && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> /etc/skel/.bashrc && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> ~/.bashrc

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Sydney/Australia
# Etc/UTC
RUN apt-get update
RUN apt-get -yqq upgrade
RUN apt-get install -yqq --fix-missing build-essential libpq-dev xorg libxi6 libxfixes3 libxcursor1 libxdamage1 libxext6 libxfont2 nginx
RUN apt-get autoremove -y
# RUN timedatectl set-timezone Sydney/Australia

RUN mkdir /api
RUN mkdir /notebooks
RUN mkdir /projects

WORKDIR /api
COPY ./bin/$CONDA_YAML /api
RUN mamba update -y -n base conda -c conda-forge -c anaconda
RUN mamba env create -f $CONDA_YAML

COPY ./test-cuda.py /api
COPY ./bin/py.sh  /api
COPY ./bin/jupyter.sh /api
COPY ./bin/run_ipynb.sh /api
COPY ./bin/tensorboard.sh /api
COPY ./bin/build-ops.sh /api
