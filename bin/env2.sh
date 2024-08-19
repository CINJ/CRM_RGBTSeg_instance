#!/bin/bash
# (C)2022 J.Cincotta
# Version 5.2
# Written by Joe Cincotta
#
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$DIR")"
ARCH=$(arch)
OS=$(uname)
# ARCH Values:
#   aarch64 : Linux ARM
#   arm64   : M1
#   x86_64  : X86
# OS Values:
#   Linux
#   Darwin


# Available Anaconda Architectures:
# linux-64
# linux-aarch64
# linux-ppc64le
# osx-64
# osx-arm64
# win-64
COMPAT=0
COMARCH="osx-arm64"

if [ "$ARCH" == "arm64" ] && [ "$OS" == "Darwin" ];
then
  echo "Compatability mode enabled for M1 Mac"
  COMPAT=1
fi

if [ "$1" == "-delete" ] || [ "$1" == "-d" ];
then
  conda remove --name $2 --all -y
  exit 0
fi

if [ "$1" == "-update" ] || [ "$1" == "-u" ];
then
  CONDENV="$2"
  eval "$(conda shell.bash hook)"
  conda activate ${CONDENV}
  if [ "$COMPAT" == "1" ];
  then
    # conda update -y -n base -c conda-forge conda
    #conda update -y -n base -c conda-forge mamba
    conda config --env --set subdir $COMARCH
    # conda install -y mamba -n ${CONDENV} -c conda-forge
    CONDA_SUBDIR=$COMARCH mamba env update -f "${DIR}/${CONDENV}.yml"
  else
    mamba env update -f "${DIR}/${CONDENV}.yml"
  fi
else
  CONDENV="$1"
  echo "Install Mamba"
  conda install -y mamba -n base -c conda-forge
  echo "Create ${CONDENV}"
  if [ "$COMPAT" == "1" ];
  then
    CONDA_SUBDIR=$COMARCH mamba env create -f "${DIR}/${CONDENV}.yml"
    eval "$(conda shell.bash hook)"
    conda activate ${CONDENV}
    conda config --env --set subdir $COMARCH
    conda install -y mamba -n ${CONDENV} -c conda-forge
  else
    mamba env create -f "${DIR}/${CONDENV}.yml"
    eval "$(conda shell.bash hook)"
    conda activate ${CONDENV}
    conda install -y mamba -n ${CONDENV} -c conda-forge
  fi
fi
