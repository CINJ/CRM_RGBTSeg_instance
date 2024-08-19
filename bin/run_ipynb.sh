#!/bin/bash
# (C)2022 J.Cincotta
# Version 3.0
# Written by Joe Cincotta
#
# usage: run_ipynb.sh <environemtn name> <notebook ipynb filename>
#
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$DIR")"
CONDENV="$1"
shift
NBROOT="$1"
shift
NOTEBOOK="$1"
shift
ARCH=$(arch)
OS=$(uname)
COMPAT=0
COMARCH="osx-arm64"

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
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

if [ "$ARCH" == "arm64" ] && [ "$OS" == "Darwin" ];
then
  echo "Compatability mode enabled for M1 Mac"
  COMPAT=1
fi
eval "$(conda shell.bash hook)"
conda activate $CONDENV
if [ "$COMPAT" == "1" ];
then
  conda config --env --set subdir $COMARCH
fi

cd /projects/CRM_RGBTSeg/models/mask2former/pixel_decoder/ops/
source ./make.sh

echo "$NBROOT"
cd $NBROOT
for file in *.ipynb
  do
    jupyter trust $file
  done

rm -f pm-log.txt
touch pm-log.txt
tail -f pm-log.txt &
tensorboard --logdir=$NBROOT/checkpoints --host localhost --port=8008 --DEBUG &
papermill $NOTEBOOK $NOTEBOOK -k python3 --log-output --log-level DEBUG --request-save-on-cell-execute --report-mode  --stdout-file pm-log.txt


