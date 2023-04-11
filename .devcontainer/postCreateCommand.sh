#!/bin/bash
export DEBIAN_FRONTEND=noninteractive
sudo apt-get -y --force-yes update
sudo apt-get -y --force-yes install libgdal-dev
echo CPLUS_INCLUDE_PATH=/usr/include/gdal >> ~/.bashrc
echo C_INCLUDE_PATH=/usr/include/gdal >> ~/.bashrc
pip install --upgrade pip
yes | pip install .[datasets,docs,style,tests]
pip install --user pre-commit