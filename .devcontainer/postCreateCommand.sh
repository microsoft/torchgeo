#!/bin/bash
# Update and install GDAL
sudo apt-get -y --force-yes update
sudo apt-get -y --force-yes install libgdal-dev
# Add GDAL to the environment variables
echo CPLUS_INCLUDE_PATH=/usr/include/gdal >> ~/.bashrc
echo C_INCLUDE_PATH=/usr/include/gdal >> ~/.bashrc
source ~/.bashrc
# Update pip and install the package
pip install --upgrade pip
yes | pip install .[datasets,docs,style,tests]
pip install --user pre-commit