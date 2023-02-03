#!/bin/bash
export DEBIAN_FRONTEND=noninteractive
sudo apt-get -y --force-yes update
sudo apt-get -y --force-yes install libgdal-dev
echo CPLUS_INCLUDE_PATH=/usr/include/gdal >> ~/.bashrc
echo C_INCLUDE_PATH=/usr/include/gdal >> ~/.bashrc
pip install --upgrade pip
pip install --user -r ./requirements/datasets.txt
pip install --user -r ./requirements/docs.txt
pip install --user -r ./requirements/required.txt
pip install --user -r ./requirements/style.txt
pip install --user -r ./requirements/tests.txt
pip install --user pre-commit