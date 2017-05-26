#!/bin/bash

cd src
rm main.cpp
ln -s main.cpp.$1 main.cpp
cd ..

