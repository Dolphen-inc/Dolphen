#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status
set -o errexit

# 1. Upgrade pip and install build toolchain
pip install --upgrade pip
pip install cmake

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Download dlib facial landmarks file if missing
if [ ! -f "shape_predictor_68_face_landmarks.dat" ]; then
  echo "Downloading facial landmark predictor..."
  curl -L -O https://raw.githubusercontent.com/italojs/facial-landmarks-recognition/master/shape_predictor_68_face_landmarks.dat.bz2
  bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
fi
