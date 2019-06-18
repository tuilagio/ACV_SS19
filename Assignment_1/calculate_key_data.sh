#!/bin/bash
# Calculates the key points data (extracts SIFT features) and outputs a .key file
# Parameter 1: the directory that contains the photos from which we want to extract key points data

photos_dir="$1"
return_dir=${pwd}

cd "$1"
for file in *.jpg; do
	file_name=${file::-4}
	ocv_sift_detector "$file" "${file_name}.key"
done
cd "$return_dir"

