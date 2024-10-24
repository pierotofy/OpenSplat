#!/bin/bash

# This script runs the SPLAT program with the given input file and output file
# The input file is the first argument and the output file is the second argument

# Check if the number of arguments is correct
if [ -z "$1" ]; then
    echo "Usage: $0 <input .MOV file, or images/ directory"
    exit 1
fi

# Check if the input is exists
if [ -f "$1" && ]; then
    # currently only support .MOV video files.
    if [[ "$1" != *.MOV ]]; then
        echo "Input file is not a .MOV file, so not supported"
        exit 1
    fi
    # create 8 bit images using ffmpeg
    IMAGES_DIR=$(dirname "$1")/images
    mkdir -p $IMAGES_DIR
    ffmpeg -i "$1" -r 5 -vf "format=rgb24" ${IMAGES_DIR}/image%08d.png
elif [ -d "$1" ]; then
    IMAGES_DIR="$1"
else
    echo "Input file or directory does not exist"
    exit 1
fi

# run colmap
colmap automatic_reconstructor --image_path $IMAGES_DIR --workspace_path result

# Run the SPLAT program with the input file and output file
./opensplat $IMAGES_DIR -n 2000