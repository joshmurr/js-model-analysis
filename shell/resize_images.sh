#!/bin/bash

DIR=$1
OUTPUT_DIR=$2

a=0

for IMG in ${DIR}/*.jpg; do
  new=$(printf "%04d.jpg" "$a")
  ffmpeg -i "$IMG" -vf scale=256:256 "${OUTPUT_DIR}/${new}"
  let a=a+1
done
