#!/bin/bash

echo "Insert 'img' for Image or 'vid' for video:"
read mode
echo "Insert the path of the file:"
read target

python phase1.py $mode $target