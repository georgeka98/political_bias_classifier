#!/bin/bash

# edit this if you cloned the repo elsewhere
repo_dir=/home/$USER/political_vias_classifier
cd $repo_dir

conda activate
# Print the man page
python3 main.py -h

# # Set the input path and output path, with arguments -i and -o, and run!
# python3 train.py -i data/input -o data/output

# # You have just run an experiment! Check out what it did...
# ls data/output
# cat data/output/*

# # Try running another experiment!
# python3 train.py -i data/input -o data/output --lr 0.001
# ls data/output
# cat data/output/*

# Which learning rate was more accurate?!