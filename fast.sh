#!/bin/bash 
#
# train for a few epochs while performing "fast sweeps"
#

pip3 install -q -r ../code/requirements.txt
python3 ../code/train.py --epochs 10
