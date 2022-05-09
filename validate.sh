#!/bin/bash -x
#
# Validate a model
#

/usr/bin/time -f "Time taken: %E" python3 ../code/train.py --validate model.pth
