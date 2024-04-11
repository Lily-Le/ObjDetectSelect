#!/bin/bash
python detect_obj_select2.py --source data/video/batc_lab1.mp4 --view-img

python detect_obj_select2.py --source data/images/motorbike1.jpeg --view-img

python detect.py --source data/images/motorbike.jpeg --classes 3

python detect.py --source data/images/motorbike1.jpeg --classes 3 --view-img
