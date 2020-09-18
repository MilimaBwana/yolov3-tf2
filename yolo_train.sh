#!bin/bash

VENV=/home/jakob/Dokumente/INFORMATIK/_SOMMERSEMESTER_20/Masterarbeit/pollenanalysis2.0/detectionvenv/bin/python
TFRECORD_TRAIN=/home/jakob/Dokumente/INFORMATIK/_SOMMERSEMESTER_20/Masterarbeit/pollenanalysis2.0/src/pollen_detection/data/chess/chess12_train.tfrecords
TFRECORD_VAL=/home/jakob/Dokumente/INFORMATIK/_SOMMERSEMESTER_20/Masterarbeit/pollenanalysis2.0/src/pollen_detection/data/chess/chess12_val.tfrecords
TFRECORD_TEST=/home/jakob/Dokumente/INFORMATIK/_SOMMERSEMESTER_20/Masterarbeit/pollenanalysis2.0/src/pollen_detection/data/chess/chess12_test.tfrecords
CLASSES=/home/jakob/Dokumente/INFORMATIK/_SOMMERSEMESTER_20/Masterarbeit/pollenanalysis2.0/src/pollen_detection/data/chess/chess12.names
YOLO=/home/jakob/Dokumente/INFORMATIK/_SOMMERSEMESTER_20/Masterarbeit/yolov3-tf2
TRAIN_WEIGHTS=/home/jakob/Dokumente/INFORMATIK/_SOMMERSEMESTER_20/Masterarbeit/pollenanalysis2.0/src/pollen_detection/checkpoints/yolov3_train_10.tf
WEIGHTS=/home/jakob/Dokumente/INFORMATIK/_SOMMERSEMESTER_20/Masterarbeit/yolov3-tf2/snapshots/yolov3.tf

$VENV $YOLO/train.py --batch_size 4 --dataset $TFRECORD_TRAIN --val_dataset $TFRECORD_VAL --epochs 10 --transfer no_output --classes $CLASSES --weights $WEIGHTS
#$VENV $YOLO/detect.py --classes $CLASSES --tfrecord $TFRECORD_TEST --weights $TRAIN_WEIGHTS 
