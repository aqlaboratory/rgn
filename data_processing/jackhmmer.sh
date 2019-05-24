#!/bin/bash

NUM_ITERATIONS=5
Z_VALUE=1e8
E_VALUE=1e-10
NUM_THREADS=8
STEM=$1
DB=$2

jackhmmer -N ${NUM_ITERATIONS} -Z ${Z_VALUE} --incE ${E_VALUE} --incdomE ${E_VALUE} -E ${E_VALUE} --domE ${E_VALUE} --cpu ${NUM_THREADS} -o /dev/null -A $STEM.sto --tblout $STEM.tblout $STEM $DB
esl-reformat -o $STEM.a2m a2m $STEM.sto
esl-weight -p --amino --informat a2m -o $STEM.weighted.sto $STEM.a2m
esl-alistat --weight --amino --icinfo $STEM.icinfo --cinfo $STEM.cinfo $STEM.weighted.sto > /dev/null