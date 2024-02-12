#!/bin/bash

[ _$SLURM_ARRAY_TASK_ID == _ ] && SLURM_ARRAY_TASK_ID=0
  
I=$(($SLURM_ARRAY_TASK_ID+1))
FIN=`sed -ne "${I}p" files.txt`
FOUT=`basename ${FIN/.root/.h5}`
OUTDIR=/store/cpnr/users/jhgoh/KNO/20240212_1
./convert.py $FIN $OUTDIR/$FOUT

