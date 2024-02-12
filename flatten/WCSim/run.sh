#!/bin/bash

[ _$SLURM_ARRAY_TASK_ID == _ ] && SLURM_ARRAY_TASK_ID=0
  
I=$(($SLURM_ARRAY_TASK_ID+1))
FIN=`sed -ne "${I}p" files.txt`
[ _$OUTDIR == _ ] && OUTDIR=/store/cpnr/users/jhgoh/KNO/20240212_1
[ -d $OUTDIR ] || mkdir -p $OUTDIR
FOUT=$OUTDIR/`basename ${FIN/.root/.h5}`

echo -n "@@@ Setup anaconda"
if [ _$CONDA_DEFAULT_ENV != _ ]; then
  echo "... already set to $CONDA_DEFAULT_ENV. skip."
else
  source /store/sw/anaconda3/etc/profile.d/conda.sh
  CONDA_DEFAULT_ENV=ds4hep
  echo "... to $CONDA_DEFAULT_ENV"
  conda activate $CONDA_DEFAULT_ENV
fi

echo "@@@ Converting WCSim .root to .h5"
./convert.py $FIN $FOUT

echo "@@@ Done."
