#!/bin/bash

[ _$SLURM_ARRAY_TASK_ID == _ ] && SLURM_ARRAY_TASK_ID=0
  
I=$(($SLURM_ARRAY_TASK_ID+1))
[ _$FILES == _ ] && FILES=files/positron_10MeV.txt
FIN=`sed -ne "${I}p" $FILES`
[ _$OUTDIR == _ ] && OUTDIR=/store/cpnr/users/jhgoh/JSNS2/Vertex/20240212_1
[ -d $OUTDIR ] || mkdir -p $OUTDIR
FOUT1=$OUTDIR/`basename ${FIN}`
FOUT2=$OUTDIR/`basename ${FIN/.root/.h5}`

echo "@@@ Setup anaconda: deactivate env"
if [ _$CONDA_DEFAULT_ENV != _ ]; then
  echo "@@@ Unset conda..."
  source `dirname $CONDA_PREFIX`/../etc/profile.d/conda.sh
  _CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV
  conda deactivate
else
  source /store/sw/anaconda3/etc/profile.d/conda.sh
  _CONDA_DEFAULT_ENV=ds4hep
fi

echo "@@@ Convert RAT root file to flat root file"
INPUTDIR=`dirname $FIN`
SIF=/store/sw/singularity/jsns2/jsns2-jade0-20221212.sif
singularity run -B$INPUTDIR:$INPUTDIR -B$OUTDIR:$OUTDIR $SIF <<EOF
[ -f convert ] || make || exit
./convert $FIN $FOUT1
EOF

echo "@@@ Setup anaconda: conda activate $_CONDA_DEFAULT_ENV"
conda activate $_CONDA_DEFAULT_ENV
echo "@@@ Converting .root to .h5"
./convert.py $FOUT1 $FOUT2
rm -f $FOUT1

echo "@@@ Done."
