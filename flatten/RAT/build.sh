#!/bin/bash
[ _$SIF == _ ] && SIF=/store/sw/singularity/jsns2/jsns2-jade0-20221212.sif
singularity run $SIF <<EOF
make
EOF

