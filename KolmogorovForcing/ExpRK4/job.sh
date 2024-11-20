#!/bin/sh

#JX --bizcode ACA39
#JX -L rscunit=RURI
#JX -L rscgrp=default
#JX -L vnode=1
#JX -L vnode-core=36
#JX -L elapse=6:00:00
#JX -S

export OMP_NUM_THREADS=36

module load intel
./Kolmogorov_ExpRK4.o