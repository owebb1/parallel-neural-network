#!/usr/bin/bash
OUTFILE=/home/owebb1/cs87/project/source/parallel/accperepoc

cd /home/owebb1/cs87/project/source/parallel/ProjectCode/src


for ((n=2; n < 65; n=n*2))
do
  for ((k=0; k < 1; k=k+1))
  do
    echo "****START*****" &>> $OUTFILE
    echo "num_procs=$n" &>> $OUTFILE
    echo "" &>> $OUTFILE
    mpirun -np $n --hostfile hostfilec87 ./neuralnetparallel >> $OUTFILE
    echo "***STOP****" &>> $OUTFILE
  done
done
