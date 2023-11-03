= Layout of the repo:

### Sequential Program Code:
  `sequential/ProjectCode/src/*`

  - Very few edits here in this repository
  - Most everything besides a few edits is copies from: https://github.com/tharrington923/neuralnet_handwritten_digit_recognition

### Parallel Adjusted Program
  `parallel/ProjectCode/src/*`

  - All parallelization is done in two files: ./network.cpp ./neuralnet.cpp
  - other files are the same implementation as the sequential
  - Added makefile, run script for experiments, and experiment output files

# Running the Sequential Program
  1. `$ cd sequential/ProjectCode/build`
  2. `$ make`
  3. `$ cd ../bin`
  4. `$ ./neuralnet`


# Running the Parallel Program
  1. `$ cd parallel/ProjectCode/src`
  2. `$ make`
  3. `$ mpirun -np 4 --hostfile hostfilec87 ./neuralnetparallel`
