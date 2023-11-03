#include "aliases.hpp"
#include "network.hpp"
#include "network.cpp"
#include <vector>
#include <cassert>
#include <fstream>
#include <iostream>
#include <random>
#include <mpi.h>
//#include "testfunctions.cpp"

/* Edited code from the sequential repository */
int main(int argc, char * argv[]) {
  std::clock_t start;
  double duration;
  int rank=0;
  unsigned long size=0;
  int nprocs=0;
  MPI_Status status;

  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  std::vector<int> a = {784,30,10};
  Network b(a);
  start = std::clock();
  b.train_network(30, 10, 3.0, rank, nprocs, true);
  duration = ( std::clock()-start ) / (double) CLOCKS_PER_SEC;

  MPI_Finalize();
  if(rank == 1){
    b.save_network_parameters(
      "trained_network_hiddenlayer1_neurons30_batchsize10_eta3_parallel");
    std::cout << "Duration is " << duration << " seconds." << std::endl;
  }
  return 0;
}
