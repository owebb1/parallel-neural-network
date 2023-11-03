#include "aliases.hpp"
#include "network.hpp"
#include "network.cpp"
#include <vector>
#include <cassert>
#include <fstream>
#include <iostream>
#include <random>
//#include "testfunctions.cpp"

int main(int argc, const char * argv[]) {
  std::clock_t start;
  double duration;

  std::vector<int> a = {784,30,10};

  Network b(a);
  start = std::clock();
  b.train_network(30, 10, 3.0, false);
  duration = ( std::clock()-start ) / (double) CLOCKS_PER_SEC;
  b.save_network_parameters("trained_network_hiddenlayer1_neurons30_batchsize10_eta3");
  std::cout << "Duration is " << duration << " seconds." << std::endl;
  return 0;
}
