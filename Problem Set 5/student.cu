/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


#include "utils.h"
#include "stdio.h"
int block_size = 512; //Keep it less than the number of bins

__global__
void atomic_kernel(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals)
{
  //TODO fill in this kernel to calculate the histogram
  //as quickly as possible

  //Although we provide only one kernel skeleton,
  //feel free to use more if it will help you
  //write faster code
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if(id>=numVals){
    return;
  }

  atomicAdd(&histo[vals[id]],1);

}

__global__
void atomic_kernel2(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals)
{
  //TODO fill in this kernel to calculate the histogram
  //as quickly as possible

  //Although we provide only one kernel skeleton,
  //feel free to use more if it will help you
  //write faster code
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id>=numVals){
    return;
  }

  __shared__
  int memory[512]; //512 is the block size
  memory[threadIdx.x] = vals[id];

  __syncthreads();
  

  atomicAdd(&histo[memory[threadIdx.x]],1);

}


//Assumes that Block Size is less than or equal to the number of bins
__global__
void yourHisto(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               const int numVals,
               const int numBins)
{
  //TODO fill in this kernel to calculate the histogram
  //as quickly as possible

  //Although we provide only one kernel skeleton,
  //feel free to use more if it will help you
  //write faster code
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id>=numVals){
    return;
  }

  __shared__ int memory[1024]; //Here 1024 is the total number of bins

  int i = 0;
  while(i<numBins){
    memory[threadIdx.x + i] = 0;
    i+= 512; //Block Size
  }
  
  __syncthreads();

  atomicAdd(&memory[vals[id]],1); //Actual Histogram calculation is done in the shared memory

  __syncthreads();


  //Merging of Histograms
  i = 0;
  while(i<numBins){
    atomicAdd(&histo[threadIdx.x+i],memory[threadIdx.x + i]);
    i+= 512; //Block Size
  }

}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
  //TODO Launch the yourHisto kernel

  //if you want to use/launch more than one kernel,
  //feel free

  //printf("Blocks: %d\n",(numElems+block_size-1)/block_size);


  //Runs in 8.298688 ms -- block_size = 1024
  //Runs in 8.755904 ms -- block_size = 512
  //Runs in 9.031616 ms -- block_size = 256
  //If I make it 128 or less -- error -- i.e 80,000 blocks
  //Reason: maximum available registers per block -- 65536
  //atomic_kernel<<<dim3((numElems+block_size-1)/block_size,1,1),dim3(block_size,1,1)>>>(d_vals,d_histo,numElems);

  //Time Increases -- Because No computation is done after copying into shared memory -- 10 ms
  //atomic_kernel2<<<dim3((numElems+block_size-1)/block_size,1,1),dim3(block_size,1,1)>>>(d_vals,d_histo,numElems);

  //Actual Computation is done using Shared Memory
  //Time Taken -- 3.3 ms
  yourHisto<<<dim3((numElems+block_size-1)/block_size,1,1),dim3(block_size,1,1),numBins*sizeof(int)>>>(d_vals,d_histo,numElems,numBins);
  


  //printf("message: %s\n",cudaGetErrorString(cudaGetLastError()));
  cudaDeviceSynchronize(); 
  
  checkCudaErrors(cudaGetLastError());
}
