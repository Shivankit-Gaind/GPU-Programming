/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include "stdio.h"


//Kernel To find the minimum and maximum

__global__ 
void parallel_minimum_maximum(float* min_array, float* max_array, float* min_array2, float* max_array2, float &min_logLum, float &max_logLum, int totalSize){

    int myId = threadIdx.x + blockDim.x * blockIdx.x;

    if(myId>=totalSize)
      return;

    int tid  = threadIdx.x;

    // do reduction in global mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s && myId+s <totalSize)
        {

            min_array[myId] = fmin(min_array[myId], min_array[myId + s]);
            max_array[myId] = fmax(max_array[myId], max_array[myId + s]);
        }

        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        min_array2[blockIdx.x] = min_array[myId];
        max_array2[blockIdx.x] = max_array[myId];        
    }
}

//Function to find minimum and maximum
__host__
void minimum_maximum(const float* const d_logLuminance, float &min_logLum, float &max_logLum, const size_t numRows, const size_t numCols){
  
  int totalSize = numRows * numCols;
  
  //Maximum number of threads in a block
  int maxThreads = 1024;
  
  
  //Allocating 2 arrays for doing all computations
  float* min_array;
  float* max_array;
  float* min_array2;
  float* max_array2;
  checkCudaErrors(cudaMalloc(&min_array, totalSize*sizeof(float)));
  checkCudaErrors(cudaMalloc(&max_array, totalSize*sizeof(float)));
  checkCudaErrors(cudaMalloc(&min_array2, totalSize*sizeof(float)));
  checkCudaErrors(cudaMalloc(&max_array2, totalSize*sizeof(float)));
  checkCudaErrors(cudaMemcpy(min_array, d_logLuminance, totalSize*sizeof(float), cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(max_array, d_logLuminance, totalSize*sizeof(float), cudaMemcpyDeviceToDevice));  
  checkCudaErrors(cudaMemcpy(min_array2, d_logLuminance, totalSize*sizeof(float), cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(max_array2, d_logLuminance, totalSize*sizeof(float), cudaMemcpyDeviceToDevice));

  float* temp;

  int blocks = (totalSize+maxThreads-1)/maxThreads;
  
  while(totalSize!=1){
    parallel_minimum_maximum<<<blocks,maxThreads>>>(min_array, max_array, min_array2, max_array2, min_logLum, max_logLum, totalSize);    

    //Changing the total size for next step
    totalSize = blocks;
    blocks = (totalSize+maxThreads-1)/maxThreads;

    //Exchanging arrays
    temp = min_array;
    min_array = min_array2;
    min_array2 = temp;

    temp = max_array;
    max_array = max_array2;
    max_array2 = temp;
  }


  //We want only the first element
  float* h_min = (float*)malloc(sizeof(float));
  float* h_max = (float*)malloc(sizeof(float));

  checkCudaErrors(cudaMemcpy(h_min, min_array, sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_max, max_array, sizeof(float), cudaMemcpyDeviceToHost));

  //Extracting the minimum and maximum
  min_logLum = h_min[0];
  max_logLum = h_max[0];
   
}


//Generate Histogram 

__global__
void generate_histogram(const float* const d_logLuminance,
                        unsigned int* const d_cdf,
                        const float min_logLum,
                        const float max_logLum,
                        const size_t numRows,
                        const size_t numCols,
                        const size_t numBins){

  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;

  int index = x*numCols+y;

  if(x>=numRows || y>=numCols)
    return;

  float lumRange = max_logLum - min_logLum;
  
  int bin = int(min((int)(numBins-1), int((d_logLuminance[index] - min_logLum) / lumRange * numBins)));  
  atomicAdd(&(d_cdf[bin]),1);
}


//Generate Cumulative Distribution

__global__
void generate_cdf(unsigned int* d_cdf,
                        const size_t numRows,
                        const size_t numCols,
                        const size_t numBins){

  int index = threadIdx.x + blockDim.x * blockIdx.x;
  
  if(index>=numBins)
    return;

  double l = log2((double)numBins);
  int steps = ceil(l);

  for(int step=0;step<steps;step++){

    int jump = (int)powf(2,step);
    int temp1 = d_cdf[index];
    int temp2 = 0;
    if(index-jump>=0){
      temp2 = d_cdf[index-jump];      
    }
    
    //Values are read
    __syncthreads();

    d_cdf[index] = temp1 + temp2;

    //Don't start next step until all threads are done executing the current step
    __syncthreads();
  }
  
  //Make this scan exclusive
  int temp;
  if(index>0)
    temp = d_cdf[index-1];
  else
    temp = 0;

  __syncthreads();

  d_cdf[index] = temp;

}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

  //Maximum threads in a block = 1024 (along one dimension)

  
  //1 and 2
  minimum_maximum(d_logLuminance, min_logLum, max_logLum, numRows,numCols);  
  //3
  const dim3 blockSize(32,32,1);
  const dim3 gridSize(numRows/32 +1, numCols/32 +1, 1);
  generate_histogram<<<gridSize,blockSize>>>(d_logLuminance, d_cdf, min_logLum, max_logLum, numRows,numCols,numBins);  
  //4
  generate_cdf<<<(numBins+1024-1)/1024,1024>>>(d_cdf,numRows,numCols,numBins);
}


  