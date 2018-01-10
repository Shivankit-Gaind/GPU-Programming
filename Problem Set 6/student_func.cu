//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */



#include "utils.h"
#include <thrust/host_vector.h>

//This kernel takes in an image represented as a uchar4 and splits
//it into three images consisting of only one color channel each
__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{
  
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;

  if(x<numRows && y<numCols){
    int index = x*numCols+y;
    redChannel[index] = inputImageRGBA[index].x;
    greenChannel[index] = inputImageRGBA[index].y;
    blueChannel[index] = inputImageRGBA[index].z;
  }  

}

__global__
void mask_kernel(unsigned char* mask,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{
  
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;

  if(x<numRows && y<numCols){
    int index = x*numCols+y;
    mask[index] = ((redChannel[index] + greenChannel[index] + blueChannel[index]) < 3 * 255) ? 1 : 0;
  }  

}

__global__
void interior_and_border_pixels(unsigned char* mask,
                      int numRowsSource,
                      int numColsSource,
                      unsigned char* borderPixels,
                      unsigned char* strictInteriorPixels)
                      
{
  
  int r = threadIdx.x + blockDim.x * blockIdx.x;
  int c = threadIdx.y + blockDim.y * blockIdx.y;

  if(r<=0 || r>= numRowsSource-1 || c<=0 || c>=numColsSource-1)
    return;


  //int index = r*numColsSource+c;
  if (mask[r * numColsSource + c]==1) {
        if (mask[(r -1) * numColsSource + c]==1 && mask[(r + 1) * numColsSource + c]==1 &&
            mask[r * numColsSource + c - 1]==1 && mask[r * numColsSource + c + 1]==1) {
         
          strictInteriorPixels[r * numColsSource + c] = 1;
          borderPixels[r * numColsSource + c] = 0;
          //interiorPixelList.push_back(make_uint2(r, c));
        }
        else {
          strictInteriorPixels[r * numColsSource + c] = 0;
          borderPixels[r * numColsSource + c] = 1;
        }
      }
      else {
          strictInteriorPixels[r * numColsSource + c] = 0;
          borderPixels[r * numColsSource + c] = 0;

      }
}

__global__
void computeG_kernel(unsigned char* channel,
                      float* g,
                      int numRowsSource,
                      int numColsSource,
                      unsigned char* strictInteriorPixels)
{
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;

  if(x>=numRowsSource || y>=numColsSource)
    return;

  int offset = x*numColsSource+y;

  if(!(strictInteriorPixels[offset]==1))
    return;

  float sum = 4.f * channel[offset];
  sum -= ((float)channel[offset - 1] + (float)channel[offset + 1]);
  sum -= ((float)channel[offset + numColsSource] + (float)channel[offset - numColsSource]);

  g[offset] = sum;
}


//Copy Kernel

__global__
void copy_kernel(unsigned char* red_src,
                      unsigned char* green_src,
                      unsigned char* blue_src,
                      int numRowsSource,
                      int numColsSource,
                      float* blendedValsRed_1,
                      float* blendedValsGreen_1,
                      float* blendedValsBlue_1,
                      float* blendedValsRed_2,
                      float* blendedValsGreen_2,
                      float* blendedValsBlue_2)

{
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;

  if(x>=numRowsSource || y>=numColsSource )
    return;

  int i = x*numColsSource+y;

    blendedValsRed_1[i] = (float)red_src[i];
    blendedValsRed_2[i] = (float)red_src[i];
    blendedValsBlue_1[i] = (float)blue_src[i];
    blendedValsBlue_2[i] = (float)blue_src[i];
    blendedValsGreen_1[i] = (float)green_src[i];
    blendedValsGreen_2[i] = (float)green_src[i];
}
                      

__global__
//Performs 800 iterations of the solver
void computeIteration(unsigned char* dstImg,
                      unsigned char* strictInteriorPixels,
                      unsigned char* borderPixels,
                      int numRowsSource,
                      int numColsSource,
                      float* f,
                      float* g,
                      float* f_next)
{


  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;

  if(x>=numRowsSource || y>=numColsSource )
    return;

  int offset = x*numColsSource+y;

  if(!(strictInteriorPixels[offset]==1))
    return;


  
    float blendedSum = 0.f;
    float borderSum  = 0.f;

    //process all 4 neighbor pixels
    //for each pixel if it is an interior pixel
    //then we add the previous f, otherwise if it is a
    //border pixel then we add the value of the destination
    //image at the border.  These border values are our boundary
    //conditions.
    if (strictInteriorPixels[offset - 1]) {
      blendedSum += f[offset - 1];
    }
    else {
      borderSum += dstImg[offset - 1];
    }

    if (strictInteriorPixels[offset + 1]) {
      blendedSum += f[offset + 1];
    }
    else {
      borderSum += dstImg[offset + 1];
    }

    if (strictInteriorPixels[offset - numColsSource]) {
      blendedSum += f[offset - numColsSource];
    }
    else {
      borderSum += dstImg[offset - numColsSource];
    }

    if (strictInteriorPixels[offset + numColsSource]) {
      blendedSum += f[offset + numColsSource];
    }
    else {
      borderSum += dstImg[offset + numColsSource];
    }

    float f_next_val = (blendedSum + borderSum + g[offset]) / 4.f;

    f_next[offset] = fmin(255.f, fmax(0.f, f_next_val)); //clip to [0, 255]

    __syncthreads();

    
    //Swapping phase


  

}

__global__
void blend_kernel(uchar4* outputImageRGBA,
                      unsigned char* strictInteriorPixels,
                      int numRowsSource,
                      int numColsSource,
                      float* const redChannel,
                      float* const greenChannel,
                      float* const blueChannel)
{
  
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;

  if(x>=numRowsSource || y>=numColsSource )
    return;

  int offset = x*numColsSource+y;

  if(!(strictInteriorPixels[offset]==1))
    return;

  outputImageRGBA[offset].x = (unsigned char)redChannel[offset];
  outputImageRGBA[offset].y = (unsigned char)greenChannel[offset];
  outputImageRGBA[offset].z = (unsigned char)blueChannel[offset];    

}



void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{

  /* To Recap here are the steps you need to implement
  
     1) Compute a mask of the pixels from the source image to be copied
        The pixels that shouldn't be copied are completely white, they
        have R=255, G=255, B=255.  Any other pixels SHOULD be copied.

     2) Compute the interior and border regions of the mask.  An interior
        pixel has all 4 neighbors also inside the mask.  A border pixel is
        in the mask itself, but has at least one neighbor that isn't.

     3) Separate out the incoming image into three separate channels

     4) Create two float(!) buffers for each color channel that will
        act as our guesses.  Initialize them to the respective color
        channel of the source image since that will act as our intial guess.

     5) For each color channel perform the Jacobi iteration described 
        above 800 times.

     6) Create the output image by replacing all the interior pixels
        in the destination image with the result of the Jacobi iterations.
        Just cast the floating point values to unsigned chars since we have
        already made sure to clamp them to the correct range.

      Since this is final assignment we provide little boilerplate code to
      help you.  Notice that all the input/output pointers are HOST pointers.

      You will have to allocate all of your own GPU memory and perform your own
      memcopies to get data in and out of the GPU memory.

      Remember to wrap all of your calls with checkCudaErrors() to catch any
      thing that might go wrong.  After each kernel call do:

      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      to catch any errors that happened while executing the kernel.
  */

  size_t srcSize = numRowsSource * numColsSource;
  dim3 block_size(32,32,1);
  dim3 grid_size((numRowsSource+32-1)/32, (numColsSource+32-1)/32, 1);

  //Allocate Memories for Source, Destination and Blended images on GPU
  uchar4* d_sourceImg;
  uchar4* d_destImg;
  uchar4* d_blendedImg;
  checkCudaErrors(cudaMalloc(&d_sourceImg, srcSize*sizeof(uchar4)));
  checkCudaErrors(cudaMalloc(&d_destImg, srcSize*sizeof(uchar4)));
  checkCudaErrors(cudaMalloc(&d_blendedImg, srcSize*sizeof(uchar4)));

  //Copying Source, Destination and Blended Images on the GPU
  checkCudaErrors(cudaMemcpy(d_sourceImg,h_sourceImg,srcSize*sizeof(uchar4),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_destImg,h_destImg,srcSize*sizeof(uchar4),cudaMemcpyHostToDevice));
  //Copying Destination Image to the Blended Image
  checkCudaErrors(cudaMemcpy(d_blendedImg,d_destImg,srcSize*sizeof(uchar4),cudaMemcpyDeviceToDevice));


  //split the source and destination images into their respective channels
  unsigned char* red_src;  
  unsigned char* green_src;
  unsigned char* blue_src;
  
  unsigned char* red_dst;    
  unsigned char* green_dst; 
  unsigned char* blue_dst;

  checkCudaErrors(cudaMalloc(&red_src, srcSize*sizeof(unsigned char)));  
  checkCudaErrors(cudaMalloc(&green_src, srcSize*sizeof(unsigned char)));
  checkCudaErrors(cudaMalloc(&blue_src, srcSize*sizeof(unsigned char)));
  checkCudaErrors(cudaMalloc(&red_dst, srcSize*sizeof(unsigned char)));  
  checkCudaErrors(cudaMalloc(&green_dst, srcSize*sizeof(unsigned char)));
  checkCudaErrors(cudaMalloc(&blue_dst, srcSize*sizeof(unsigned char)));


  //Launching the kernels For separating channels
  separateChannels<<<grid_size,block_size>>>(d_sourceImg,numRowsSource,numColsSource,red_src,green_src,blue_src);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  separateChannels<<<grid_size,block_size>>>(d_destImg,numRowsSource,numColsSource,red_dst,green_dst,blue_dst);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  


  //Then create mask  
  unsigned char* mask;
  checkCudaErrors(cudaMalloc(&mask, srcSize*sizeof(unsigned char)));

  //Launch the kernel
  mask_kernel<<<grid_size,block_size>>>(mask, numRowsSource, numColsSource, red_src, green_src, blue_src);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


  //next compute strictly interior pixels and border pixels
  unsigned char *borderPixels;
  unsigned char *strictInteriorPixels;
  checkCudaErrors(cudaMalloc(&borderPixels, srcSize*sizeof(unsigned char)));
  checkCudaErrors(cudaMalloc(&strictInteriorPixels, srcSize*sizeof(unsigned char)));

  //Launch the kernel
  interior_and_border_pixels<<<grid_size,block_size>>>(mask, numRowsSource, numColsSource, borderPixels,strictInteriorPixels);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  //next we'll precompute the g term - it never changes, no need to recompute every iteration
  float *g_red;   
  float *g_green; 
  float *g_blue;

  checkCudaErrors(cudaMalloc(&g_red, srcSize*sizeof(float)));  
  checkCudaErrors(cudaMalloc(&g_green, srcSize*sizeof(float)));
  checkCudaErrors(cudaMalloc(&g_blue, srcSize*sizeof(float)));

  //Initialize to 0
  checkCudaErrors(cudaMemset(g_red, 0, srcSize * sizeof(float)));
  checkCudaErrors(cudaMemset(g_green, 0, srcSize * sizeof(float)));
  checkCudaErrors(cudaMemset(g_blue, 0, srcSize * sizeof(float)));

  //Launch Kernels
  computeG_kernel<<<grid_size,block_size>>>(red_src,g_red,numRowsSource,numColsSource,strictInteriorPixels);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  computeG_kernel<<<grid_size,block_size>>>(green_src,g_green,numRowsSource,numColsSource,strictInteriorPixels);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  computeG_kernel<<<grid_size,block_size>>>(blue_src,g_blue,numRowsSource,numColsSource,strictInteriorPixels);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  //for each color channel we'll need two buffers and we'll ping-pong between them
  float *blendedValsRed_1;
  float *blendedValsRed_2;

  float *blendedValsBlue_1;
  float *blendedValsBlue_2;

  float *blendedValsGreen_1;
  float *blendedValsGreen_2;

  //Allocating Memory on the GPU
  checkCudaErrors(cudaMalloc(&blendedValsRed_1, srcSize*sizeof(float)));
  checkCudaErrors(cudaMalloc(&blendedValsRed_2, srcSize*sizeof(float)));

  checkCudaErrors(cudaMalloc(&blendedValsBlue_1, srcSize*sizeof(float)));
  checkCudaErrors(cudaMalloc(&blendedValsBlue_2, srcSize*sizeof(float)));

  checkCudaErrors(cudaMalloc(&blendedValsGreen_1, srcSize*sizeof(float)));
  checkCudaErrors(cudaMalloc(&blendedValsGreen_2, srcSize*sizeof(float)));


  //Launch Copy Kernel
  copy_kernel<<<grid_size,block_size>>>(red_src,green_src,blue_src,numRowsSource,numColsSource,blendedValsRed_1,blendedValsGreen_1,blendedValsBlue_1,blendedValsRed_2,blendedValsGreen_2,blendedValsBlue_2);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());  


  //Launching the iterations

  for(int i=0;i<800;i++){
    computeIteration<<<grid_size,block_size>>>(red_dst, strictInteriorPixels, borderPixels,
                     numRowsSource, numColsSource, blendedValsRed_1, g_red,
                     blendedValsRed_2);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    float* temp = blendedValsRed_1;
    blendedValsRed_1 = blendedValsRed_2;
    blendedValsRed_2 = temp;

  }
  for(int i=0;i<800;i++){
    computeIteration<<<grid_size,block_size>>>(green_dst, strictInteriorPixels, borderPixels,
                     numRowsSource, numColsSource, blendedValsGreen_1, g_green,
                     blendedValsGreen_2);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    float* temp = blendedValsGreen_1;
    blendedValsGreen_1 = blendedValsGreen_2;
    blendedValsGreen_2 = temp;
  }
  for(int i=0;i<800;i++){
    computeIteration<<<grid_size,block_size>>>(blue_dst, strictInteriorPixels, borderPixels,
                     numRowsSource, numColsSource, blendedValsBlue_1, g_blue,
                     blendedValsBlue_2);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    float* temp = blendedValsBlue_1;
    blendedValsBlue_1 = blendedValsBlue_2;
    blendedValsBlue_2 = temp;
  }
  
  
  //Blending Kernel Launched
  blend_kernel<<<grid_size,block_size>>>(d_blendedImg,strictInteriorPixels,numRowsSource,numColsSource,blendedValsRed_1,blendedValsGreen_1,blendedValsBlue_1);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  //Finally Copy back the destination image from the device to the host
  checkCudaErrors(cudaMemcpy(h_blendedImg,d_blendedImg,srcSize*sizeof(uchar4),cudaMemcpyDeviceToHost));

  //Freeing the allocated memory
  checkCudaErrors(cudaFree(mask));
  checkCudaErrors(cudaFree(blendedValsRed_1));
  checkCudaErrors(cudaFree(blendedValsRed_2));
  checkCudaErrors(cudaFree(blendedValsGreen_1));
  checkCudaErrors(cudaFree(blendedValsGreen_2));
  checkCudaErrors(cudaFree(blendedValsBlue_1));
  checkCudaErrors(cudaFree(blendedValsBlue_2));
  checkCudaErrors(cudaFree(g_red));
  checkCudaErrors(cudaFree(g_blue));
  checkCudaErrors(cudaFree(g_green));
  checkCudaErrors(cudaFree(red_src));
  checkCudaErrors(cudaFree(red_dst));
  checkCudaErrors(cudaFree(green_src));
  checkCudaErrors(cudaFree(green_dst));
  checkCudaErrors(cudaFree(blue_src));
  checkCudaErrors(cudaFree(blue_dst));
  checkCudaErrors(cudaFree(borderPixels));
  checkCudaErrors(cudaFree(strictInteriorPixels));
 
}


