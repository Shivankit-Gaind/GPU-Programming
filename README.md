# GPU-Programming

This repository contains my solutions to the programming assignments of the Udacity cs344 "Intro to Parallel Programming" course. The assignments involve coding a series of image processing algorithms, such as you might find in Photoshop or Instagram.

I have uploaded this code which contains my completed solutions along with the rest of the code in the library that was needed to compile and run it. To find the code that I wrote, please look in: `Problem Set <INT>/student_func.cu`.

Udacity course that this work comes from: https://eg.udacity.com/course/intro-to-parallel-programming--cs344

## Problem Descriptions

### Problem 1 : Color to Greyscale Conversion
Converted RGBA image to greyscale using the formula recommended by the NTSC.

### Problem 2 : Image Blurring
1) Convert RGBA image that is defined as an array of structures (AoS) with a byte representing the R, G, B or A weight at each pixel to a structure of arrays (SoA) so that each channel (which is to be blurred separately) is layed out in contiguous memory. 
2) Used gaussian blurring to blur/smooth each channel separately in the image. Among other things, this problem set employed the **map**/**stencil** parallel primitive algorithm.

### Problem 3 : HDR Tone-Mapping
The main aspect of Tone Mapping that is parallelizeable and was the focus of this problem set is the conversion of an image of luminance values at each pixel to a histogram of luminance values and then finding the cumulative distribution function (CDF) so that histogram equalization could be performed. To find the CDF, it was necessary to 
1) compute the min/max of the luminance value in parallel, 
2) generate a histogram of the luminance values, and 
3) perform an exclusive scan on the histogram. I chose the Hillis and Steele Scan to do the final step. 
Among other things, this assigment employed the **reduce**, **histogram**, and **scan** parallel primitive algorithms.

### Problem 4 : Red Eye Removal
The algorithm for red eye removal was to find the likelihood for every pixel that tells us how likely it is to be a red eye pixel, which was given to the student. The step that was parallelizable and the subject of this problem set was to implement a parallel sorting algorithm to perform on the GPU. In order to accomplish this, I performed a **Parallel Radix Sort** of the unsigned integer histogram. This algorithm relied on an implementation of a **Blelloch Scan (prefix sum scan) for an array of arbitrary number of elements** (except it must be less than 1024*1024).

### Problem 5 : Histogramming for Speed
The goal of this assignment is compute a histogram as fast as possible. The problem was simplified as much as possible to allow me to focus solely on the histogramming algorithm. The input values that we need to histogram are already the exact bins that need to be updated.
