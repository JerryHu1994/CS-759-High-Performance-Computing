/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * This software and the information contained herein is PROPRIETARY and 
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and 
 * conditions of a Non-Disclosure Agreement.  Any reproduction or 
 * disclosure to any third party without the express written consent of 
 * NVIDIA is prohibited.     
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

#ifndef _VECTOR_REDUCTION_KERNEL_H_
#define _VECTOR_REDUCTION_KERNEL_H_

#define NUM_ELEMENTS 1024


// **===----------------- Modify this function ---------------------===**
//! @param g_idata  input data in global memory
//                  result is expected in index 0 of g_idata
//! @param n        input number of elements to scan from input data
// **===------------------------------------------------------------------===**
__global__ void reduction(float *g_data, int n)
{
	int threadId = threadIdx.x;
	
	int index = threadId;
    int startSize = n/2;
    __shared__ double s_data[512];
    
    //copy the data into the shared memory
    if(index < n/2){
        s_data[threadId] = g_data[index*2] + g_data[index*2+1];
    }else{
        s_data[threadId] = 0;
    }
    //synchronize the threads to make sure all the data is loaded
    __syncthreads();
    
    for(unsigned int i=startSize/2;i>0;i>>=1){
        //add the second half of the data to the first half
        if(threadId < i){
            s_data[threadId] += s_data[threadId+i];
        }
        //synchronize the threads to make sure all the caculation is made
        __syncthreads();
		
    }
    //copy the result back
    if(threadId==0){
        g_data[threadId] = s_data[0];
    }
}

#endif // #ifndef _VECTOR_REDUCTION_KERNEL_H_
