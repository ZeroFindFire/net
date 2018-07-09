#coding=utf-8

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
source = """
	#define Gfloat double
	__device__ int idint(){
		const int blkId=blockIdx.x+blockIdx.y*gridDim.x+blockIdx.z*gridDim.x*gridDim.y;
		return threadIdx.x+threadIdx.y*blockDim.x+threadIdx.z*blockDim.x*blockDim.y+blkId*blockDim.x*blockDim.y*blockDim.z;
	}
	__global__ void test(Gfloat* outs, const Gfloat* ins, const int* sizes)
	{
		int thdId=idint();
		if (thdId >= sizes[0]) {
			return;
		}
		outs[thdId] *= outs[thdId] * ins[thdId];
	}
"""
mod = SourceModule(source)
test = mod.get_function("test")
a = np.arange(100,dtype = np.float64)
b = np.arange(100,dtype = np.float64)
outs = drv.InOut(a)
ins = drv.In(b)
sizes = drv.In(np.array([100],dtype=np.int32))
test(outs,ins,sizes,block = (100,1,1),grid=(1,1))
print "resule:",a