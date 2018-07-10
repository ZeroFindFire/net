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




filter_source = """
	#define Gfloat double
	__device__ int idint(){
		const int blkId=blockIdx.x+blockIdx.y*gridDim.x+blockIdx.z*gridDim.x*gridDim.y;
		return threadIdx.x+threadIdx.y*blockDim.x+threadIdx.z*blockDim.x*blockDim.y+blkId*blockDim.x*blockDim.y*blockDim.z;
	}
	#define nums_image parameters[0]
	#define nums_filter parameters[1]
	#define depth parameters[2]
	#define height_image parameters[3]
	#define width_image  parameters[4]
	#define height_filter parameters[5]
	#define width_filter parameters[6]
	#define left parameters[7]
	#define top parameters[8]
	#define step_left parameters[9]
	#define step_top parameters[10]
	#define num_left parameters[11]
	#define num_top parameters[12]
	#define step_height_image parameters[13]
	#define step_width_image parameters[14]
	#define step_thread parameters[15]
	#define max_num_thread parameters[16]
	__device__ Gfloat piece_filter(const Gfloat* image, const Gfloat* filter, const int* parameters, left, top){
		Gfloat rst = 0;
		int index_height_image = top;
		int index_width_image = left;
		for (int index_height = 0; index_height < height_filter; ++index_height) {
			index_height_image += step_height_image;
			if (index_height_image < 0 || index_height_image >= height_image) {
				continue;
			}
			for (int index_width = 0; index_width < width_filter; ++index_width) {
				index_width_image += step_width_image;
				if (index_width_image < 0 || index_width_image >= width_image) {
					continue;
				}
				rst += image[index_height_image * width_image + index_width_image] * filter[index_width * filter_width + index_height];
			}
			index_width_image = left;
		}
		return rst;
	}
	__device__ void dv_filter(const Gfloat* images, const Gfloat* filters, const int* parameters, int index_thread, Gfloat* outputs){
		int index_output = index_thread;
		int index_left = index_thread % num_left;
		index_thread /= num_left;
		int index_top = index_thread % num_top;
		index_thread /= num_top;
		int index_filter = index_thread % nums_filter;
		index_thread /= nums_filter;
		int index_image = index_thread;

		index_left = left + index_left * step_left;
		index_top = top + index_top * step_top;
		Gfloat rst = 0;
		images = images + index_image * (width_image * height_image * depth);
		filters = filters + index_filter * (width_filter * height_filter * depth);
		for (int index_depth = 0; index_depth < depth; ++index_depth){
			rst += piece_filter(images , filters, parameters, index_left, index_top);
			images += width_image * height_image;
			filters += width_filter * height_filter;
		}
		outputs[index_output] = rst;
	}
	__global__ void g_filter(const Gfloat* images, const Gfloat* filters, const int* parameters, Gfloat* outputs){
		int index_thread = idint();
		while (index_thread < max_num_thread) {
			dv_filter(images, filters, parameters, index_thread, outputs);
			index_thread += step_thread;
		}
	}
"""

parameters_description = """
	#define nums_image parameters[0]
	#define nums_filter parameters[1]
	#define depth parameters[2]
	#define height_image parameters[3]
	#define width_image  parameters[4]
	#define height_filter parameters[5]
	#define width_filter parameters[6]
	#define left parameters[7]
	#define top parameters[8]
	#define step_left parameters[9]
	#define step_top parameters[10]
	#define num_left parameters[11]
	#define num_top parameters[12]
	#define step_height_image parameters[13]
	#define step_width_image parameters[14]
	#define step_thread parameters[15]
	#define max_num_thread parameters[16]
"""
Valid=0
Same=1
Full=2
ft_mod = SourceModule(filter_source)
g_filter = ft_mod.get_function("g_filter")
def image_filter(image, filter):
	image = np.array(image).astype(np.float64)
	filter = np.array(filter).astype(np.float64)
	image = image.reshape([1]+list(image.shape))
	filter = filter.reshape([1]+list(filter.shape))
	parameters, outshape = build_parameters(image, filters, 100)
	outputs = np.zeros(outshape,dtype = np.float64)
	g_filter(drv.In(image),drv.In(filter),drv.In(parameters),drv.Out(outputs),block = (100,1,1),grid=(1,1))
	return outputs
def build_parameters(images, filters, num_threads, step_left=1, step_top=1, step_width_image=1, step_height_image=1, filter_type = Valid):
	nums_image, depth, height_image, width_image = images.shape 
	nums_filter, depth, height_filter, width_filter = filters.shape
	height_box = height_filter + (height_filter - 1) * step_height_image;
	width_box = width_filter + (width_filter - 1) * step_width_image;
	if filter_type == Valid:
		left, top = 0, 0
		#left + last_index_left * step_left + width_box <= width_image
		last_index_left = (width_image - width_box) / step_left
		num_left = last_index_left + 1
		last_index_top = (height_image - height_box) / step_top
		num_top = last_index_top + 1
	elif filter_type == Same:
		raise Exception("Uncomplete filter type: "+str(filter_type))
		pass 
	elif filter_type == Full:
		left, top = 1 - width_box, 1 - height_box 
		#left + last_index_left * step_left < width_image
		#left + last_index_left * step_left <= width_image-1
		last_index_left = (width_image - 1 - left) / step_left 
		num_left = last_index_left + 1
		last_index_top = (height_image - 1 - top) / step_top
		num_top = last_index_top + 1
	else:
		raise Exception("Unknown filter type: "+str(filter_type))
	rst = [nums_image, nums_filter, depth, height_image, width_image, height_filter, width_filter]
	rst += [left, top, step_left, step_top, num_left, num_top, step_height_image, step_width_image]
	total_num = nums_image * nums_filter * num_left * num_top
	max_num_thread = total_num
	step_thread = num_threads
	rst += [step_thread, max_num_thread]
	rst = np.array(rst,dtype = np.int32)
	return rst,[nums_image,nums_filter,num_top,num_left]
