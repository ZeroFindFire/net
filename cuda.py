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
		__syncthreads();
		__shared__ int sharedata[128];
	}
"""
mod = SourceModule(source)
test = mod.get_function("test")
size = 100
a = np.arange(size,dtype = np.float64)
b = np.arange(size,dtype = np.float64)
outs = drv.InOut(a)
ins = drv.In(b)
sizes = drv.In(np.array([size],dtype=np.int32))
test(outs,ins,sizes,block = (size,1,1),grid=(1,1))
print("resule:",a.shape)

def img2arr(img):
	img = np.array(img).astype(np.float)
	img /= 256
	shape = list(img.shape)
	if len(shape)==2:
		img.shape = [1]+shape
		return img 
	elif len(shape) != 3:
		raise Exception("Unknow shape of image: "+str(shape))
	depth = shape[2]
	shape = [depth] + shape[:-1]
	outs = np.zeros(shape,dtype = img.dtype)
	for i in xrange(depth):
		outs[i,:,:]=img[:,:,i]
	return outs 

def arr2img(img):
	img = np.array(img)
	img *= 256
	img = img.astype(np.uint8)
	shape = list(img.shape)
	if len(shape)==2:
		return img
	elif len(shape) != 3:
		raise Exception("Unknow shape of array: "+str(shape))
	depth = shape[0]
	shape = shape[1:]+[depth]
	outs = np.zeros(shape,dtype = img.dtype)
	for i in xrange(depth):
		outs[:,:,i]=img[i,:,:]
	if outs.shape[-1] == 1:
		outs.shape = outs.shape[:-1]
	return outs

def show(img, tmp_path = r"E:\tmp.bmp",wait_time = 3.0, clean = False):
	from PIL import Image
	img=Image.fromarray(img)
	img.save(tmp_path)
	import os 
	os.system(r"start %s"%(tmp_path,))
	if not clean:
		return 
	import time 
	time.sleep(wait_time)
	del_cmd = r"del /F /Q %s"%(tmp_path,)
	os.system(del_cmd)
	


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
	__device__ Gfloat piece_filter(const Gfloat* image, const Gfloat* filter,
		const int* parameters, const int base_left, const int base_top){
		Gfloat rst = 0;
		int index_height_image = base_top;
		int index_width_image = base_left;
		for (int index_height_filter = 0; index_height_filter < height_filter; ++index_height_filter) {
			index_height_image += step_height_image;
			if (index_height_image < 0 || index_height_image >= height_image) {
				continue;
			}
			for (int index_width_filter = 0; index_width_filter < width_filter; ++index_width_filter) {
				index_width_image += step_width_image;
				if (index_width_image < 0 || index_width_image >= width_image) {
					continue;
				}
				rst += image[index_height_image * width_image + index_width_image] 
					* filter[index_height_filter * width_filter + index_width_filter];
			}
			index_width_image = base_left;
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
	__device__ void filter_point_feedback(const Gfloat* images, Gfloat* alters, const int* parameters, 
		const Gfloat* reverses, int index_thread){
		int index_width_filter = index_thread % width_filter;
		index_thread /= width_filter;
		int index_height_filter = index_thread % height_filter;
		index_thread /= height_filter;
		int index_depth = index_thread % depth;
		index_thread /= depth;
		int index_filter = index_thread;
		
		Gfloat rst = 0;
		reverses += num_left * num_top * index_filter;
		images += width_image * height_image * index_depth;
		for (int index_image = 0; index_image < nums_image; ++index_image) {
			for (int index_top = 0; index_top < num_top; ++index_top) {
				int base_top = top + index_top * step_top;
				int index_height_image = base_top + step_height_image * index_height_filter;
				if (index_height_image < 0 || index_height_image >= height_image) {
					continue;
				}
				for (int index_left = 0; index_left < num_left; ++index_left) {
					int base_left = left + index_left * step_left;
					int index_width_image = base_left + step_width_image * index_width_filter;
					if (index_width_image < 0 || index_width_image >= width_image) {
						continue;
					}
					rst += images[index_height_image * width_image + index_width_image] * reverses[index_top * num_left + index_left];
				}
			}
			reverses += num_left * num_top * nums_filter;
			images += width_image * height_image * depth;
		}
		alters[((index_filter * depth + index_depth) * height_filter + index_height_filter) * width_filter + index_width_filter] += rst;
	}
	__device__ void filter_input_feedback(Gfloat* outputs, const Gfloat* filters, const int* parameters, 
		const Gfloat* reverses, int index_thread){
		int index_width_image = index_thread % width_image;
		index_thread /= width_image;
		int index_height_image = index_thread % height_image;
		index_thread /= height_image;
		int index_depth = index_thread % depth;
		index_thread /= depth;
		int index_image = index_thread;
		reverses += (index_image * depth + index_depth) * width_image * height_image;
		filters += width_filter * height_filter * index_depth;
		Gfloat rst = 0;
		for (int index_filter = 0; index_filter < nums_filter; ++index_filter) {
			for (int index_height_filter = 0; index_height_filter < height_filter; ++index_height_filter) {
				int base_height_image = index_height_image - index_height_filter * step_height_image;
				int index_height_reverse = (base_height_image - top) / step_top;
				if (index_height_reverse < 0 || index_height_reverse >= num_top){
					continue;
				}
				for (int index_width_filter = 0; index_width_filter < width_filter; ++index_width_filter) {
					int base_width_image = index_width_image - index_width_filter * step_width_image;
					int index_width_reverse = (base_width_image - left) / step_left;
					if (index_width_reverse < 0 || index_width_reverse >= num_left){
						continue;
					}
					rst += filters[index_height_filter * width_filter + index_width_filter] 
						* reverses[index_height_reverse * num_left + index_width_reverse];
				}
			}
			filters += width_filter * height_filter * depth;
		}
		outputs[((index_image * depth + index_depth) * height_image + index_height_image) * width_image + index_width_image] = rst;
	}
	__global__ void g_filter_feedback(const Gfloat* images, const Gfloat* filters, const int* parameters, 
		const Gfloat* reverses, Gfloat* alters, Gfloat* outputs) {
		int index_thread = idint();
		int total_filters = nums_filter * depth * width_filter * height_filter;
		while (index_thread < max_num_thread) {
			if (index_thread < total_filters) {
				filter_point_feedback(images, alters, parameters,reverses, index_thread);
			} else {
				filter_input_feedback(outputs, filters, parameters, reverses, index_thread - total_filters);
			}
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
ft_mod = None#SourceModule(filter_source)
g_filter = None#ft_mod.get_function("g_filter")
def image_filter(image, filter, nums_thread = 100):
	global ft_mod 
	global g_filter 
	if ft_mod is None:
		ft_mod = SourceModule(filter_source)
		g_filter = ft_mod.get_function("g_filter")
	image = np.array(image).astype(np.float64)
	filter = np.array(filter).astype(np.float64)
	if len(image.shape) != 4:
		image = image.reshape([1]+list(image.shape))
	if len(filter.shape) != 4:
		filter = filter.reshape([1]+list(filter.shape))
	parameters, outshape = build_parameters(image, filter, nums_thread)
	#print("outshape:",outshape,"from shape", image.shape)
	outputs = np.zeros(outshape,dtype = np.float64)
	g_filter(drv.In(image),drv.In(filter),drv.In(parameters),drv.Out(outputs),block = (nums_thread,1,1),grid=(1,1))
	return outputs
def build_parameters(images, filters, nums_thread, step_left=1, step_top=1, step_width_image=1, step_height_image=1, filter_type = Valid):
	nums_image, depth, height_image, width_image = images.shape 
	nums_filter, depth, height_filter, width_filter = filters.shape
	height_box = 1 + (height_filter - 1) * step_height_image;
	width_box = 1 + (width_filter - 1) * step_width_image;
	if filter_type == Valid:
		left, top = 0, 0
		#left + last_index_left * step_left + width_box <= width_image
		last_index_left = (width_image - width_box) / step_left
		num_left = last_index_left + 1
		last_index_top = (height_image - height_box) / step_top
		num_top = last_index_top + 1
	elif filter_type == Same:
		raise Exception("Uncomplete filter type: Same("+str(filter_type)+")")
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
	step_thread = min(nums_thread,total_num)
	rst += [step_thread, max_num_thread]
	rst = np.array(rst,dtype = np.int32)
	return rst,[nums_image,nums_filter,num_top,num_left]
