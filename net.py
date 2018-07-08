#coding=utf-8

import numpy as np

ReadMe="""
神经网络函数类，
功能函数：
work: 将输入数据集处理，输出
feedback：传入输入数据集，反馈数据集和输出数据集，待修改参数，返回反馈数据集和修改后的参数，修改参数可能被修改
learn_work：可能与word不同
"""

class BaseCalculator(object):
	def work(self, input_data, weights):
		return input_data 

	def feedback(self, input_data, reverse_data, weights, alters):
		return reverse_data
	def forward(self,input_data, weights):
		return self.work(input_data, weights)
	def update(self):
		pass 
class L2CCalculator(BaseCalculator):
	def __init__(self, calculator, l2c=0.0):
		self.l2c = l2c 
		self.calculator = calculator
	def work(self, input_data,weights):
		return self.calculator.work(input_data,weights)
	def feedback(self, input_data, reverse_data, weights, alters):
		reverse_data = self.calculator.feedback(input_data, reverse_data, weights, alters)
		if self.l2c != 0.0:
			alters += weights * self.l2c 
		return reverse_data 
	def forward(self,input_data, weights):
		return self.calculator.forward(input_data, weights)
class FuncCalculator(BaseCalculator):
	def _work(self, input_data):
		return input_data 
	def _feedback(self, input_data, reverse_data):
		return reverse_data
	def _forward(self,input_data):
		return self._work(input_data)

	def work(self, input_data,weights):
		return self._work(input_data)
	def feedback(self, input_data, reverse_data, weights, alters):
		return self._feedback(input_data, reverse_data)
	def forward(self,input_data, weights):
		return self._forward(input_data)

class SigmodCalculator(FuncCalculator):
	def _work(self,vt):
		out=np.exp(-1*vt)
		out+=1.0
		tmp=out.reshape(1,out.size)
		tmp[0]=1.0/tmp[0]
		return out
	
	def _feedback(self,vt,rvs):
		out=self._work(vt)
		out*=(1.0-out)
		out*=rvs
		return out

class ReLuCalculator(FuncCalculator):
	def _work(self,vt):
		out = np.maximum(vt,0)
		return out
	def _feedback(self,sums,rvs):
		out=(sums>0).astype(sums.dtype)
		out*=rvs
		return out

class FullCalculator(BaseCalculator):
	def work(self, input_data, weights):
		output_data=np.dot(input_data,weights)
		return output_data 

	def feedback(self, input_data, reverse_data, weights, alters):
		num, input_size=input_data.shape
		for n in xrange(num):
			tmp=reverse_data[n]*input_data[n].reshape(input_size,1)
			alters+=tmp
		reverse_data=np.dot(reverse_data,weights.T);
		return reverse_data

class LinearCalculator(BaseCalculator):
	def work(self, input_data, weights):
		output_data=input_data * weights
		return output_data 

	def feedback(self, input_data, reverse_data, weights, alters):
		input_shape = input_data.shape[1:]
		weight_shape = wegiths.shape 
		updates = input_data * reverse_data
		axis = []
		for i in xrange(len(weight_shape)):
			if weight_shape[i] != input_shape[i]:
				axis.append(i+1)
		axis=tuple([0]+axis)
		updates = (input_data * reverse_data).sum(axis=axis, keepdims = True)[0]
		alters += updates
		reverse_data=reverse_data*weights
		return reverse_data
def linear_net(input, axis = None, l2c = 0.0, momentum = 1.0):
	shape = get_shape(input)
	if axis is not None:
		for i in axis:
			shape[i]=1
	weights = np.random.random(shape) - 0.5
	calculator = L2CCalculator(LinearCalculator(),l2c)
	net = BaseNet(calculator,weights,momentum)
	net.build_input_shape(*shape)
	net.build_output_shape(*shape)
	return net
class BaseNet(object):
	def __init__(self, calculator = BaseCalculator(), weights = None, momentum = 1.0, alters = None):
		self.calculator = calculator
		self.weights = weights 
		self.alters = alters
		if weights is not None and alters is None:
			alters = np.zeros(weights.shape,dtype=weights.dtype)
		self.momentum = momentum
		self.output = self.work
	def build_shape(self, *sizes):
		return list(sizes)
	def build_input_shape(self, *sizes):
		self.input_shape = self.build_shape(*sizes)
	def build_output_shape(self, *sizes):
		self.input_shape = self.build_shape(*sizes)
	def input_reshape(self, input_data):
		size = sum(self.input_shape)
		num = input_data.size/size 
		input_data.shape=[num]+self.input_shape
		return input_data
	def feedback_reshape(self, feedback_data):
		size = sum(self.output_shape)
		num = feedback_data.size/size 
		feedback_data.shape=[num]+self.output_shape
		return feedback_data

	def work(self, input_data):
		self.input_reshape(input_data)
		return self.calculator.work(input_data, self.weights)
	def forward(self, input_data):
		self.input_reshape(input_data)
		return self.calculator.forward(input_data, self.weights)
	def feedback(self, input_data, reverse_data):
		self.input_reshape(input_data)
		self.feedback_reshape(reverse_data)
		return self.calculator.feedback( input_data, reverse_data, self.weights, self.alters)
	def update(self, weight=1.0):
		self.calculator.update()
		weights,alters=self.weights,self.alters
		if alters is None:
			return 
		weights*=self.momentum
		weights-=alters*weight
		alters[:]=0
	def cost(self, output_data, stand_data):
		return stand_data
	def compute(self, input_data, reverse_data):
		self.forward(input_data)
		return self.feedback(input_data, reverse_data)
	def run_update(self,input_data, reverse_data,weight=1.0):
		rst = self.compute(input_data, reverse_data)
		self.update(weight)
		return rst 
class ListNet(BaseNet):
	def __init__(self):
		self.output = self.work
		self.nets = list()
		self.outputs = list()
		self.add=self.push
		self.append=self.push
	def __getattr__(self,name):
		if name in ['input_shape','output_shape']:
			if name == 'input_shape':
				return self.nets[0].input_shape 
			return self.nets[-1].output_shape  
		return object.__getattribute__(self,name)
	def push(self, net):
		self.nets.append(net)
		return net
	def update(self, weight=1.0):
		for net in self.nets:
			net.update(weight)
	def work(self, input_data):
		for net in self.nets:
			input_data=net.work(input_data)
		return input_data 
	def cost(self, output_data, stand_data):
		return self.nets[-1].cost(output_data, stand_data)
	def forward(self, input_data):
		self.outputs=[]
		for net in self.nets:
			input_data=net.forward(input_data)
			self.outputs.append(input_data)
		self.outputs.pop(-1)
		return input_data 
	def feedback(self, input_data, reverse_data):
		self.outputs.insert(0,input_data)
		for net in self.nets[::-1]:
			input_data = self.outputs.pop(-1)
			reverse_data = net.feedback(input_data,reverse_data)
		if len(self.outputs)!= 0:
			raise Exception("error in listnet feedback, self.outputs != 0")
		return reverse_data

def get_size(input):
	if hasattr(input,'output_shape'):
		size = 1
		for sz in input.output_shape:
			size *= sz 
	else:
		size = input 
	return size 
def get_shape(input):
	if hasattr(input,'output_shape'):
		shape = input.output_shape 
	else:
		shape = input 
	return shape 

def fullnet(input, output_size, l2c = 0.0, momentum = 1.0):
	input_size = get_size(input)
	weights = np.random.random([input_size, output_size]) - 0.5
	calculator = L2CCalculator(FullCalculator(),l2c)
	net = BaseNet(calculator,weights,momentum)
	net.build_input_shape(input_size)
	net.build_output_shape(output_size)
	return net 

def funcnet(input, calculator = FuncCalculator()):
	size = get_size(input)
	net = BaseNet(calculator)
	net.build_input_shape(size)
	net.build_output_shape(size)
	return net 

def sigmodnet(input):
	return funcnet(input, SigmodCalculator())

def relunet(input):
	return funcnet(input, ReLuCalculator())

class CostCalculator(FuncCalculator):
	def _cost(self, input_data, stand_data):
		return stand_data
	def cost(self, input_data, stand_data):
		return self._cost(input_data, stand_data)
	def feedback(self, input_data, reverse_data, weights, alters):
		inv_num = 1.0 / reverse_data.shape[0]
		reverse_data = self._feedback(input_data, reverse_data)
		return reverse_data * inv_num 

class LogCost(CostCalculator):
	def _cost(self,outs,stds):
		wk=outs*stds+(1.0-stds)
		wk=-stds*np.log(outs)
		return wk
	def _feedback(self,outs,stds):
		wk=(outs+(1.0-stds))
		wk=-stds/wk
		return wk

class SqrCost(CostCalculator):
	def _cost(self,outs,stds):
		wk= 0.5*((outs-stds)**2)
		return wk
	def _feedback(self,outs,stds):
		diff=(outs-stds)
		return diff

class deal_data(object):
	def __init__(self, dtype=np.float ):
		self.dtype = dtype 
	def __call__(self,*args):
		outs = []
		for arg in args:
			obj = np.asarray(arg, dtype = self.dtype)
			outs.append(obj)
		if len(outs)==1:
			outs = outs[0]
		return outs 

class CostNet(BaseNet):
	def cost(self, input_data, stand_data):
		self.input_reshape(input_data)
		self.feedback_reshape(stand_data)
		return self.calculator.cost(input_data,stand_data)

def costnet(input, calculator = CostCalculator()):
	size = get_size(input)
	net = CostNet( calculator)
	net.build_input_shape(size)
	net.build_output_shape(size)
	return net 

def sqrcost(input):
	return costnet(input, SqrCost())

def logcost(input):
	return costnet(input,LogCost())