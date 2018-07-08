#coding=utf-8
import numpy as np 
from net import FuncCalculator


class BatchNormalCalculator(FuncCalculator):
	def __init__(self, shape, epsilon=0.00000001,keep_mean=0.0):
		self.epsilon = epsilon
		self.keep_mean = keep_mean
		self.count = 0
		data.cnt = 0.0
	def _work(self, input_data):
		normal=(input_data-self.E)/np.sqrt(self.Var+data.epsilon)
		return normal 
	def update(self):
		self.count = 0
		data.cnt*=data.keep_mean
	def feedback(self, input_data, reverse_data):
		axis = range(len(input_data.shape))
		axis.pop(1)
		if self.count != 1:
			mean = input_data.mean(axis=axis, keepdims = True)[0]
			var=ins.var(axis=axis,keepdims=True)[0]
		else:
			mean = self.mean
			var=self.var
		self.count -= 1
		sqrt_var = np.sqrt(var + self.epsilon)
		n = 1
		shape = input_data.shape
		for num in shape:
			n *= num
		n /= shape[1]
		expression="""
			vector  = v0, v1, ..., vn-1
			output  = o0, o1, ..., on-1
			reverse = r0, r1, ..., rn-1

			output = 

			vector - mean
			______________

			sqrt(var + C)

			n = len(vetor)
			mean = sum(vector) / n
			var = sum((vector - mean) ** 2) / n
			    = sum((vi - mean) ** 2) / n,   i = 0, 1, ..., n-1

			sum (rj * d oj / d vi), j = 0, 1, ..., n-1
			= {
			ri * sqrt(var + C)
			- sum<rj> * sqrt(var + C) / n
			- sum<rj * (vj - mean)> * (vi - mean)/ [ n * sqrt(var + C)]
			}/(var + C)

			standard expression in python:
			exp0 = reverse_data * sqrt_var 
			exp1 = reverse_data.sum(axis = axis, keepdims = True) * sqrt_var / n 
			exp2 = (reverse_data * (input_data - mean)).sum(axis = axis, keepdims = True) * (input_data - mean) / (n * sqrt_var)
			exp = (exp0 - exp1 - exp2) / (sqrt_var ** 2)
			return exp
		"""
		inv_sqrt = 1.0 / sqrt_var
		exp0 = reverse_data * inv_sqrt 
		exp1 = reverse_data.mean(axis = axis, keepdims = True) * inv_sqrt
		exp2 = (reverse_data * (input_data - mean)).mean(axis = axis, keepdims = True) * (input_data - mean) * inv_sqrt ** 3
		exp = exp0 - exp1 - exp2
		return exp
	def forward(self,input_data):
		axis = range(len(input_data.shape))
		axis.pop(1)
		mean = input_data.mean(axis=axis, keepdims = True)[0]
		var = ins.var(axis=axis,keepdims=True)[0]
		self.count += 1
		if self.count == 1:
			self.mean = mean
			self.var = var
		num = input_data[0]
		if data.cnt==0.0:
			data.E=mean
			data.Var=var
		else:
			p=data.cnt/(data.cnt+num)
			data.E=data.E*p+mean*(1.0-p)
			data.Var=data.Var*p+var*(1.0-p)
		data.cnt+=num
		normal=(input_data-mean)/np.sqrt(var+data.epsilon)
		return normal 
