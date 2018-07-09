#coding=utf-8
import numpy as np 
from net import FuncCalculator, get_shape, BaseNet


class BatchNormalCalculator(FuncCalculator):
	def __init__(self, epsilon=0.00000001,keep_mean=0.0):
		self.epsilon = epsilon
		self.keep_mean = keep_mean
		self.count = 0
		self.cnt = 0.0
	def _work(self, input_data):
		normal=(input_data-self.E)/np.sqrt(self.Var+self.epsilon)
		return normal 
	def update(self):
		self.count = 0
		self.cnt*=self.keep_mean
	def _feedback(self, input_data, reverse_data):
		axis = range(len(input_data.shape))
		axis.pop(1)
		axis = tuple(axis)
		if self.count != 1:
			mean = input_data.mean(axis=axis, keepdims = True)[0]
			var=input_data.var(axis=axis,keepdims=True)[0]
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
	def _forward(self,input_data):
		axis = range(len(input_data.shape))
		axis.pop(1)
		axis = tuple(axis)
		mean = input_data.mean(axis=axis, keepdims = True)[0]
		var = input_data.var(axis=axis,keepdims=True)[0]
		self.count += 1
		if self.count == 1:
			self.mean = mean
			self.var = var
		num = input_data.shape[0]
		if self.cnt==0.0:
			self.E=mean
			self.Var=var
		else:
			p=self.cnt/(self.cnt+num)
			self.E=self.E*p+mean*(1.0-p)
			self.Var=self.Var*p+var*(1.0-p)
		self.cnt+=num
		normal=(input_data-mean)/np.sqrt(var+self.epsilon)
		return normal 

def batch_normal_net(input, epsilon=0.00000001,keep_mean=0.0):
	shape = get_shape(input)
	calculator = BatchNormalCalculator(epsilon, keep_mean)
	net = BaseNet(calculator)
	net.build_input_shape(*shape)
	net.build_output_shape(*shape)
	return net 



expressions_build = """
(x-mean(x))/sqrt(var(x)+ep) = A / B
x = x1,...,xn
d(mean(x))/dxi = xi/n


vector = v0, v1, ..., vn-1
output = o0, o1, ..., on-1

output = 

vector - mean
______________

sqrt(var + C)


n = len(vetor)
mean = sum(vector) / n
var = sum((vector - mean) ** 2) / n
    = sum((vi - mean) ** 2) / n,   i = 0, 1, ..., n-1

d output
________

  d vi 

= sum ( d oj / d vi ) j = 0, 1, ..., n-1


oj = 

vj - mean
______________

sqrt(var + C)


d oj / d vi = 

[d (vj - mean) / d vi] * sqrt(var + C) - [d (sqrt(var + C)) / d vi] * (vj - mean)
____________________________________________________________________________________
				var + C


d mean / d vi = 1 / n
d (vj - mean) / d vi = d vj / d vi - 1 / n
d (vj - mean) ** 2 / d vi = 2 * (vj - mean) * (d vj / d vi - 1 / n)

d var / d vi = 1/n * sum{2 * (vj - mean) * (d vj / d vi - 1 / n)}
= 2 * sum { (vj - mean) * (d vj / d vi - 1 / n) } / n, j = 0, 1, ..., n-1
= 2 *[ sum { d vj / d vi * (vj - mean) } - sum { (vj - mean) / n} ] / n
= 2 * sum { d vj / d vi * (vj - mean) } / n
= 2 * (vi - mean) / n

d sqrt(var + C) / d vi = 1/(2 * sqrt(var + C)) * d var / dvi 
= (vi - mean) / [ n * sqrt(var + C)]


d oj / d vi = 

[d vj / d vi - 1 / n] * sqrt(var + C) - {(vi - mean) / [ n * sqrt(var + C)]} * (vj - mean)
____________________________________________________________________________________________
				var + C

= 

d vj / d vi * sqrt(var + C) - sqrt(var + C) / n - {(vi - mean) / [ n * sqrt(var + C)]} * (vj - mean)
____________________________________________________________________________________________________
				var + C


sum (rj * d oj / d vi), j = 0, 1, ..., n-1

= sum {

rj * <d vj / d vi * sqrt(var + C) - sqrt(var + C) / n - {(vi - mean) / [ n * sqrt(var + C)]} * (vj - mean)>
____________________________________________________________________________________________________________
				var + C

}

= sum {

rj * <d vj / d vi * sqrt(var + C) - sqrt(var + C) / n - (vi - mean) * (vj - mean) / [ n * sqrt(var + C)]>
____________________________________________________________________________________________________________
				var + C

}

= sum {
rj * <d vj / d vi * sqrt(var + C) - sqrt(var + C) / n - (vi - mean) * (vj - mean) / [ n * sqrt(var + C)]>
}/(var + C)

= sum {
rj * d vj / d vi * sqrt(var + C) - rj * sqrt(var + C) / n - rj * (vi - mean) * (vj - mean) / [ n * sqrt(var + C)]}
}/(var + C)

= {
sum<rj * d vj / d vi * sqrt(var + C)>
- sum<rj * sqrt(var + C) / n>
- sum<rj * (vi - mean) * (vj - mean) / [ n * sqrt(var + C)]>
}
/(var + C)

= {
ri * sqrt(var + C)
- sum<rj> * sqrt(var + C) / n
- sum<rj * (vi - mean) * (vj - mean) / [ n * sqrt(var + C)]>
}
/(var + C)

= {
ri * sqrt(var + C)
- sum<rj> * sqrt(var + C) / n
- sum<rj * (vj - mean)> * (vi - mean)/ [ n * sqrt(var + C)]
}
/(var + C)

= {
ri * sqrt(var + C)
- sum<rj> * sqrt(var + C) / n
- sum<rj * vj - rj * mean> * (vi - mean)/ [ n * sqrt(var + C)]
}
/(var + C)

= sum {

- {(vi - mean) / [ n * sqrt(var + C)]} * (vj - mean)
_____________________________________________________
               var + C

}

= sum {

- (vi - mean) * (vj - mean)
________________________________
(var + C) * n * sqrt(var + C)

}

= 

- (vi - mean) * sum(vj - mean)
________________________________
(var + C) * n * sqrt(var + C)

= 

        - (vi - mean)
________________________________
(var + C) * n * sqrt(var + C)







sum (d oj / d vi), j = 0, 1, ..., n-1

= sum {

- {(vi - mean) / [ n * sqrt(var + C)]} * (vj - mean)
_____________________________________________________
               var + C

}

= sum {

- (vi - mean) * (vj - mean)
________________________________
(var + C) * n * sqrt(var + C)

}

= 

- (vi - mean) * sum(vj - mean)
________________________________
(var + C) * n * sqrt(var + C)

= 

        - (vi - mean)
________________________________
(var + C) * n * sqrt(var + C)



"""