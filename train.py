#coding=utf-8
import numpy as np

class Train(object):
	def learn_work(self,ins,stds):
		net=self.net
		net.forward(ins)
		net.feedback(ins,stds)
		return None
	def learn_summary(self,weight):
		net=self.net
		net.update(weight)
	def empty_summary(self,weight):
		pass
	def work_work(self,ins,stds):
		net=self.net
		return net.work(ins)
	def cost_work(self,ins,stds):
		net=self.net
		outs = net.work(ins)
		dff = net.cost(outs,stds)
		return dff 
	def update(self,start=0,end=-1,minbh=-1):
		return self.loop(self.learn_work,self.learn_summary,start,end,minbh)
	def work(self,start=0,end=-1,minbh=-1):
		return self.loop(self.work_work,self.empty_summary,start,end,minbh)
	def cost(self,start=0,end=-1,minbh=-1):
		return self.loop(self.cost_work,self.empty_summary,start,end,minbh)
	def loop_update(self,num,start=0,end=-1,minbh=-1):
		for i in xrange(num):
			self.update(start,end,minbh)
		return self.cost(start,end,minbh).sum()
	def loop(self,work,summary,start=0,end=-1,minbh=-1):
		alpha=self.alpha
		l=len(self.ins)
		start=max(start,0)
		end=min(end,l)
		if end==-1:
			end=l
		l=end-start
		if minbh==-1:
			minbh=l
		if start+minbh>end:
			return None
		num=1.0*l/minbh
		net=self.net
		outs=[]
		for base in xrange(start, end, minbh):
			ins=self.ins[start:end]
			stds=self.stds[start:end]
			touts=work(ins,stds)
			if touts is not None:
				outs.append(touts)
			base+=minbh
		summary(alpha*1.0/num)
		if len(outs)==0:
			outs=None
		else:
			outs=np.array(outs)
		return outs
	def set_io(self,ins,stds):
		self.ins=ins 
		self.stds=stds 
	def set_parm(self,alpha=None,l2c=None,momentum=None):
		if alpha is not None:
			self.alpha=alpha
		if l2c is not None:
			self.l2c=l2c
		if momentum is not None:
			self.momentum=momentum

	def __init__(self,net = None,alpha=1.0,ins=None,stds=None):
		self.alpha=alpha
		self.net=net
		self.ins = ins
		self.stds=stds

