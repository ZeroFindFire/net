#coding=utf-8
import filereader
class Dir:
	def __init__(self,img,sym):
		self.img=filereader.ImageReader(img)
		self.sym=filereader.SymbolReader(sym)
	def getImg(self,id):
		return self.img.get(id)
	def getStd(self,id):
		out=[0.0,0.0,0.0,0.0,0.0,
			0.0,0.0,0.0,0.0,0.0]
		out[self.sym.get(id)]=1.0
		return out
	def getImgs(self,base,num):
		out=[]
		for i in xrange(base,base+num):
			out.append(self.getImg(i))
		return out
	def getStds(self,base,num):
		out=[]
		for i in xrange(base,base+num):
			out.append(self.getStd(i))
		return out
	@staticmethod
	def argmax(vals):
		id=0
		for i in xrange(len(vals)):
			if vals[i]>vals[id]:
				id=i 
		return id
	def check(self,id,vals):
		cid=self.sym.get(id)
		vid=self.argmax(vals)
		if cid != vid:
			return False 
		if vals[vid]<0.6:
			return False
		return True 
	def checks(self,base,num,vals):
		out = 0
		for i in xrange(num):
			out += self.check(i+base,vals[i])
		return out
root="/home/zlinux/test/python/ai/data/"
root="D:/MyWork/2015/python/ai_bak/data/"
root="E:/zff/works/chess/python/data/"
train=Dir(root+"train-images.idx3-ubyte",root+"train-labels.idx1-ubyte")
test=Dir(root+"t10k-images.idx3-ubyte",root+"t10k-labels.idx1-ubyte")
