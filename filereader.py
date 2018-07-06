#coding=utf-8
import struct
from sys import stdout
class SymbolReader:
	def filename(self):
		return self._filename

	def __init__(self,filename):
		fp=open(filename,"rb")
		fp.read(4)
		bts=fp.read(4)
		num,=struct.unpack('>i',bts)
		self._num=num
		self._fp=fp
		self._filename=filename

	def get(self,index):
		fp=self._fp
		fp.seek(index+8)
		bts=fp.read(1)
		symbol,=struct.unpack('B',bts)
		return symbol

	def len(self):
		return self._num

	def done(self):
		self._fp.close()

class ImageReader(SymbolReader):
	def cols(self):
		return self._columns

	def rows(self):
		return self._rows

	def __init__(self,filename):
		fp=open(filename,"rb")
		fp.read(4)
		bts=fp.read(4)
		num,=struct.unpack('>i',bts)
		bts=fp.read(4)
		self._rows,=struct.unpack('>i',bts)
		bts=fp.read(4)
		self._columns,=struct.unpack('>i',bts)
		self._num=num
		self._fp=fp
		self._imgsz=self._columns*self._rows
		self._inv=1.0/255.0

	def get(self,index):
		fp=self._fp
		fp.seek(index*self._imgsz+4*4)
		bts=fp.read(self._imgsz)
		r_bytes=struct.unpack(str(self._imgsz)+'B',bts)
		image=[]
		for i in range(self._imgsz):
			image.append((float(r_bytes[i])*self._inv))
		return image

	def show(self,image):
		for i in range(self.rows()):
			for j in range(self.cols()):
				if(image[self.cols()*i+j]>0.0):
					#print(1, end='')
					stdout.write(str(1))
				else:
					#print(0, end='')
					stdout.write(str(0))
			print('')
			

