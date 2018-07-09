#coding=utf-8

from net import *
from bn import *
from dataset import train as t
import train
scripts = """

python
from net import demo
tr = demo.demo()
demo.loop(tr,10,10)

"""
def demo(size=100, alpha = 0.1):
	ins = t.getImgs(0,size)
	stds = t.getStds(0,size)
	ins, stds = deal_data()(ins,stds)
	nets = ListNet()
	momentum = 0.99999
	l2c = 0.0001
	net = nets.push(fullnet(28*28,10*10, l2c, momentum))
	net = nets.push(belta_net(net,None, momentum))
	net = nets.push(relunet(net))
	net = nets.push(fullnet(net,10, l2c, momentum))
	net = nets.push(belta_net(net,None, momentum))
	net = nets.push(sigmodnet(net))
	net = nets.push(sqrcost(net))
	tr = train.Train(nets,0.1,ins,stds)
	return tr

def loop(tr,lp,num):
	for i in xrange(lp):
		print(i,tr.loop_update(num))
