
import gpu_IoU
#from gpu_IoU import gpu_it_IoU
import numpy as np
import matplotlib.pyplot as plt
from cytool import iou



"""
R1=np.array([10,5,np.deg2rad(60),80,40])
R2=np.array([5,5,np.deg2rad(45),60,30])
(0.48953234350722175, array([[ 0.52811736]], dtype=float32))

R1=np.array([20,5,np.deg2rad(60),80,40])
R2=np.array([5,5,np.deg2rad(45),60,30])
(0.48953234350722175, array([[ 0.52811736]], dtype=float32))

R1=np.array([20,5,np.deg2rad(60),80,40])
R2=np.array([15,51,np.deg2rad(45),60,30])
(0.26193694149203622, array([[ 0.50330728]], dtype=float32))

R1=np.array([20,-15,np.deg2rad(60),80,40])
R2=np.array([15,51,np.deg2rad(45),60,30])
(0.14906816611752807, array([[ 0.34989202]], dtype=float32))

R1=np.array([20,-15,np.deg2rad(60),20,10])
R2=np.array([15,51,np.deg2rad(45),60,30])
(0.0, array([[ 0.0373444]], dtype=float32))
"""


"""
# fixed I

R1=np.array([20,-15,np.deg2rad(60),20,10])
R2=np.array([15,51,np.deg2rad(45),60,30])
(0.0, array([[ 0.]], dtype=float32))

R1=np.array([20,5,np.deg2rad(60),80,40])
R2=np.array([5,5,np.deg2rad(45),60,30])
(0.48953234350722175, array([[ 0.50330728]], dtype=float32))

R1=np.array([20,5,np.deg2rad(60),80,40])
R2=np.array([15,51,np.deg2rad(45),60,30])
(0.26193694149203622, array([[ 0.32415253]], dtype=float32))

R1=np.array([20,-15,np.deg2rad(60),80,40])
R2=np.array([15,51,np.deg2rad(45),60,30])
(0.14906816611752807, array([[ 0.1682243]], dtype=float32))

R1=np.array([20,-15,np.deg2rad(60),20,10])
R2=np.array([15,51,np.deg2rad(45),60,30])
(0.0, array([[ 0.]], dtype=float32))

R1=np.array([10,5,np.deg2rad(60),80,40])
R2=np.array([5,5,np.deg2rad(45),60,30])
(0.48953234350722175, array([[ 0.55376011]], dtype=float32))
"""



"""
#  fixed II
iou_gpu = gpu_IoU.gpu_it_IoU(np.reshape(R1,(1,5)),np.reshape(R2,(1,5)),10);#reload(gpu_IoU);
iou_cpu = iou(R1,R2,10)
iou_cpu,iou_gpu




R1=np.array([20,5,np.deg2rad(60),80,40])
R2=np.array([5,5,np.deg2rad(45),60,30])
(0.48953234350722175, array([[ 0.48721001]], dtype=float32))  #***1

R1=np.array([20,5,np.deg2rad(60),80,40])
R2=np.array([15,51,np.deg2rad(45),60,30])
(0.26193694149203622, array([[ 0.26968005]], dtype=float32))

R1=np.array([20,-15,np.deg2rad(60),80,40])
R2=np.array([15,51,np.deg2rad(45),60,30])
(0.14906816611752807, array([[ 0.14889705]], dtype=float32))


R1=np.array([10,5,np.deg2rad(60),80,40])
R2=np.array([5,5,np.deg2rad(45),60,30])
(0.48953234350722175, array([[ 0.54511744]], dtype=float32))  #***1    gpu version more accuent
"""





R1=np.array([20,-15,np.deg2rad(60),20,10])
R2=np.array([15,51,np.deg2rad(45),60,30])

iou_gpu = gpu_IoU.gpu_it_IoU(np.reshape(R1,(1,5)),np.reshape(R2,(1,5)),10);#reload(gpu_IoU);
iou_cpu = iou(R1,R2,10)
iou_cpu,iou_gpu




def cp2p(cp,alpha,rh,rw):
	"""
	            return the four vertices gaven the centerPoint and angle, etc
	"""
	dirH=np.array([np.cos(alpha),np.sin(alpha)])
	dirW=np.array([np.sin(alpha),-np.cos(alpha)])
	return np.array( [ _r+cp for _r in  [rh*dirH+rw*dirW, -rh*dirH+rw*dirW, -rh*dirH-rw*dirW, rh*dirH-rw*dirW]   ]).transpose()


def get_tl_p(p_array):
  """ p_list : 4 x 2 np.array"""
  return p_array.min(axis=1)

def get_rb_p(p_array):
  return p_array.max(axis=1)


def drawAngleBox(vecP,color='red'):
	plt.plot(np.array( [vecP[0,0], vecP[0,1] ]),np.array([vecP[1,0],vecP[1,1]]),color=color,marker='*')
	plt.plot(np.array( [vecP[0,1], vecP[0,2] ]),np.array([vecP[1,1],vecP[1,2]]),color=color,marker='*')
	plt.plot(np.array( [vecP[0,2], vecP[0,3] ]),np.array([vecP[1,2],vecP[1,3]]),color=color,marker='*')
	plt.plot(np.array( [vecP[0,0], vecP[0,-1] ]),np.array([vecP[1,0],vecP[1,-1]]),color=color,marker='*')
	plt.axis('equal')



#  720.          480.           -1.04719755   69.2820323    23.09401077
# 7.08000000e+02   4.82000000e+02  -1.11225998e+00    5.10000000e+01   1.60000000e+01
R1=np.array([10,5,np.deg2rad(60),80,40])
R2=np.array([5,5,np.deg2rad(45),60,30])



iou_v = iou(R1,R2,10)





#R1=np.array([720.,480.,-1.04719755,69.2820323,23.09401077])
#R2=np.array([7.08000000e+02,4.82000000e+02,-1.11225998e+00,5.10000000e+01,1.60000000e+01])

drawAngleBox(cp2p(R1[:2],R1[2],R1[3],R1[4]))
drawAngleBox(cp2p(R2[:2],R2[2],R2[3],R2[4]),'green')
plt.xlabel('$x$')
plt.ylabel('$y$')
iou_v = iou(R1,R2,10)
plt.title(iou_v)
plt.show(block=False)



plt.figure()
R1=np.array([10,5,np.deg2rad(60-60),80,40])
R2=np.array([5,5,np.deg2rad(45-60),60,30])

#R1=np.array([720.,480.,-1.04719755,69.2820323,23.09401077])
#R2=np.array([7.08000000e+02,4.82000000e+02,-1.11225998e+00,5.10000000e+01,1.60000000e+01])

drawAngleBox(cp2p(R1[:2],R1[2],R1[3],R1[4]))
drawAngleBox(cp2p(R2[:2],R2[2],R2[3],R2[4]),'green')


tl_p1=get_tl_p(cp2p(R1[:2],R1[2],R1[3],R1[4]))
rb_p2=get_rb_p(cp2p(R1[:2],R1[2],R1[3],R1[4]))

#print cp2p(R1[:2],R1[2],R1[3],R1[4]), tl_p1, rb_p2

x=np.linspace(tl_p1[0], rb_p2[0], 10)
y=np.linspace(tl_p1[1], rb_p2[1], 10)
X,Y = np.meshgrid(x,y)
plt.plot(X.flatten(), Y.flatten(),ls='None',marker='*')
plt.xlabel('$x$')
plt.ylabel('$y$')



plt.figure()
R1=np.array([10,5,np.deg2rad(60-45),80,40])
R2=np.array([5,5,np.deg2rad(45-45),60,30])

#R1=np.array([720.,480.,-1.04719755,69.2820323,23.09401077])
#R2=np.array([7.08000000e+02,4.82000000e+02,-1.11225998e+00,5.10000000e+01,1.60000000e+01])

drawAngleBox(cp2p(R1[:2],R1[2],R1[3],R1[4]))
drawAngleBox(cp2p(R2[:2],R2[2],R2[3],R2[4]),'green')


iou_v = iou(R1,R2,10)
plt.title(iou_v)
plt.xlabel('$x$')
plt.ylabel('$y$')

plt.show(block=False)



plt.figure()
R1=np.array([30.,40,0,20,20])
R2=np.array([50.,60,0,20,20])

#R1=np.array([720.,480.,-1.04719755,69.2820323,23.09401077])
#R2=np.array([7.08000000e+02,4.82000000e+02,-1.11225998e+00,5.10000000e+01,1.60000000e+01])

drawAngleBox(cp2p(R1[:2],R1[2],R1[3],R1[4]))
drawAngleBox(cp2p(R2[:2],R2[2],R2[3],R2[4]),'green')


iou_v = iou(R1,R2,10)
plt.title(iou_v)
plt.xlabel('$x$')
plt.ylabel('$y$')

plt.show(block=False)

s=raw_input('press any key to close...')

