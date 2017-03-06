#-*-coding:UTF-8 -*-

'''
-Created on Jan 25,2017
-@author:Jiajing Guo

-it is the implementation of the paper
 "Sparse and Dense Hybrid Representation via Dictionary 
 Decomposition for Face Recognition"

-函数功能说明：
	get_normalize():对矩阵每一列进行二范数归一化，并返回结果
	Obtain_B_IALM():实现论文中的算法2，获得子字典B
	Update_X_IALM():实现论文中的算法3，更新X
	Update_A_IALM():实现论文中的公式(25)，更新A
	SLR_Decomposition():实现论文中的算法4，进行SLR分解
	SDR_IALM():实现论文中的算法1，计算系数
	SDR_Calculate():计算识别率
'''

from __future__ import division
from numpy import *
from numpy import linalg as la
#from numba import jit
from datetime import datetime
import numpy as np
import solvehomotopy





#对矩阵每一列进行二范数归一化，并返回结果
def get_normalize(data_input):
	data_mul = multiply(data_input,data_input) #矩阵点乘
	data_sum = map(sum,zip(*data_mul)) #矩阵列求和，得到的是一个行向量
	data_sqrt = sqrt(data_sum) #每个元素开根号
	data_rp = tile(data_sqrt,(len(data_input),1)) #由行向量扩充得到一个每行都一样的矩阵
	data_return = data_input/data_rp #归一化处理
	return data_return


#@jit
#实现论文中的算法2，获得子字典B
def Obtain_B_IALM(D,A,X,Lambda,Delta,Iter_max):
	#参数初始化
	B = zeros(shape(D))
	J = zeros(shape(D))
	T = zeros(shape(D))
	Y1 = zeros(shape(D))
	Y2 = zeros(shape(D))
	mu = 1e-3
	mu_max = 1e6
	rho = 1.5
	epsilon = 1e-4
	converged_flag = False
	Iter = 0

	while converged_flag==False :

		Iter += 1

		#用奇异值阈值法更新J
		u,sigma,vt = la.svd(B+Y2/mu)
		a = (sigma>(Lambda/mu)).sum()
		if a>=1 :
			sig2 = diag(sigma[0:a]-Lambda/mu)
		else:
			a = 1
			sig2 = diag(np.array([0]))
		J = dot(dot(u[:,0:a],sig2),vt[0:a])

		#更新B
		b1 = dot((D-A-T),X.T)
		b2 = (dot(Y1,X.T)-Y2)/mu
		b3 = la.inv(dot(X,X.T)+identity(len(X)))
		B = dot((b1+J+b2),b3)

		#更新T
		Resi_e = D-A-dot(B,X)+Y1/mu
		T = maximum(Resi_e-Delta/mu,0)+minimum(Resi_e+Delta/mu,0)

		#更新拉格朗日乘子Y1、Y2
		c = D-A-dot(B,X)-T
		Y1 = Y1+mu*c
		Y2 = Y2+mu*(B-J)

		#更新mu
		mu = min(rho*mu,mu_max)

		#检测是否达到收敛条件
		#if pow(la.norm(c),2)<epsilon and pow(la.norm(B-J),2)<epsilon:
		if la.norm(c)<epsilon and la.norm(B-J)<epsilon:
			converged_flag = True

		if converged_flag==False and Iter>=Iter_max:
			converged_flag = True
			print('Max Iter Reached !')

	B = J
	return B



#@jit
#实现论文中的算法3，更新X
def Update_X_IALM(D,A,B,Tau,Delta,Iter_max):
	#参数初始化
	T = zeros(shape(D))
	Y = zeros(shape(D))
	X = zeros([len(D[0]),len(D[0])])
	mu = 1e-3
	mu_max = 1e6
	rho = 1.5
	epsilon = 1e-4
	converged_flag = False
	Iter = 0

	while converged_flag==False :

		Iter += 1
		
		#更新X
		x1 = la.inv(2*Tau*identity(len(D[0]))+mu*dot(B.T,B))
		x2 = D-A-T+Y/mu
		X = mu*dot(dot(x1,B.T),x2)

		#更新T
		Resi_e = D-A-dot(B,X)+Y/mu
		T = maximum(Resi_e-Delta/mu,0)+minimum(Resi_e+Delta/mu,0)

		#更新拉格朗日乘子Y
		c = D-A-dot(B,X)-T
		Y = Y+mu*c

		#更新mu
		mu = min(rho*mu,mu_max)

		#检测是否达到收敛条件
		#if pow(la.norm(c),2)<epsilon:
		if la.norm(c)<epsilon:
			converged_flag = True

		if converged_flag==False and Iter>=Iter_max:
			converged_flag = True
			print 'Max Iter Reached !'

	return X


#@jit
#实现论文中的公式(25)，更新A
def Update_A_IALM(D,B,X,Eta,Iter_max):
	#参数初始化
	Y = zeros(shape(D))
	A = zeros(shape(D))
	E = zeros(shape(D))
	mu = 1e-3
	mu_max = 1e6
	rho = 1.5
	epsilon = 1e-4
	converged_flag = False
	Iter = 0

	while converged_flag==False :

		Iter += 1

		#更新E
		Resi_e = D-A-dot(B,X)+Y/mu
		E = maximum(Resi_e-Eta/mu,0)+minimum(Resi_e+Eta/mu,0)

		#更新A
		u,sigma,vt = la.svd(D-dot(B,X)-E+Y/mu)
		a = (sigma>(1/mu)).sum()
		if a>=1 :
			sig2 = diag(sigma[0:a]-1/mu)
		else:
			a = 1
			sig2 = diag(np.array([0]))
		A = dot(dot(u[:,0:a],sig2),vt[0:a])

		#更新拉格朗日乘子Y
		c = D-dot(B,X)-A-E
		Y = Y+mu*c

		#更新mu
		mu = min(rho*mu,mu_max)

		#检测是否达到收敛条件
		if (la.norm(c)/la.norm(D-dot(B,X)))<epsilon:
			converged_flag = True

		if converged_flag==False and Iter>=Iter_max:
			converged_flag = True
			print 'Max Iter Reached !'

	return A



#实现论文中的算法4，进行SLR分解
def SLR_Decomposition(D,k,Train_Label,class_num,Lambda,Delta,Tau,Eta,Iter_max):
	#参数初始化
	for i in range(1,class_num+1):
		#初始化X为单位矩阵
		X = identity(len(D[0])) 
		#从 Train_Label 中找出第 i 类的所有下标
		allindex = [idx for idx,a in enumerate(Train_Label) if a==i]
		#第 i 类的所有样本构成子字典Di
		Di = D[:,allindex]
		#对子字典Di进行SVD分解
		[u,sigma,vt] = la.svd(Di)
		#得到子字典Ai
		#Ai = outer(dot(u[:,0],sigma[0]),vt[0])
		Ai = dot(dot(u[:,0:1],diag(sigma[0:1])),vt[0:1])
		#得到字典A的初始化值，类型为array
		A = Ai if i==1 else np.hstack((A,Ai))

	#迭代更新
	for j in range(1,k+1):

		print j

		start = datetime.now()
		#解公式(20)获得子字典B
		B = Obtain_B_IALM(D,A,X,Lambda,Delta,Iter_max)
		stop = datetime.now()
		print(stop-start)

		start = datetime.now()
		#解公式(23)更新X
		X = Update_X_IALM(D,A,B,Tau,Delta,Iter_max)
		stop = datetime.now()
		print(stop-start)

		start = datetime.now()
		#解公式(25)更新子字典A
		A = Update_A_IALM(D,B,X,Eta,Iter_max)
		stop = datetime.now()
		print(stop-start)

	#返回分解结果
	return A,B



#@jit
#实现论文中的算法1，计算系数
def SDR_IALM(A,B,y,Beta,Gamma,Iter_max):
	#参数初始化
	alpha = zeros(len(A[0]))
	x = zeros(len(A[0]))
	e_l = zeros(len(A))
	phi = zeros(len(A))
	xi = 1
	xi_max = 1e6
	rho = 1.5
	epsilon = 5e-3
	converged_flag = False
	Iter = 0


	tolerance = 1e-5
	stoppingCriterion = 3
	maxiter = 5000


	#调用m文件，获取句柄
	#src_handle = My_SolveHomotopy.initialize() 

	while converged_flag==False :

		Iter += 1

		#print 'SDR_Iter',Iter

		#更新e_l
		tmp_e = y-dot(A,alpha)-dot(B,x)+phi/xi
		e_l = maximum(tmp_e-Beta/xi,0)+minimum(tmp_e+Beta/xi,0)
		#print 'e_l',la.norm(e_l)

		
		#start = datetime.now()
		#更新x
		#tmp_x1 = la.inv(2*Gamma*identity(len(A[0]))+xi*dot(B.T,B))
		#tmp_x2 = y-dot(A,alpha)-e_l+phi/xi
		#x = xi*dot(dot(tmp_x1,B.T),tmp_x2)
		#stop = datetime.now()
		#print(stop-start)
		
		#更新x
		tmp_x1 = la.inv(2*Gamma*identity(len(A[0]))+xi*dot(B.T,B))
		tmp_x2 = dot(B.T,(phi+xi*(y-dot(A,alpha)-e_l)))
		x = dot(tmp_x1,tmp_x2)
		#print 'x',la.norm(x)


		
		#更新alpha(同伦法)
		tmp_a = y-dot(B,x)-e_l+phi/xi
		lambda_coef = 1/xi
		alpha = solvehomotopy.SolveHomotopy(A,tmp_a,lambda_coef,tolerance,stoppingCriterion,maxiter)
		#print 'alpha',la.norm(alpha)

	

		#更新拉格朗日乘子phi
		c = y-dot(A,alpha)-dot(B,x)-e_l
		phi = phi+xi*c
		#print 'phi',la.norm(phi)

		#更新xi
		xi = min(rho*xi,xi_max)
		#print 'xi',xi

		#检测是否达到收敛条件
		#if pow(la.norm(c),2)<epsilon:
		if la.norm(c)<epsilon:
			converged_flag = True

		if converged_flag==False and Iter>=Iter_max:
			converged_flag = True
			print 'Max Iter Reached !' 

	return alpha,x,e_l



#计算识别率
def SDR_Calculate(D,Train_Label,Test,Test_Label,class_num,A,B,Beta,Gamma,Iter_max):
	#参数初始化
	alpha_all = []
	x_all = []
	e_all = []
	resd_r = []
	correct_num = 0
	accuracy = 0
	
			

	for i in range(0,len(Test_Label)):
	#for i in range(0,1):
		print 'test_sample:',i
		start = datetime.now()
		
		#获取测试样本
		y = Test[:,i]

		#计算系数
		alpha,x,e_l = SDR_IALM(A,B,y,Beta,Gamma,Iter_max)

		#print 'alpha',alpha


		#恢复出清晰的样本
		y_recovered = y-dot(B,x)-e_l

		#保存系数
		alpha_all.append(alpha)
		x_all.append(x)
		e_all.append(e_l)

		r_tmp = []

		#计算测试样本和各类重构样本之间的残差
		for j in range(1,class_num+1):
			#从 Train_Label 中找出第 j 类的所有下标
			allindex = [idx for idx,a in enumerate(Train_Label) if a==j]
			#第 j 类的所有样本构成子字典Aj
			A_j = A[:,allindex]
			#第 j 类的系数
			alpha_j = alpha[allindex]
			#计算残差
			r_j = la.norm(y_recovered-dot(A_j,alpha_j))
			r_tmp.append(r_j)
		stop = datetime.now()
		print(stop-start)


		#把残差最小的类别作为当前测试样本的类别
		y_label = r_tmp.index(min(r_tmp))+1 
		if y_label == Test_Label[i]:
			correct_num = correct_num+1
		resd_r.append(r_tmp)
	accuracy = correct_num/len(Test_Label) #计算识别准确率
	return accuracy,alpha_all,x_all,e_all,resd_r




