#-*-coding:UTF-8 -*-

'''
-Created on Jan 20,2017
-@author:Jiajing Guo

-函数功能说明：
	get_normalize():对矩阵每一列进行二范数归一化，并返回结果
	matrix_mul_vector():矩阵和向量相乘
	src_calculate():计算准确率、系数、残差
'''

from __future__ import division
#from numba import autojit
from numpy import *
from numpy.linalg import *
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


#@autojit
def matrix_mul_vector(arr1,arr2):
	result = []
	M, N = arr1.shape
	for i in range(M):
		sum_sub = 0.0
		for j in range(N):
			sum_sub += arr1[i,j]*arr2[j]
		result.append(sum_sub)
	return result

'''
@jit
def matrix_mul_vector(arr1,arr2):
	mul_sub = multiply(arr1,arr2)
	result = map(sum,mul_sub)
	return result
'''

#计算准确率、系数、残差
def src_calculate(train_in,trlabel_in,test_in,telabel_in,class_num):
	#src_handle = My_SolveHomotopy.initialize() #调用m文件，获取句柄
	print 't333'
	x = []
	r = []
	right_num = 0
	accuracy = 0

	lambda_coef = 0.0001
	tolerance = 1e-5
	stoppingCriterion = 3
	maxiter = 5000



	for i in test_in: #循环判断每张测试样本的类别
	#for pp in range(1):
		#i = test_in[0]
		print 'test_sample:',test_in.index(i)
		start = datetime.now()
		
		xi_tmp = solvehomotopy.SolveHomotopy(np.array(train_in),np.array(i),lambda_coef,tolerance,stoppingCriterion,maxiter) #计算稀疏系数
		
		xi = list(xi_tmp) 

		x.append(xi)
		r_tmp = []
		
		for j in range(class_num): #依次计算与各个类别之间的残差
		
			#从trlabel_in中找出第(j+1)类的所有下标
			allindex = [idx for idx,a in enumerate(trlabel_in) if a==(j+1)]
			
			#保留xi中第(j+1)类的系数，其他类系数置零
			xi_sub = [x_i if i_sub in allindex else 0 for i_sub,x_i in enumerate(xi)]
			#由第(j+1)类得到重构图像restructed_img
			restructed_img = dot(train_in,xi_sub)
			#计算测试样本i和重构图像restructed_img之间的残差
			ri = norm(np.array(i)-restructed_img)
			
			r_tmp.append(ri)

		stop = datetime.now()
		print(stop-start)

		

		#把残差最小的类别作为当前测试样本的类别
		get_class = r_tmp.index(min(r_tmp))+1 
		if get_class == telabel_in[test_in.index(i)]:
			right_num = right_num+1
		r.append(r_tmp)
	accuracy = right_num/len(test_in) #计算识别准确率
	return accuracy,x,r
	



