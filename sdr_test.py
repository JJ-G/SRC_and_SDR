#-*-coding:UTF-8 -*-

'''
-Created on Jan 25,2017
-@author:Jiajing Guo

-it is the implementation of algorithm 1 in paper
 "Sparse and Dense Hybrid Representation via Dictionary 
 Decomposition for Face Recognition"

-实现步骤：
 1、读取数据，并进行二范数归一化处理
 2、进行SLR分解，由字典D分解获取字典A和字典B
 3、计算系数和识别率
 4、对返回的结果进行处理
'''

from numpy import *
import scipy.io as sio
import numpy as np
import sdr_function
import matplotlib.pyplot as plt
#import gc



#读取数据
load_AR_data = 'AR_Data_1200.mat'
load_data = sio.loadmat(load_AR_data)
#打印 load_data 中所有的键
print load_data.keys()


#根据键读取需要的值
TrainSamples_tmp = load_data.get('TrainFace') #训练样本,每一列表示一个样本
TestSamples_tmp = load_data.get('TestFace') #测试样本，每一列表示一个样本
trainlabel_tmp = load_data.get('TrainLabel') 
testlabel_tmp = load_data.get('TestLabel') 


#训练样本和测试样本的类别标签
Train_Label = trainlabel_tmp[0] #训练样本类别,一维array
Test_Label = testlabel_tmp[0] #测试样本类别,一维array

#读取类的数目（测试数据和训练数据的类数目是一样的，读取其中一个即可）
ClassesNum = max(Train_Label)
#读取测试样本总数
TestSamp_Num = len(Test_Label)

#对每列数据分别进行二范数归一化，类型为nparray
Train_Norm_arr = sdr_function.get_normalize(TrainSamples_tmp)
Test_Norm_arr = sdr_function.get_normalize(TestSamples_tmp)

#参数初始化
v = pow(sqrt(len(Train_Norm_arr)),-1)
Lambda = 1
Delta = 1.2*v
k = 4
Tau = 0.01
Eta = 1*v
Beta = 10
Gamma = 10
Iter_max = 1000

#样本初始化
D = Train_Norm_arr
Test = Test_Norm_arr

#调用slr_function中的slr_decomposition方法得到字典A和字典B
A,B = sdr_function.SLR_Decomposition(D,k,Train_Label,
	int(ClassesNum),Lambda,Delta,Tau,Eta,Iter_max)
print A[0:4,0:4]


#计算识别率和系数
accuracy,alpha_all,x_all,e_all,resd_r = sdr_function.SDR_Calculate(D,
	Train_Label,Test,Test_Label,int(ClassesNum),A,B,Beta,Gamma,Iter_max)




#保存数据
sio.savemat('SDR_Result_1200.mat', {'accuracy': accuracy,'alpha_all': alpha_all,
	'x_all': x_all,'e_all': e_all,'resd_r': resd_r})



#读取保存的数据
srcResult = sio.loadmat('SDR_Result_1200.mat')
accuracy = srcResult['accuracy']
alpha_all = srcResult['alpha_all']
x_all = srcResult['x_all']
e_all = srcResult['e_all']
resd_r = srcResult['resd_r']

#读取原始数据
load_data = sio.loadmat('AR_Data_1200.mat')
TrainSamples = load_data['TrainFace']
TestSamples = load_data['TestFace']
rows = int(load_data['rows'])
columns = int(load_data['columns'])

print accuracy*100
i = 10
plt.figure(1)
plt.plot(alpha_all[i])
plt.figure(2)
plt.plot(x_all[i])
plt.figure(3)
plt.bar(np.arange(len(resd_r[i])),resd_r[i])
#显示测试图像
plt.figure(4)
plt.imshow(TestSamples[:,i].reshape((columns,rows)).T)
#显示最大系数对应的训练图像
j = [idx for idx,a in enumerate(alpha_all[i]) if a==max(alpha_all[i])]
plt.figure(5)
plt.imshow(TrainSamples[:,j].reshape((columns,rows)).T)
plt.show()

