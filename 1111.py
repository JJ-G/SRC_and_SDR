#-*-coding:UTF-8 -*-

from numpy import *
import scipy.io as sio
import numpy as np
import sdr_function
import matplotlib.pyplot as plt
#import gc



#读取数据
load_AR_data = 'AR_Data_540.mat'
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


#样本初始化
D = Train_Norm_arr
Test = Test_Norm_arr

sio.savemat('Data_540.mat', {'D': D,'Test_Samples': Test,
	'Train_Label': Train_Label,'Test_Label': Test_Label})