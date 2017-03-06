#-*-coding:UTF-8 -*-

'''
-Created on Jan 19,2017
-@author:Jiajing Guo

-it is the implementation of algorithm 1 in paper
 "Robust face recognition via sparse representation"

-实现步骤：
 1、读取数据，并进行二范数归一化处理
 2、调用src_function中的src_calculate()方法计算识别率
 3、对返回的结果进行处理
'''


from numpy import *
import scipy.io as sio
import numpy as np
import src_function
import matplotlib.pyplot as plt



#读取数据
load_AR_data = 'AR_Data_2200.mat'
load_data = sio.loadmat(load_AR_data)
#打印 load_data 中所有的键
print load_data.keys()

#根据键读取需要的值
TrainSamples = load_data.get('TrainFace') #训练样本,每一列表示一个样本
TestSamples = load_data.get('TestFace') #测试样本，每一列表示一个样本
trainlabel_tmp = load_data.get('TrainLabel') 
TrainSamp_Label = trainlabel_tmp[0] #训练样本类别,一维array
testlabel_tmp = load_data.get('TestLabel') 
TestSamp_Label = testlabel_tmp[0] #测试样本类别,一维array

#读取类的数目（测试数据和训练数据的类数目是一样的，读取其中一个即可）
ClassesNum = max(TrainSamp_Label)
#读取测试样本总数
TestSamp_Num = len(TestSamp_Label)

#对每列数据分别进行二范数归一化
Train_Norm = src_function.get_normalize(TrainSamples)
Test_Norm = src_function.get_normalize(TestSamples)


#把nparray转换成list,tolist是为了方便传递给m文件
Train_Data = Train_Norm.tolist()
Test_Data = transpose(Test_Norm).tolist() #转置是为了方便后面子函数的循环操作
Train_Label = TrainSamp_Label.tolist()
Test_Label = TestSamp_Label.tolist()

[accuracy,coef_xp,resd_r] = src_function.src_calculate(Train_Data,Train_Label,Test_Data,Test_Label,int(ClassesNum))



#保存数据
sio.savemat('SRC_Result_2200.mat', {'accuracy': accuracy,'coef_xp': coef_xp,'resd_r': resd_r})



#读取保存的数据
srcResult = sio.loadmat('SRC_Result_2200.mat')
accuracy = srcResult['accuracy']
coef_xp = srcResult['coef_xp']
resd_r = srcResult['resd_r']

#读取原始数据
load_data = sio.loadmat('AR_Data_2200.mat')
TrainSamples = load_data['TrainFace']
TestSamples = load_data['TestFace']
rows = int(load_data['rows'])
columns = int(load_data['columns'])

print accuracy*100

i = 0
#显示测试图像稀疏系数
plt.figure(1)
plt.plot(coef_xp[i],'b')
#显示与各类之间的残差
plt.figure(2)
plt.bar(np.arange(len(resd_r[i])),resd_r[i],facecolor='b')
#显示测试图像
plt.figure(3)
plt.imshow(TestSamples[:,i].reshape((columns,rows)).T)
#显示最大系数对应的训练图像
j = [idx for idx,a in enumerate(coef_xp[i]) if a==max(coef_xp[i])]
plt.figure(4)
plt.imshow(TrainSamples[:,j].reshape((columns,rows)).T)
plt.show()



