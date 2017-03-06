
1、src_test.py：（测试成功）
	实现论文"Robust face recognition via sparse representation"中
        的src算法，可直接运行

2、src_function.py：
	get_normalize(data_input):对矩阵每一列进行二范数归一化，并返回结果
	matrix_mul_vector():矩阵和向量相乘
	src_calculate():计算准确率、系数、残差


3、sdr_test.py：（测试成功）
	实现论文"Sparse and Dense Hybrid Representation via Dictionary 
 	Decomposition for Face Recognition"中的sdr算法，可直接运行

4、sdr_function.py：
	get_normalize():对矩阵每一列进行二范数归一化，并返回结果
	Obtain_B_IALM():实现论文中的算法2，获得子字典B
	Update_X_IALM():实现论文中的算法3，更新X
	Update_A_IALM():实现论文中的公式(25)，更新A
	SLR_Decomposition():实现论文中的算法4，进行SLR分解
	SDR_IALM():实现论文中的算法1，计算系数
	SDR_Calculate():计算识别率


5、solvehomotopy.py:
	用于求解L1范数问题：

6、SolveHomotopy.m：
	solvehomotopy.py根据这个文件编写的，可在 http://yima.csl.illinois.edu/
	下载


7、训练数据集：
	ResizeARDatabase_540.mat: 540维的AR数据集，SDR论文作者主页下载的
	
	以下的AR训练数据集都是个人从AR原始数据集下采样得到的：
	（均由 Create_Train_Data 这个文件夹下的代码生成）
	AR_Data_540.mat： 每列由大小为 27*20 的样本reshape得到，共700张样本
	AR_Data_850.mat： 每列由大小为 34*25 的样本reshape得到，共700张样本
	AR_Data_1200.mat： 每列由大小为 40*30 的样本reshape得到，共700张样本
	AR_Data_2200.mat： 每列由大小为 50*44 的样本reshape得到，共700张样本

8、把每个训练集的运行结果保存起来，命名相互对应，如：AR_Data_540.mat的运行
   结果保存在SRC_Result_540.mat（用src_test.py生成）以及SDR_Result_540.mat
   （用sdr_test.py生成）,并集中保存在 Result_Data 这个文件夹下



