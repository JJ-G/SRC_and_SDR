
1��src_test.py�������Գɹ���
	ʵ������"Robust face recognition via sparse representation"��
        ��src�㷨����ֱ������

2��src_function.py��
	get_normalize(data_input):�Ծ���ÿһ�н��ж�������һ���������ؽ��
	matrix_mul_vector():������������
	src_calculate():����׼ȷ�ʡ�ϵ�����в�


3��sdr_test.py�������Գɹ���
	ʵ������"Sparse and Dense Hybrid Representation via Dictionary 
 	Decomposition for Face Recognition"�е�sdr�㷨����ֱ������

4��sdr_function.py��
	get_normalize():�Ծ���ÿһ�н��ж�������һ���������ؽ��
	Obtain_B_IALM():ʵ�������е��㷨2��������ֵ�B
	Update_X_IALM():ʵ�������е��㷨3������X
	Update_A_IALM():ʵ�������еĹ�ʽ(25)������A
	SLR_Decomposition():ʵ�������е��㷨4������SLR�ֽ�
	SDR_IALM():ʵ�������е��㷨1������ϵ��
	SDR_Calculate():����ʶ����


5��solvehomotopy.py:
	�������L1�������⣺

6��SolveHomotopy.m��
	solvehomotopy.py��������ļ���д�ģ����� http://yima.csl.illinois.edu/
	����


7��ѵ�����ݼ���
	ResizeARDatabase_540.mat: 540ά��AR���ݼ���SDR����������ҳ���ص�
	
	���µ�ARѵ�����ݼ����Ǹ��˴�ARԭʼ���ݼ��²����õ��ģ�
	������ Create_Train_Data ����ļ����µĴ������ɣ�
	AR_Data_540.mat�� ÿ���ɴ�СΪ 27*20 ������reshape�õ�����700������
	AR_Data_850.mat�� ÿ���ɴ�СΪ 34*25 ������reshape�õ�����700������
	AR_Data_1200.mat�� ÿ���ɴ�СΪ 40*30 ������reshape�õ�����700������
	AR_Data_2200.mat�� ÿ���ɴ�СΪ 50*44 ������reshape�õ�����700������

8����ÿ��ѵ���������н�����������������໥��Ӧ���磺AR_Data_540.mat������
   ���������SRC_Result_540.mat����src_test.py���ɣ��Լ�SDR_Result_540.mat
   ����sdr_test.py���ɣ�,�����б����� Result_Data ����ļ�����



