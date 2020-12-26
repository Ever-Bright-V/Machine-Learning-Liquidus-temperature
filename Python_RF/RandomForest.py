import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler
import sklearn.ensemble as se  # 集合算法模块
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error  # mse

#ValueError: Expected 2D array, got 1D array instead:
#array=[1730.4288 1579.9968 1506.1499 ... 1481.3967 1431.4591 1491.4399].
#Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.

# 稍微 缩小 array中国数据 精度

# 导入数据
import os
from sys import argv 
ParentDir=os.path.dirname(argv[0])

Data=pd.read_excel(ParentDir + '\\溅渣护炉数据库计算.xlsx',sheet_name='AllDataX')
print(Data)

Input=np.array(Data[Data.columns[:-1]],dtype=np.float64)  
Output=np.array(Data[Data.columns[-1]],dtype=np.float64)
print(Input)
print(Output)

print(Input.dtype)
print(Output.dtype)





## 归一化 和 划分 的 顺序问题
# 先 划分
Train_Input,Test_Input,Train_Output,Test_Output=train_test_split(Input,Output,test_size=0.1)  # 注意参数顺序

#print(Train_Input.isnull().any())
print(Train_Input)
print(np.isnan(Train_Input).any())
print(np.isfinite(Train_Input).all())


# 归一化(只对输出 归一)   
Used_Scale=MinMaxScaler()  # 用全集 来 初始化 Xmax 和 Xmin参数
Used_Scale.fit(Input)

Std_Train_Input=Used_Scale.transform(Train_Input)
Std_Test_Input=Used_Scale.transform(Test_Input)



# 模型
#RF = se.RandomForestRegressor(max_depth=10, n_estimators=1000, min_samples_split=3) #
RF = se.RandomForestRegressor(n_estimators=200, random_state=0) #
#Train_Output=Train_Output.reshape(-1,1)
RF.fit(Std_Train_Input,Train_Output)


# 模型测试
#Test_Output=Test_Output.reshape(-1,1)
RF_Test_Out=RF.predict(Std_Test_Input)
print('R2:{}'.format(RF.score(Std_Test_Input,Test_Output)))
print('MSE:{}'.format(mean_squared_error(Test_Output,RF_Test_Out)))


# 测试集 画图
    # 绝对误差作图

x=np.arange(1,len(Test_Output)+1)
plt.plot(x,RF_Test_Out-Test_Output)
plt.title='Error:Prdic-Real'
plt.text(0,-1,'R2:{}'.format(RF.score(Std_Test_Input,Test_Output)))
plt.text(0,0,'MSE:{}'.format(mean_squared_error(Test_Output,RF_Test_Out)))
plt.show()







