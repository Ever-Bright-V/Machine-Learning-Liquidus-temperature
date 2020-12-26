%% 初始化
clear
close all
clc

%% 读取数据
XlsData=xlsread('溅渣护炉数据库计算.xlsx',7);
OrignData=XlsData';
InputData=OrignData(1:7,:);
OutputData=OrignData(8,:);
TotalNum=size(OutputData,2);

%   减小精度
InputData=single(InputData);
OutputData=single(OutputData);





%% 训练集、测试集
    %% %80为训练集， %20为 测试集 
TrainSize=0.45;
TrainNum=round(TotalNum*TrainSize);
ChooseForTrain=randperm(TotalNum,TrainNum);
Train_Input=InputData(:,ChooseForTrain);
Train_Output=OutputData(:,ChooseForTrain);
Test_Input=InputData;
Test_Input(:,ChooseForTrain)=[];
Test_Output=OutputData;
Test_Output(:,ChooseForTrain)=[];

%% 数据归一化
[Nom_Train_Input,NomSyb_Train_Input]=mapminmax(Train_Input,0,1);   
[Nom_Train_Output,NomSyb_Train_Output]=mapminmax(Train_Output,0,1);
Nom_Test_Input=mapminmax('apply',Test_Input,NomSyb_Train_Input) ;

%% 构建 RBF 神经网络
RBF_Net=newrbe(Nom_Train_Input,Nom_Train_Output);

    
%% RBF 不需要参数

%% RBF网络不需要训练


%% 测试 RBF_Net神经网络
Nom_RBF_Test_Output=sim(RBF_Net,Nom_Test_Input);
RBF_Test_Output=mapminmax('reverse',Nom_RBF_Test_Output,NomSyb_Train_Output);
RBF_Test_Err=RBF_Test_Output-Test_Output;




figure
Mse=mse(RBF_Test_Err)

plot(RBF_Test_Err,'bo-')
text(1000,5,"MSE:"+num2str(Mse))
xlabel("TestData")
ylabel("Predict-Real")
title('Error')



