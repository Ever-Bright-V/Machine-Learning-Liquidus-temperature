%% 初始化
clear
close all
clc

%% 读取数据
XlsData=xlsread('溅渣护炉数据库计算.xlsx',6); % 7 是保留后的值
OrignData=XlsData';
InputData=OrignData(1:7,:);
OutputData=OrignData(8,:);
TotalNum=size(OutputData,2);





%% 训练集、测试集
    %% %80为训练集， %20为 测试集 
TrainSize=0.9;
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

%% 构建BP神经网络
Bp_Net=newff(Nom_Train_Input,Nom_Train_Output,[16,16,16,16])
    
 % bp 神经网络参数
Bp_Net.trainParam.epochs=200000;
Bp_Net.trainParam.lr=0.008;
Bp_Net.trainParam.goal=0.000000001;
Bp_Net.trainParam.max_fail=20;

%% 开始 训练 Bp神经网络
Bp_Net=train(Bp_Net,Nom_Train_Input,Nom_Train_Output)

%% 测试 Bp神经网络
Nom_Bp_Test_Output=sim(Bp_Net,Nom_Test_Input)
Bp_Test_Output=mapminmax('reverse',Nom_Bp_Test_Output,NomSyb_Train_Output)
Bp_Test_Err=Bp_Test_Output-Test_Output




figure
Mse=mse(Bp_Test_Err)

plot(Bp_Test_Err,'bo-')
text(1000,5,"MSE:"+num2str(Mse))
xlabel("TestData")
ylabel("Predict-Real")
title('Error')



