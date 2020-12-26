%% ��ʼ��
clear
close all
clc

%% ��ȡ����
XlsData=xlsread('������¯���ݿ����.xlsx',7);
OrignData=XlsData';
InputData=OrignData(1:7,:);
OutputData=OrignData(8,:);
TotalNum=size(OutputData,2);

%   ��С����
InputData=single(InputData);
OutputData=single(OutputData);





%% ѵ���������Լ�
    %% %80Ϊѵ������ %20Ϊ ���Լ� 
TrainSize=0.45;
TrainNum=round(TotalNum*TrainSize);
ChooseForTrain=randperm(TotalNum,TrainNum);
Train_Input=InputData(:,ChooseForTrain);
Train_Output=OutputData(:,ChooseForTrain);
Test_Input=InputData;
Test_Input(:,ChooseForTrain)=[];
Test_Output=OutputData;
Test_Output(:,ChooseForTrain)=[];

%% ���ݹ�һ��
[Nom_Train_Input,NomSyb_Train_Input]=mapminmax(Train_Input,0,1);   
[Nom_Train_Output,NomSyb_Train_Output]=mapminmax(Train_Output,0,1);
Nom_Test_Input=mapminmax('apply',Test_Input,NomSyb_Train_Input) ;

%% ���� RBF ������
RBF_Net=newrbe(Nom_Train_Input,Nom_Train_Output);

    
%% RBF ����Ҫ����

%% RBF���粻��Ҫѵ��


%% ���� RBF_Net������
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



