%%随机权重→前向计算→误差→权重更新，再来一遍→达到误差限，停止

%% 初始化
clear
close all
clc

%% 读取数据
input=rand(4,200);
output=input(1,:).*input(2,:);

%% 训练集、测试集
input_train = input(:,1:150);
output_train =output(1:150);
input_test =input(:,151:end);
output_test =output(151:end);

%% 数据归一化
[inputn,inputps]=mapminmax(input_train,0,1);
[outputn,outputps]=mapminmax(output_train); %% 只对 train的数据集，进行 归一化
inputn_test=mapminmax('apply',input_test,inputps); %% test 测试集也使用 相同的 的 归一化 参数
  %% 此处 不将 test的output 归一化，因为 测试 误差时，使用 反归一化 ，所以 可以 这样

%% 构建BP神经网络
net=newff(inputn,outputn,[8 10]);

% 网络参数
net.trainParam.epochs=1000;         % 训练次数
net.trainParam.lr=0.01;                   % 学习速率
net.trainParam.goal=0.000001;        % 训练目标最小误差
% net.dividefcn='';
%% BP神经网络训练
net=train(net,inputn,outputn);

%% BP神经网络测试
an=sim(net,inputn_test); %用训练好的模型进行仿真 
test_simu=mapminmax('reverse',an,outputps); % 预测结果反归一化

error=test_simu-output_test;      %预测值和真实值的误差

%%真实值与预测值误差比较
figure(1)
plot(output_test,'bo-')
hold on
plot(test_simu,'r*-')
hold on
plot(error,'square','MarkerFaceColor','b')
legend('期望值','预测值','误差')
xlabel('数据组数'),ylabel('值'),title('测试集预测值和期望值的误差对比'),set(gca,'fontsize',12)
%计算误差
[~,len]=size(output_test);
MAE1=sum(abs(error./output_test))/len;
MSE1=error*error'/len;
RMSE1=MSE1^(1/2);
disp(['-----------------------误差计算--------------------------'])
disp(['平均绝对误差MAE为：',num2str(MAE1)])
disp(['均方误差MSE为：       ',num2str(MSE1)])
disp(['均方根误差RMSE为：  ',num2str(RMSE1)])
%注
figure
text(0.25,0.65,char([26356,22810,27169,22411]),'fontsize',40)
text(0.25,0.43,char([81,81,65306,50,56,54,49,54,54,54,57,50,55]),'fontsize',25)