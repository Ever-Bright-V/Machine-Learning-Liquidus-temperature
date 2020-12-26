%%���Ȩ�ء�ǰ����������Ȩ�ظ��£�����һ����ﵽ����ޣ�ֹͣ

%% ��ʼ��
clear
close all
clc

%% ��ȡ����
input=rand(4,200);
output=input(1,:).*input(2,:);

%% ѵ���������Լ�
input_train = input(:,1:150);
output_train =output(1:150);
input_test =input(:,151:end);
output_test =output(151:end);

%% ���ݹ�һ��
[inputn,inputps]=mapminmax(input_train,0,1);
[outputn,outputps]=mapminmax(output_train); %% ֻ�� train�����ݼ������� ��һ��
inputn_test=mapminmax('apply',input_test,inputps); %% test ���Լ�Ҳʹ�� ��ͬ�� �� ��һ�� ����
  %% �˴� ���� test��output ��һ������Ϊ ���� ���ʱ��ʹ�� ����һ�� ������ ���� ����

%% ����BP������
net=newff(inputn,outputn,[8 10]);

% �������
net.trainParam.epochs=1000;         % ѵ������
net.trainParam.lr=0.01;                   % ѧϰ����
net.trainParam.goal=0.000001;        % ѵ��Ŀ����С���
% net.dividefcn='';
%% BP������ѵ��
net=train(net,inputn,outputn);

%% BP���������
an=sim(net,inputn_test); %��ѵ���õ�ģ�ͽ��з��� 
test_simu=mapminmax('reverse',an,outputps); % Ԥ��������һ��

error=test_simu-output_test;      %Ԥ��ֵ����ʵֵ�����

%%��ʵֵ��Ԥ��ֵ���Ƚ�
figure(1)
plot(output_test,'bo-')
hold on
plot(test_simu,'r*-')
hold on
plot(error,'square','MarkerFaceColor','b')
legend('����ֵ','Ԥ��ֵ','���')
xlabel('��������'),ylabel('ֵ'),title('���Լ�Ԥ��ֵ������ֵ�����Ա�'),set(gca,'fontsize',12)
%�������
[~,len]=size(output_test);
MAE1=sum(abs(error./output_test))/len;
MSE1=error*error'/len;
RMSE1=MSE1^(1/2);
disp(['-----------------------������--------------------------'])
disp(['ƽ���������MAEΪ��',num2str(MAE1)])
disp(['�������MSEΪ��       ',num2str(MSE1)])
disp(['���������RMSEΪ��  ',num2str(RMSE1)])
%ע
figure
text(0.25,0.65,char([26356,22810,27169,22411]),'fontsize',40)
text(0.25,0.43,char([81,81,65306,50,56,54,49,54,54,54,57,50,55]),'fontsize',25)