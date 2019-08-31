clc;clear all

x = [-50:0.01:0]
y = sin([x])
[input,ps1] = mapminmax(x)
[target,ps2] = mapminmax(y)

net = newff(input,target,100,{'tansig','purelin'},'trainlm')
net.trainParam.epochs = 1000000
net.trainParam.goal = 0.00000000001
LP.lr = 0.0000001
net = train(net,input,target)

x1 = [-50:0.01:0]
input1 = mapminmax('apply',[x1],ps1)
output1 = net(input1)
y1 = mapminmax('reverse',output1,ps2)

figure(1)
subplot(1,2,1)
plot(x,y)
subplot(1,2,2)
plot(x1,y1)
figure(2)
plot(x,y)
hold on
plot(x1,y1,'r')
