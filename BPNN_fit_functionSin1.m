clear all;clc;

x_train = [-3:0.01:3];
x_train_size = size(x_train);
m = x_train_size(2);
y_train = sin([x_train]);

hide_layer_amount = 10;
W1 = rand(1,hide_layer_amount);
B1 = rand(1,hide_layer_amount);
W2 = rand(hide_layer_amount,1);
B2 = rand(1,1);

alpha = 0.1;
iterations = 5000;

error = zeros(1,iterations);
output = zeros(1,m);

for ii = 1:iterations
    error(ii) = 0;
    for nn = 1:m
        input = x_train(nn);
        hide_layer_input = input*W1-B1;
        hide_layer_output = sigmoid(hide_layer_input);
        output(nn) = hide_layer_output*W2-B2;
        e = output(nn)-y_train(nn);
        
        dB2 = -1*alpha*e;
        dW2 = e*alpha*hide_layer_output;
        dB1 = W2'.*hide_layer_output.*(1-hide_layer_output)*(-1)*e*alpha;
        dW1 = W2'.*hide_layer_output.*(1-hide_layer_output)*input*e*alpha; 
        
        W1 = W1-dW1;
        B1 = B1-dB1;
        W2 = W2-dW2';
        B2 = B2-dB2;
        
        error(ii) = error(ii)+abs(e);
    
    end
    if mod(ii,100)==0
        sprintf('已迭代%d次，共需迭代%d次',ii,iterations)
    end
end

x_test = -3:0.01:3;
x_test_size = size(x_test);
m1 = x_test_size(2);
y_test = zeros(1,m1);
for aa = 1:m1
    y_test(aa) = sigmoid(x_test(aa)*W1-B1)*W2-B2;
end
figure(1)
plot(x_train,y_train)
hold on
plot(x_test,y_test)
figure(2)
plot(error)
    
    
    
    
    