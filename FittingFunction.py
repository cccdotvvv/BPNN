import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
	y = 1/(1+np.exp(1)**(-x))
	return y

x_train = np.arange(-3,3.01,0.01)
m = x_train.size
y_train = np.sin(x_train)

hide_layer_amount = 10
W1 = np.random.rand(1,hide_layer_amount)
B1 = np.random.rand(1,hide_layer_amount)
W2 = np.random.rand(hide_layer_amount,1)
B2 = np.random.rand(1,1)

alpha = 0.1
iterations = 5000

error = np.zeros((1,iterations))
output = np.zeros((1,m))

for ii in range(iterations):
    error[0,ii] = 0
    for nn in range(m):
        input_layer_input = x_train[nn]
        hide_layer_input = np.dot(input_layer_input,W1)-B1
        hide_layer_output = sigmoid(hide_layer_input)
        output[0,nn] = np.dot(hide_layer_output,W2)-B2
        e = output[0,nn]-y_train[nn]
        
        dB2 = -1*alpha*e
        dW2 = e*alpha*hide_layer_output
        dB1 = W2.T*hide_layer_output*(1-hide_layer_output)*(-1)*e*alpha
        dW1 = W2.T*hide_layer_output*(1-hide_layer_output)*input_layer_input*e*alpha
        
        W1 = W1-dW1
        B1 = B1-dB1
        W2 = W2-dW2.T
        B2 = B2-dB2
        
        error[0,ii] = error[0,ii]+np.fabs(e)
    if np.mod(ii,100)==0:
        print('Now finished '+str(ii/iterations*100)+'%')
    elif ii==iterations-1:
        print("Finished!")

x_test = np.arange(-3,3.01,0.01)  
m1 = x_test.size  
y_test = np.sin(x_test)   
for aa in range(m1):
    y_test[aa] = np.dot(sigmoid(np.dot(x_test[aa],W1)-B1),W2)-B2
plt.plot(x_train,y_train,'b',x_test,y_test,'r')
plt.show()
