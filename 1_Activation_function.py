#Linear
import numpy as np
import math
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 500)
print("First 20 values of x are : ", x[0:20])
def linear(x):
    return (x)

plt.title("Linear Function Graph")
plt.xlabel('x')
plt.ylabel('y')

y = list(map(lambda x: linear(x), x))
print("First 20 values of x after apply linear function are :", y[0:20])
plt.plot(x, y)
plt.show()


#Binary Step
import numpy as np  
import matplotlib.pyplot as plt  
def binaryStep(x):  
    return np.heaviside(x,1)  
x = np.linspace(-10, 10)  
print("Value of x", x)  
plt.title('Activation Function : binary Step function')  
plt.xlabel('x')
plt.ylabel('y')

y = binaryStep(x)  
print("Values output of function Binary Step function on x are :", y)

plt.plot(x, y)  
plt.show()  


#Ramp
def ramp(x):  
   return np.maximum(0, np.minimum(1, x))  
  
x = np.linspace(-2, 2, 500)  
print("Value of x range:", f"[{x[0]:.2f}, {x[-1]:.2f}]")  
plt.title('Activation Function : Ramp function')  
plt.xlabel('x')  
plt.ylabel('y')     
y = ramp(x)  
print("First 20 values output of Ramp function on x are:", y[0:20])  
  
plt.plot(x, y, 'g-', linewidth=2)  
plt.grid(True, alpha=0.3)  
plt.show()  



#Gaussian
def gaussian(x):  
   return np.exp(-x**2)
x = np.linspace(-3, 3, 500)  
print("Value of x range:", f"[{x[0]:.2f}, {x[-1]:.2f}]")  
plt.title('Activation Function : Gaussian function')  
plt.xlabel('x')  
plt.ylabel('y')     
y = gaussian(x)  
print("First 20 values output of Gaussian function on x are:", y[0:20]) 
  
plt.plot(x, y, 'r-', linewidth=2)  
plt.grid(True, alpha=0.3)  
plt.show()


#Sigmoid
def sigmoid(x):  
   return 1 / (1 + np.exp(-x))  
  
x = np.linspace(-10, 10, 500)  
print("Value of x range:", f"[{x[0]:.2f}, {x[-1]:.2f}]")  
plt.title('Activation Function : Sigmoid function')  
plt.xlabel('x')  
plt.ylabel('y')

y = sigmoid(x)  
print("First 20 values output of Sigmoid function on x are:", y[0:20])  
plt.plot(x, y, 'm-', linewidth=2)  
plt.grid(True, alpha=0.3)  
plt.show()


#ReLU
def relu(x):  
    return np.maximum(0, x)  
x = np.linspace(-5, 5, 500)  
print("Value of x range:", f"[{x[0]:.2f}, {x[-1]:.2f}]")  
plt.title('Activation Function : ReLU (Rectified Linear Unit)')  
plt.xlabel('x')  
plt.ylabel('y')     
y = relu(x)
print("First 20 values output of ReLU function on x are:", y[0:20])  
  
plt.plot(x, y, 'c-', linewidth=2)  
plt.grid(True, alpha=0.3)  
plt.show()


#Leaky ReLU
def leaky_relu(x, alpha=0.1):  
   return np.where(x > 0, x, alpha * x)  
  
x = np.linspace(-5, 5, 500)  
print("Value of x range:", f"[{x[0]:.2f}, {x[-1]:.2f}]")  
plt.title('Activation Function : Leaky ReLU (alpha=0.1)') 
plt.xlabel('x')  
plt.ylabel('y')  
    
y = leaky_relu(x)  
print("First 20 values output of Leaky ReLU function on x are:",  
y[0:20])  
  
plt.plot(x, y, 'orange', linewidth=2)  
plt.grid(True, alpha=0.3)
plt.show()


#Tanh
def tanh(x):  
    return np.tanh(x)  
x = np.linspace(-5, 5, 500)  
print("Value of x range:", f"[{x[0]:.2f}, {x[-1]:.2f}]")  
plt.title('Activation Function : Hyperbolic Tangent (tanh)')  
plt.xlabel('x')  
plt.ylabel('y')  
y = tanh(x)  
print("First 20 values output of Tanh function on x are:", y[0:20])  
plt.plot(x, y, 'purple', linewidth=2)  
plt.grid(True, alpha=0.3)  
plt.show() 

