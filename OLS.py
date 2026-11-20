import numpy as np
import matplotlib.pyplot as plt


data1 = np.load("../Fichiers_TP/data1.npy")

print(data1.shape)
x = data1[0]
y = data1[1]


plt.figure(1)
plt.plot(x, y, 'bx')

X = np.column_stack([np.ones(len(x)), x])  

beta_est = (np.linalg.inv(X.T @ X) @ X.T) @ y

x_line = np.linspace(min(x), max(x), 200)
y_line = beta_est[0] + beta_est[1] * x_line
plt.plot(x_line, y_line, 'r-')  


y_est = X @ beta_est

error = (1 / len(y)) * np.sum((y - y_est) ** 2)

print("beta =", beta_est)
print("training error =", error)

x_line = np.linspace(x.min(), x.max(), 200)  
plt.plot(x, y, 'bx', label="data") # cloud of points
plt.plot(x_line, beta_est[0] + beta_est[1] * x_line, 'r-', label="OLS") # fitted line
plt.legend()




data2 = np.load("../Fichiers_TP/data2.npy")   
print(data2.shape)                           
x, y = data2[0], data2[1] # features and labels

X = np.column_stack([np.ones(len(x)), x, x**2])  
beta_est = np.linalg.solve(X.T @ X, X.T @ y)    

y_est = X @ beta_est                          
error = np.mean((y - y_est) ** 2)             
print("beta =", beta_est)
print("training error =", error)              

x_line = np.linspace(x.min(), x.max(), 200)    
X_line = np.column_stack([np.ones(len(x_line)), x_line, x_line**2])  
y_line = X_line @ beta_est               

plt.figure(2)                                  
plt.plot(x, y, 'bx', label="data")             
plt.plot(x_line, y_line, 'r-', label="OLS")    
plt.legend()                                  
plt.show()        
