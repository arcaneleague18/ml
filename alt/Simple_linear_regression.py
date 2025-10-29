from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import numpy as np
x= np.array([1,2,3,4,5,7,8,9])
y = np.array([2,4,6,8,10,14,9,9])

xm = np.mean(x)
ym = np.mean(y)

m = np.sum((x-xm)*(y-ym))/np.sum((x-xm)**2)

b = ym - xm*m

x_test = np.array([1,4,6])
y_test = np.array([2,8,12])

def predict(x):
    return m*x + b

acc = []
acc = y_test - predict(x_test)
print(np.mean(acc**2))
# acc =np.array(acc)
# print(np.sum(acc**2)/len(acc))
