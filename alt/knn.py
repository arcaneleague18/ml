import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split

np.random.seed(42)
#making dataset
class1 = np.random.randn(50,2) + np.array([2,2])
class2 = np.random.randn(50,2) + np.array([6,6])

X = np.vstack((class1,class2))
y = np.array([0]*50 + [1]*50)

#train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print("\nConfusion matrix\n")
print(confusion_matrix(y_test,y_pred))
print("\n accuracy score\n")
print(accuracy_score(y_test,y_pred))