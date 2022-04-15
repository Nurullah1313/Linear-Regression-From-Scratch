import numpy as np
import matplotlib.pyplot as plt
from linear_regression_class import LinearRegression
from sklearn.model_selection import train_test_split


x_l = np.load('data_sets/X.npy')
y_l = np.load('data_sets/Y.npy')


#sadece 0 ile 1 görüntülerini eğiteceğimiz için bu görselleri dizi içerisinden çekiyoruz
# ve birleştirip tek bir değişkene atıyoruz
X = np.concatenate((x_l[204:409], x_l[822:1027]), axis = 0)
z = np.zeros(205)
o = np.ones(205)
Y= np.concatenate((z,o),  axis = 0).reshape(X.shape[0],1)


print("X shape: " , X.shape)
print("Y shape: " , Y.shape)

#sklearn kütüphanesinin veri parçalama fonksiyonundan faydalanarak verilerimizi train ve test
# olarak ikiye ayırıyoruz
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
num_of_train = X_train.shape[0]
num_of_test  = X_test.shape[0]


#görüntü matrisimizi eğitim için vektör haline getiriyoruz 
X_train_flatten = X_train.reshape(num_of_train, X_train.shape[1]*X_train.shape[2])
X_test_flatten  = X_test.reshape(num_of_test, X_test.shape[1]*X_test.shape[2])

print("X_train_flatten: ", X_train_flatten.shape)
print("X_test_flatten: ", X_test_flatten.shape)

x_train = X_train_flatten.T
x_test  = X_test_flatten.T
y_train = Y_train.T
y_test  = Y_test.T

print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",y_train.shape) 
print("y test: ",y_test.shape)

regression = LinearRegression(x_train = x_train,
                              y_train = y_train,
                              x_test = x_test,
                              y_test = y_test,
                              learning_rate=0.005,
                              iteration=150)
regression.linear_regression()














