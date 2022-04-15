from cmath import cos
from unittest import registerResult
import numpy as np
import matplotlib.pyplot as plt


class LinearRegression():
	def __init__(self, x_train, y_train, x_test, y_test, learning_rate, iteration):
		
		self.x_train = x_train
		self.y_train = y_train
		self.x_test = x_test
		self.y_test = y_test
		self.learning_rate = learning_rate
		self.iteration = iteration
		self.w = None
		self.b = None 
		self.y_head = None


	#ağırlık değerlerimizi initialize ediyoruz
	def initialize_weight_and_bias(self, diamention):

		self.w = np.full((diamention, 1), 0.01)
		self.b = 0.0
		return self.w, self.b

	#calculation sigmoid
	# z = np.dot(w.T, x_train)+b
	def sigmoid(self, z):
		y_head = 1/(1+np.exp(-z))
		return y_head

	# Forward propagation steps:
	# find z = w.T*x+b
	# y_head = sigmoid(z)
	# loss(error) = loss(y,y_head)
	# cost = sum(loss)
	def forward_propagation(self,w,b):
		z = np.dot(w.T, self.x_train) + b
		y_head = self.sigmoid(z)
		self.y_head = y_head
		loss = -self.y_train * np.log(y_head) - (1-self.y_train) * np.log10(1-y_head)
		cost = (np.sum(loss))/self.x_train.shape[1]
		return cost

	# backward propagation takes the derrivative for update weight and bias
	def backward_propagation(self):
		derivative_weight = (np.dot(self.x_train,((self.y_head - self.y_train).T)))/self.x_train.shape[1]
		derivative_bias   = np.sum(self.y_head-self.y_train)/self.x_train.shape[1] 
		gradients    = {"derivative_weight": derivative_weight,
						"derivative_bias"  : derivative_bias}
		return gradients

	
	# this func update weiight and bias as take into acount cost
	def updating_weight_and_bias(self,w,b):
		cost_list  = []
		cost_list2 = []
		index = []

		for i in range(self.iteration):
			cost = self.forward_propagation(w,b)
			gradients = self.backward_propagation()
			cost_list.append(cost)

			w  -= self.learning_rate * gradients["derivative_weight"]
			b  -= self.learning_rate * gradients["derivative_bias"]

			if i % 10 == 0:
				cost_list2.append(cost)
				index.append(i)
				print("%i. Denemedeki güncel maliyet: %f " % (i,cost))

		parameters = {"weight": self.w, "bias":self.b}
		plt.plot(index, cost_list2)
		plt.xticks(index, rotation = "vertical")
		plt.xlabel("iteration değerleri")
		plt.ylabel("Maliyet")
		plt.show()

		return parameters, gradients, cost_list

	def predict(self,w,b):
		z = self.sigmoid(np.dot(w.T, self.x_test)+b)
		y_prediction = np.zeros((1, self.x_test.shape[1]))

		for i in range(z.shape[1]):
			if z[0,i] <= 0.5:
				y_prediction[0,i] = 0
			else:
				y_prediction[0,i] = 1

		return y_prediction


	def linear_regression(self):
		diamention = self.x_train.shape[0]
		w,b = self.initialize_weight_and_bias(diamention)
		parameters, gradients, cost_list = self.updating_weight_and_bias(w,b)
		y_prediction_test = self.predict(parameters["weight"], parameters["bias"])
		y_prediction_train = self.predict(parameters["weight"], parameters["bias"])

		print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train- self.y_train)) * 100))
		print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test- self.y_test)) * 100))
	











