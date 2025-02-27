import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
import cnn_utils
import cv2
#load_dataset
def load_dataset():
	X_train_orig , Y_train_orig , X_test_orig , Y_test_orig , classes = cnn_utils.load_dataset()
	return X_train_orig , Y_train_orig , X_test_orig , Y_test_orig 

#init_dataset
def init_dataset(X_train_orig , Y_train_orig , X_test_orig , Y_test_orig ):
	X_train = X_train_orig/255.
	X_test = X_test_orig/255.
	Y_train = cnn_utils.convert_to_one_hot(Y_train_orig, 6).T
	Y_test = cnn_utils.convert_to_one_hot(Y_test_orig, 6).T
	print ("number of training examples = " + str(X_train.shape[0]))
	print ("number of test examples = " + str(X_test.shape[0])) 
	print ("X_train shape: " + str(X_train.shape)) 
	print ("Y_train shape: " + str(Y_train.shape))
	print ("X_test shape: " + str(X_test.shape))
	print ("Y_test shape: " + str(Y_test.shape))
	return X_train, Y_train, X_test, Y_test

#create_placeholder
def create_placeholder(n_H0, n_W0, n_C0, n_y):
	X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0], name = "X")
	# X = tf.placeholder(tf.float32, [None, n_H0*n_W0*n_C0], name = "X")
	Y = tf.placeholder(tf.float32, [None,n_y], name = "Y")
	keep_prob = tf.placeholder(tf.float32)
	return X,Y, keep_prob

#init_parameters
def init_parameters():
	tf.set_random_seed(1) #指定随机种子
	W1 = tf.get_variable("W1",[5,5,3,6], initializer=tf.contrib.layers.xavier_initializer(seed=0))
	b1 = tf.get_variable("b1",[6,], initializer=tf.zeros_initializer())
	W2 = tf.get_variable("W2",[5,5,6,16], initializer=tf.contrib.layers.xavier_initializer(seed=0))
	b2 = tf.get_variable("b2",[16,], initializer=tf.zeros_initializer())
	# W3 = tf.get_variable("W3",[5*5*16,120], initializer=tf.contrib.layers.xavier_initializer(seed=0))
	# b3 = tf.get_variable("b3",[120], initializer=tf.zeros_initializer())
	# W4 = tf.get_variable("W4",[120,84], initializer=tf.contrib.layers.xavier_initializer(seed=0))
	# b4 = tf.get_variable("b4",[84], initializer=tf.zeros_initializer())
	# W5 = tf.get_variable("W5",[84,6], initializer=tf.contrib.layers.xavier_initializer(seed=0))
	# b5 = tf.get_variable("b5",[6], initializer=tf.zeros_initializer())
	parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, }#"W3": W3, "b3": b3, "W4": W4, "b4": b4, "W5": W5, "b5": b5} 

	return parameters

#forward_propagation
def forward_propagation(X, parameters,keep_prob):
	with tf.name_scope('forward'):
		W1 = parameters['W1'] 
		b1 = parameters['b1'] 
		W2 = parameters['W2'] 
		b2 = parameters['b2'] 
		# W3 = parameters['W3'] 
		# b3 = parameters['b3']
		# W4 = parameters['W4'] 
		# b4 = parameters['b4']
		# W5 = parameters['W5'] 
		# b5 = parameters['b5']

		# X = tf.reshape(X, [-1, 64, 64, 3])

		Z0 = tf.nn.max_pool(X,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")

		Z1 = tf.nn.bias_add(tf.nn.conv2d(Z0,W1,strides=[1,1,1,1],padding="VALID"),b1)
		A1 = tf.nn.relu(Z1)
		P1 = tf.nn.avg_pool(A1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
		# print("p1"+str(P1.shape))

		Z2 = tf.nn.bias_add(tf.nn.conv2d(P1,W2,strides=[1,1,1,1],padding="VALID"),b2)
		A2 = tf.nn.relu(Z2)
		P2 = tf.nn.avg_pool(A2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
		print("p2"+str(P2.shape))

		# P2 = tf.reshape(P2,[-1,400])
		# # print("p2.reshape"+str(P2.shape))

		# Z3 = tf.matmul(P2,W3)+b3
		# A3 = tf.nn.relu(Z3)
		# # print("Z3"+str(A3.shape))

		# Z4 = tf.matmul(A3,W4)+b4
		# A4 = tf.nn.relu(Z4)
		# # print("a4"+str(A4.shape))

		# if is_train_or_prediction:
		# 	Z4 = tf.nn.dropout(A4, 0.5)
		# # print("Z4"+str(Z4.shape))
		# Z5 = tf.matmul(Z4,W5)+b5
		# print(Z5.shape)

		#f1
		P1 = tf.contrib.layers.flatten(P2)
		Z3 = tf.contrib.layers.fully_connected(P1,120,activation_fn=tf.nn.relu,scope="fu1")#None)#tf.nn.relu) tf.nn.sigmoid

		Z4 = tf.contrib.layers.fully_connected(Z3,84,activation_fn=tf.nn.relu)
		Z4 = tf.nn.dropout(Z4, keep_prob=keep_prob)
		with tf.name_scope("Z5"):
			Z5 = tf.contrib.layers.fully_connected(Z4,6,activation_fn=None)

		return Z5

#compute the loss
def compute_loss(Z5,Y):
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = Z5, labels = Y))
	return loss 

#define the model
def model(X_train,Y_train,X_test,Y_test,learning_rate,num_epochs,minibatch_size,print_cost,isPlot):
	seed = 3
	(m , n_H0, n_W0, n_C0) = X_train.shape
	n_y = Y_train.shape[1]
	costs = []

	X,Y,keep_prob = create_placeholder(n_H0, n_W0, n_C0, n_y)

	parameters = init_parameters()

	Z5 = forward_propagation(X, parameters,keep_prob)

	cost = compute_loss(Z5, Y)

	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

	init = tf.global_variables_initializer()

	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(init)
		for epoch in range(num_epochs):
			minibatch_cost = 0
			num_minibatches = int(m / minibatch_size) #获取数据块的数量
			#seed = seed + 1
			minibatches = cnn_utils.random_mini_batches(X_train,Y_train,minibatch_size,seed) 

			#对每个数据块进行处理
			for minibatch in minibatches:
				#选择一个数据块
				(minibatch_X,minibatch_Y) = minibatch
				# minibatch_X = np.reshape(minibatch_X, [-1,12288])
				#最小化这个数据块的成本
				_ , temp_cost = sess.run([optimizer,cost],feed_dict={X:minibatch_X, Y:minibatch_Y,keep_prob:0.5})

				#累加数据块的成本值
				minibatch_cost += temp_cost / num_minibatches

			#是否打印成本
			if print_cost:
				#每5代打印一次
				if epoch % 10 == 0:
					print("epoch " + str(epoch) + "costs:" + str(minibatch_cost))

			#记录成本
			if epoch % 1 == 0:
				costs.append(minibatch_cost)

		#数据处理完毕，绘制成本曲线
		if isPlot:
			plt.plot(np.squeeze(costs))
			plt.ylabel('cost')
			plt.xlabel('iterations (per tens)')
			plt.title("Learning rate =" + str(learning_rate))
			plt.show()
		#保存学习后的参数
		parameters = sess.run(parameters)
		saver.save(sess,"model/save_net.ckpt")

		#开始预测数据
		## 计算当前的预测情况
		predict_op = tf.argmax(Z5,1)

		corrent_prediction = tf.equal(predict_op , tf.argmax(Y,1))

		##计算准确度
		accuracy = tf.reduce_mean(tf.cast(corrent_prediction,"float"))
		print("corrent_prediction accuracy= " + str(accuracy))

		# X_train = np.reshape(X_train, [-1,12288])
		# X_test = np.reshape(X_test, [-1,12288])
		train_accuracy = accuracy.eval({X: X_train, Y: Y_train, keep_prob:1.0})
		test_accuary = accuracy.eval({X: X_test, Y: Y_test,keep_prob:1.0})

		print("训练集准确度：" + str(train_accuracy))
		print("测试集准确度：" + str(test_accuary))

		return parameters

if __name__ == '__main__':
	X_train_orig , Y_train_orig , X_test_orig , Y_test_orig  = load_dataset()
	X_train,Y_train,X_test,Y_test = init_dataset(X_train_orig , Y_train_orig , X_test_orig , Y_test_orig)
	parameters = model(X_train,Y_train,X_test,Y_test,learning_rate = 0.001,num_epochs = 200,minibatch_size=64,print_cost=True,isPlot=True)



