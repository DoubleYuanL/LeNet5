import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
import cnn_utils
import cv2
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
 #predict

def predict():
	X,_,keep_prob = create_placeholder(64, 64, 3, 6)

	parameters = init_parameters()

	Z5 = forward_propagation(X, parameters,keep_prob)

	Z5 = tf.argmax(Z5,1)

	init = tf.global_variables_initializer()
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(init)
		saver.restore(sess,tf.train.latest_checkpoint("model/"))

		#use the sample picture to predict the unm
		sample = 1
		cam = 1
		if (sample):
			num = 0
			my_image = "sample/" + str(num) + ".jpg"	
			num_px = 64
			fname =  my_image 
			image = np.array(ndimage.imread(fname, flatten=False))#.astype(np.float32)
			my_predicted_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1,64,64,3))/255
			my_predicted_image = my_predicted_image.astype(np.float32)

			my_predicted_image = sess.run(Z5, feed_dict={X:my_predicted_image,keep_prob:1.0})

			plt.imshow(image) 
			print("prediction num is : y = " + str(np.squeeze(my_predicted_image)))
			plt.show()
			num = num + 1
		elif(cam):# use the camera to predict the num
			cap = cv2.VideoCapture(0)
			while (1):
				num = 0
				ret, frame = cap.read()
				cv2.namedWindow("capture")
				cv2.imshow("capture", frame)
				k = cv2.waitKey(1) & 0xFF
				if  k == ord('s'):
					frame = cv2.resize(frame, (int(256), int(256)))
					cv2.imwrite("sample/cam/" + str(num)+".jpg", frame)

					my_image = "sample/cam/" + str(num) + ".jpg"	
					num_px = 64
					fname =  my_image 
					image = np.array(ndimage.imread(fname, flatten=False))#.astype(np.float32)
					my_predicted_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1,64,64,3))/255
					my_predicted_image = my_predicted_image.astype(np.float32)

					my_predicted_image = sess.run(Z5, feed_dict={X:my_predicted_image,keep_prob:1.0})

					plt.imshow(image) 
					print("预测结果: y = " + str(np.squeeze(my_predicted_image)))
					plt.show()
					num = num + 1
				elif k == ord('q'):
					break
			cap.release()
			cv2.destroyAllWindows()

if __name__ == '__main__':
	predict()


