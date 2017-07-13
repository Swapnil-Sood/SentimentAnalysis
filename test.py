import tflearn
import tensorflow as tf
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tensorflow.contrib import rnn
from preprocess import fn_glove
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score,recall_score,precision_score



train_xg,train_yg,_,_,test_xg,test_yg= fn_glove()



with tf.Graph().as_default():
	n_g= 52
	d_g=50
	
	network = input_data(shape=[None, n_g,d_g])
	network = tf.expand_dims(network,-1)
	network = conv_2d(network, 300,[5,d_g], activation='relu',padding='VALID')
	network = max_pool_2d(network, kernel_size=[1,3,1,1],strides=[1,3,1,1],padding ='VALID')
	network = conv_2d(network, 300,[4,1], activation='relu',padding = 'VALID')
	network = max_pool_2d(network, kernel_size=[1,13,1,1],strides = [1,1,1,1],padding = 'VALID')
	req_lay = fully_connected(network, 300, activation='relu')
	network = dropout(req_lay, 0.5)
	network = fully_connected(network, 3, activation='softmax')
	network = regression(network, optimizer='adam',
	                     loss='categorical_crossentropy',
	                     learning_rate=0.001)

	
	model = tflearn.DNN(network, tensorboard_verbose=0)
	model.load('glove')
	pred = model.predict(test_xg)
	correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(test_yg,1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	with tf.Session() as sess:
		print("CNN Test Accuracy",sess.run(accuracy))
	
	model2 = tflearn.DNN(req_lay, session=model.session)
	X_traing=model2.predict(train_xg)
	X_testg = model2.predict(test_xg)

with tf.Graph().as_default():
	n = 52
	d = 50
	T = n
	L = 64
	learning_rate = 0.001
	n_classes = 3 

	W = tf.Variable(tf.random_normal([2*L]))
	b = tf.Variable(tf.random_normal([1]))

	
	hidden_1_layer = {'weight':tf.Variable(tf.random_normal([2*L, 3])),
	                  'bias':tf.Variable(tf.random_normal([n_classes]))}




	x = tf.placeholder("float", [None, n, d])
	y = tf.placeholder("float", [None, n_classes])


	with tf.variable_scope('lstm'):

	    lstm_fw_cell = rnn.BasicLSTMCell(L, forget_bias=1.0)
	    lstm_bw_cell = rnn.BasicLSTMCell(L, forget_bias=1.0)  
	    outputs,_ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x,
	                                                  dtype=tf.float32,scope = 'lstm')
	    bi_lstm = tf.concat(outputs, 2)

	with tf.variable_scope('lstm1'):

	    lstm_fw_cell1 = rnn.BasicLSTMCell(L, forget_bias=1.0)
	    lstm_bw_cell1 = rnn.BasicLSTMCell(L, forget_bias=1.0)  
	    outputs1,_ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell1, lstm_bw_cell1, bi_lstm,
	                                                  dtype=tf.float32,scope = 'lstm1')
	    bi_lstm1 = tf.concat(outputs1, 2)

	   




	M = tf.tanh(tf.add(tf.einsum("aij,j->ai", bi_lstm1, W),b))
	a = tf.nn.softmax(M)
	r = tf.einsum("aij,ai->aj", bi_lstm1, a)

	pred = tf.add(tf.matmul(r,hidden_1_layer['weight']), hidden_1_layer['bias'])
	


	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

	correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


	init = tf.global_variables_initializer()
	saver = tf.train.Saver()

	with tf.Session() as sess:
	    sess.run(init)
	    saver.restore(sess,"model.ckpt")
	    X_trainw = sess.run(r,feed_dict = {x:train_xg,y:train_yg})
	    X_testw = sess.run(r,feed_dict = {x:test_xg,y:test_yg})

	    print("LSTM Test Accuracy:", \
	        sess.run(accuracy, feed_dict={x:test_xg, y: test_yg}))
	  
	    



X_trainw = np.array(X_trainw)
X_traing = np.array(X_traing)
X_train = np.concatenate((X_traing,X_trainw),axis=1)
Y_train = np.array(train_yg)
X_testw = np.array(X_testw)
X_testg = np.array(X_testg)
X_test = np.concatenate((X_testg,X_testw),axis=1)
Y_test = np.array(test_yg)

Y_train = [np.argmax(j)+1 for j in Y_train]
Y_test = [np.argmax(j)+1 for j in Y_test]
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)



def forest():
	clf = RandomForestClassifier(n_estimators = 300)
	clf.fit(X_train,Y_train)
	y = clf.predict(X_test)
	acc = accuracy_score(Y_test,y)
	print("accuracy:",acc)
	rec = recall_score(Y_test,y,average = 'macro')
	print("Recall:",rec)
	pre = precision_score(Y_test,y,average = 'macro')
	print("Precision:", pre)
	f1 = f1_score(Y_test,y,average = 'macro')
	print("f1 score:",f1)
forest()



