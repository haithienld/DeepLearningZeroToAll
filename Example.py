#Khai bao ham so
hello = tf.constant("Hello, Tensorflow")

#Khoi tao TF Session de lam viec
sess = tf.Session()

#run the operator and get result 
print (sess.run(hello))


#Tinh toan do thi

node1 = tf.constant(3.0,tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1,node2)
print("Node1:",node1, " Node2:",node2," Node3:",node3)

print (sess.run([node1,node2,node3]))

#Su dung placeholder

a=tf.placeholder(tf.float32)
b=tf.placeholder(tf.float32)
adder_node = a+b

print(sess.run(adder_node,feed_dict={a:3,b:4.5}))
print(sess.run(adder_node, feed_dict ={a:[1,2,3], b: [2,4,4]}))

#Example 2
from __future__ import print_function
import os

import tensorflow as tf
import random

x_train = [1,2,3]
y_train = [1,2,3]

W = tf.Variable(tf.random_normal([1]), name ='weight')
b = tf.Variable(tf.random_normal([1]), name ='bias')

X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

# Hàm giả thuyết

hypothesis = W *x_train + b

#loss function 
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

optimizer = tf.train.GradientDescentOptimizer(learning_rate =0.01)

train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range (4001):
    cost_val, W_val, b_val, _ = sess.run([cost,W,b,train], feed_dict = {X: [1,2,3,4,5], Y: [2.1,3.1,4.1,5.1,6.1]})
    if(step%20 ==0):
        print(step, cost_val,W_val,b_val)

#Example 3

W = tf.placeholder(tf.float32)

hypothesis = X*W

#cost / loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

#Launch the graph in a session

sess = tf.Session()

sess.run(tf.global_variables_initializer())

#Cách đơn giản nhất là khởi tạo tất cả các biến trong 1 lần:
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#Khởi tạo một số biến nhất định:
#with tf.Session() as sess:
#    sess.run(tf.variables_initializer([a, b]))
#Khởi tạo 1 biến:
#W = tf.Variable(tf.zeros([784,10]))
#with tf.Session() as sess:
#    sess.run(W.initializer)

W_val = []
cost_val = []

# https://machinelearningcoban.com/2017/01/12/gradientdescent/

for i in range (-30,50):
    feed_W = i * 0.1
    curr_cost, curr_W = sess.run([cost,W], feed_dict = {W: feed_W})
    W_val.append(curr_W)
    cost_val.append(curr_cost)
plt.plot(W_val,cost_val)
plt.show()


#Example 4

from __future__ import print_function
import os

import tensorflow as tf
import random
import matplotlib.pyplot as plt

X = [1,2,3]
Y = [1,2,3]

W = tf.Variable(5.)

hypothesis = W*X

#Doc https://khanh-personal.gitbook.io/ml-book-vn/chapter1/ham-mat-mat
#Manual gradient
gradient = tf.reduce_mean((W*X-Y)*X)*2

cost = tf.reduce_mean(tf.square(hypothesis-Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)

#Get gradients 
gvs = optimizer.compute_gradients(cost,[W])

#Apply gradients

apply_gradients = optimizer.apply_gradients(gvs)

sess = tf.Session()

sess.run (tf.global_variables_initializer())
for step in range(100):
    print(step, sess.run([gradient,W,gvs]))
    sess.run(apply_gradients)

