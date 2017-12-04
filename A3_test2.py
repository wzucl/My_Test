'''
Created on 19 Mar 2017

@author: Wilbur
'''
import tensorflow as tf
import numpy as np
import random
import gym
import math
import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt

game_id='CartPole-v0'
env = gym.make(game_id)
discount_rate=0.99
episodes=2000
eval_episodes=50
episilon=0.002


n_unit=100
epoch=30
learning_rate=0.001
model_path = "./task1/"
model_path=model_path+game_id+'A3ii'+".ckpt"
#model_path=model_path+game_id+'A3i'+".ckpt"
def stepwraper(env,action):
    s,rd,done,_=env.step(action)
    if done:
        rd=-1
    else:
        rd=0
    return s,rd,done
def randuniformint(low,high):
    r=np.random.uniform(low,high)
    threshold=(low+high)/2
    if r>threshold:
        out=1
    else:
        out=0
    return out

def qgradient1(learning_rate): #linear layer
    with tf.variable_scope("value"):
        state = tf.placeholder("float",[None,4])
        newvals = tf.placeholder("float",[None,2])
        w1 = tf.get_variable("w1",[4,2])
        b1 = tf.get_variable("b1",[2])
        Qout = tf.matmul(state,w1) + b1
        loss =tf.reduce_sum(tf.square(newvals - Qout))
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return Qout, state, newvals, optimizer, loss
def qgradient2(learning_rate): # hidden layer
    with tf.variable_scope("value"):
        state = tf.placeholder("float",[None,4])
        newvals = tf.placeholder("float",[None,2])
        w1 = tf.get_variable("w1",[4,n_unit])
        b1 = tf.get_variable("b1",[n_unit])
        h1 = tf.nn.relu(tf.matmul(state,w1) + b1)
        w2 = tf.get_variable("w2",[n_unit,2])
        b2 = tf.get_variable("b2",[2])
        Qout = tf.matmul(h1,w2) + b2
        maxQ=np.max(Qout)
        stopgrad=tf.stop_gradient(maxQ)       
        loss =tf.reduce_mean(tf.square(newvals - Qout))
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return Qout, state, newvals, optimizer, loss 


def test(env, value_grad, sess):
    vl_calculated, vl_state, vl_newvals, vl_optimizer, vl_loss = value_grad
    s1 = env.reset()
    totallength = 0
    totalreturn=0
    discountedreturn=0
    for _ in range(300):
        # calculate policy
        obs_vector = np.expand_dims(s1, axis=0)
        Q0 = sess.run(vl_calculated,feed_dict={vl_state: obs_vector})
        # record the transition
        action=np.argmax(Q0)
        s1, reward, done = stepwraper(env,action)
        totallength += 1
        discountedreturn= np.max(action)+discount_rate*discountedreturn
        totalreturn +=np.max(action)
        if done:
            break
    return totallength,totalreturn,discountedreturn

def eval(episodes,env,value_grad,sess):
    t=0
    for _ in range(episodes):
        length,totalreturn,discountedreturn = test(env,value_grad, sess)
        t =t+length
    return t/episodes,totalreturn,discountedreturn

value_grad = qgradient2(learning_rate)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver = tf.train.Saver()
saver.restore(sess, model_path)
print("Model is restored in file: %s" % model_path)
mean_length,totalreturn,discountedtotalreturn=eval(eval_episodes,env,value_grad,sess)
print("A3ii--\nTest average step length: "+str(mean_length)+ '\n'+'Averaged Empirical Return: ' \
      + str(totalreturn)+'\n'+"Discounted Averaged Empirical Return "+str(discountedtotalreturn))