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
discount_rate=0.9
learning_rate = 0.0001
episodes=2000
eval_episodes=50
eval_episodes=10
n_unit=100
epoch=30

model_path = "./task1/"
model_path=model_path+game_id+'A3i'+".ckpt"
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
        loss =tf.reduce_sum(0.5*tf.square(newvals - Qout))
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
        loss =tf.reduce_mean(0.5*tf.square(newvals - Qout))
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return Qout, state, newvals, optimizer, loss 

def build_training_data(env, value_grad, sess):
    episodes_list=[]
    actions = []
    for k in range(episodes):
        transitions = []
        s1 = env.reset()
        states = []
        totalreward=0
        for _ in range(200):
            # calculate policy
            obs_vector = np.expand_dims(s1, axis=0)
            action = randuniformint(0,1)
            # record the transition
            states.append(s1)
            actionblank = np.zeros(2)
            actionblank[action] = 1
            actions.append(actionblank)
            # take the action in the environment
            s = s1
            s1, reward, done= stepwraper(env,action)
            #s1,reward,done,_=env.step(action)
            transitions.append((s,s1, action, reward))
            totalreward += reward
    
            if done:
                break
        episodes_list.append((transitions,states))
    return episodes_list


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
        #s1,reward,done,_=env.step(action)
        totallength += 1
        discountedreturn= np.max(action)+discount_rate*discountedreturn
        totalreturn +=np.max(action)
        if done:
            break
    return totallength,totalreturn,discountedreturn

def eval(episodes):
    t=0
    for _ in range(episodes):
        length,totalreturn,discountedreturn = test(env,value_grad, sess)
        t =t+length
    return t/episodes,totalreturn,discountedreturn
def run_episode_training(env, value_grad, sess):
    vl_calculated, vl_state, vl_newvals, vl_optimizer, vl_loss = value_grad
    totalreward = 0
    episodes_list=build_training_data(env, value_grad, sess)
    bellman_loss=[]
    performance=[]
    empiricalreturn=[]
    discountedempiricalreturn=[]
    for i in range(epoch):
        total_loss=0
        for k, episode in enumerate(episodes_list):
            transitions,states=episode
            update_vals=[]
            states=[]
          
           
            for index, trans in enumerate(transitions):
                s,s1, action, reward = trans
        
                # calculate discounted  return
                states.append(s)
                future_reward = 0
                future_transitions = len(transitions) - index
                decrease = 1
                for index2 in range(future_transitions):
                    future_reward += transitions[(index2) + index][3] * decrease
                    decrease = decrease * discount_rate
                obs_vector = np.expand_dims(s, axis=0)
                x1=np.reshape(s1,[1,4])
                Q0 = sess.run(vl_calculated,feed_dict={vl_state: obs_vector})
                Q1 = sess.run(vl_calculated,feed_dict={vl_state:x1})
                maxQ1 = np.max(Q1)
                targetQ = Q0
                if index == (len(transitions)-1):
                    targetQ[0,action] = future_reward 
                else:
                    targetQ[0,action] = future_reward + discount_rate*maxQ1
                # update the value function towards new return
                update_vals.append(targetQ)
            
            # update value function
            update_vals_vector = np.reshape(update_vals,(-1,2))
            _,loss=sess.run([vl_optimizer,vl_loss],feed_dict={vl_state: states, vl_newvals: update_vals_vector})
            total_loss=total_loss+loss
        print(str(i)+" epoch Mean Loss =  " + "{:.6f}".format(total_loss/len(episodes_list)))
                       
        mean_loss=total_loss/2000
        mean_length,totalreturn,discountedtotalreturn=eval(eval_episodes)
        bellman_loss.append(mean_loss)
        performance.append(mean_length)
        empiricalreturn.append(totalreturn)
        discountedempiricalreturn.append(discountedtotalreturn)
        print("Test step: "+ str(mean_length))
    return bellman_loss,performance,empiricalreturn,discountedempiricalreturn
#env = gym.wrappers.Monitor(env, directory)` to record data.
#env.monitor.start('cartpole-hill/', force=True)



value_grad = qgradient1(learning_rate)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, model_path)
print("Model is restored in file: %s" % model_path)

#bellman_loss,performance,empiricalreturn,discountedempiricalreturn=run_episode_training(env, value_grad, sess)
#run_episode_online(env, value_grad, sess,episilon)

mean_length,totalreturn,discountedtotalreturn=eval(eval_episodes)
print("A3i --\nTest average step length: "+str(mean_length)+ '\n'+'Averaged Empirical Return: ' \
      + str(totalreturn)+'\n'+"Discounted Averaged Empirical Return "+str(discountedtotalreturn))


