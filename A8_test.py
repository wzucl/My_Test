import tensorflow as tf
import numpy as np
import random
import gym
import math
import matplotlib.pyplot as plt
epsilon=0.05
 # Probability of selecting random action
epsilon_min=0.0 # Minimum random action selection probability
num_episodes=500000
test_episode=10
test_gap=20
update_episode=5
max_stpes=300
test_run=2
switch_freq=5000
epsilon_decay=(epsilon - epsilon_min) / max_stpes
#import matplotlib.pyplot as plt
discount_rate=0.99
n_unit=100
n_action=2
learning_rate=0.00001
game_id='CartPole-v0'
model_path = "./task1/"
model_pathname=model_path+game_id+'A8_N'+str(n_unit)+".ckpt"
def stepwraper(env,action):
    s,rd,done,_=env.step(action)
    if done:
        rd=-1
    else:
        rd=0
    return s,rd,done
def weight_variable(name,shape):
    initial=tf.get_variable(name,shape)
#   initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
#current network
net0 = {}
#layer 1
net0['w1'] = tf.get_variable('w1',[4,n_unit])
net0['b1'] = bias_variable([n_unit])
#layer 2
net0['w2'] = tf.get_variable('w2',[n_unit,n_action])
net0['b2'] = bias_variable([n_action])
#target network
net1 = {}
#layer 1
net1['w1'] = tf.get_variable('net1_w1',[4,n_unit])
net1['b1'] = bias_variable([n_unit])
#layer 2
net1['w2'] = tf.get_variable('net1_w2',[n_unit,n_action])
net1['b2'] = bias_variable([n_action])
tmp = {}
#layer 1
tmp['w1'] = tf.get_variable('tmp_w1',[4,n_unit])
tmp['b1'] = bias_variable([n_unit])
#layer 2
tmp['w2'] = tf.get_variable('tmp_w2',[n_unit,n_action])
tmp['b2'] = bias_variable([n_action])
def swap_networks(sess):
    for name in net0.keys():
        assign_op1=tmp[name].assign(net0[name])
        assign_op2=net0[name].assign(net1[name])
        assign_op3=net1[name].assign(tmp[name])
        sess.run([assign_op1,assign_op2,assign_op3])

state_ph = tf.placeholder("float",[1,4])
newstate_ph = tf.placeholder("float",[1,4])
action_mask_ph=tf.placeholder("float",[n_action,1])
reward_ph=tf.placeholder("float")
TQ_ph = tf.placeholder(tf.int32)

        
def predictor(state,model):
    h0 = tf.nn.relu(tf.matmul(state,model['w1']) + model['b1'])
    q0=tf.matmul(h0,model['w2']) + model['b2']
    return q0
# Q0,Q1=predictor(state_ph,newstate_ph,net0,net1)
# 
# T0 = reward_ph + discount_rate *(reward_ph+1)* tf.stop_gradient(tf.reduce_max(Q1, 1))
# Qm=tf.matmul(Q0,action_mask_ph)
# Loss = tf.reduce_sum(0.5*tf.square(Qm-T0))
# Optimizer = tf.train.AdamOptimizer(learning_rate).minimize(Loss)



# if tf.mod(TQ_ph, 2) == 1: # switch Q1 to Q0
#     Q0 = predictor(state_ph,net1)
#     Q1 = predictor(newstate_ph,net0)
#     Q01 = predictor(newstate_ph,net1)
# else:
Q0 = predictor(state_ph,net0)
Q1 = predictor(newstate_ph,net1)
Q01 = predictor(newstate_ph,net0)
# Q0 = predictor(obs0,W0,B0)
# Q1 = predictor(obs1,W0,B0)
# Q01 = predictor(obs1,W0,B0)

act1=tf.argmax(Q01, 1)
act1_1hot = tf.one_hot(act1, n_action, axis=0, dtype='float32')
act1_1hot=tf.reshape(act1_1hot,[n_action,1])
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.matmul(Q0,act1hot), 
#                         reward + discountFactor * tf.stop_gradient(tf.reduce_max(Q1,1))))
# Qa0 = tf.add(learningRate * loss0,tf.matmul(Q0,act1hot))
T = reward_ph + (1.0+reward_ph)*discount_rate * tf.stop_gradient(tf.matmul(Q1,act1_1hot))
Q = tf.matmul(Q0,action_mask_ph)
Loss =  tf.reduce_mean(tf.square(T - Q)) 
Optimizer = tf.train.AdamOptimizer(learning_rate).minimize(Loss)    

def run_episode_test(env, sess):
    s = env.reset()
    totalreward = 0
    total_q=0
    discounted_total_q=0
    total_loss=0
    for _ in range(max_stpes):
        # calculate policy
        s_vector = np.reshape(s, [1,4])
        q0 = sess.run(Q0,feed_dict={state_ph: s_vector})
        action = np.argmax(q0)
        s1, reward, done= stepwraper(env,action)
        totalreward += 1
        total_q+=np.max(q0)
        discounted_total_q=discounted_total_q*discount_rate+np.max(q0)
        action_mask=np.zeros(n_action)
        action_int=(int)(action)
        action_mask[action_int]=1
        action_mask=np.reshape(action_mask,[n_action,1])
        reward_vec=np.reshape(reward,[1,1])
        s1_vector = np.reshape(s1, [1,4])       
        loss=sess.run(Loss, feed_dict={state_ph:s_vector, newstate_ph: s1_vector,action_mask_ph:action_mask,reward_ph:reward_vec}) 
        s=s1   
        total_loss+=loss
        if done:
            break
    mean_loss=total_loss/totalreward
    return totalreward,total_q,discounted_total_q,mean_loss
def run_eval(env, sess):
    total_rd,total_q,total_discounted_q,total_loss=0,0,0,0
    for _ in range(test_episode):
        
        result_rd,result_q,result_discount_q,m_loss=run_episode_test(env, sess)
        total_rd+=result_rd
        total_q+=result_q
        total_discounted_q+=result_discount_q
        total_loss+=m_loss
    return total_rd/test_episode,total_q/test_episode,total_discounted_q/test_episode,total_loss/test_episode

def run_episodes(env):
    size=(int)(num_episodes/test_gap)
    episode_rd_l=np.zeros(size)
    episode_q_l=np.zeros(size)
    episode_discounted_q=np.zeros(size)
    episode_loss=np.zeros(size)
    numStep = 0
    old_swflag=False
    for k in range(test_run):
        test_idx=0
        sess = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        for i in range(num_episodes):
            s = env.reset()
            totalreward = 0         
            lepsilon=epsilon
            for steps in range(max_stpes):
                #update_vals = []
                s_vector = np.reshape(s, [1,4])
                tq = ((numStep%(switch_freq*2)>switch_freq))
                if old_swflag != tq:
                    swap_networks(sess)
                    print("Network swap")
                old_swflag = tq
                q0 = sess.run(Q0,feed_dict={state_ph: s_vector,TQ_ph: tq})
                if np.random.uniform(0,1) < lepsilon:
                    # Either 0 or 1 sample the action randomly
                    action = np.random.randint(2)
                #    print("take Random action")
                else:
                    action = np.argmax(q0)       
                s1, reward, done = stepwraper(env,action)
                numStep+=1               
                action_mask=np.zeros(2)
                action_int=(int)(action)
                action_mask[action_int]=1
                action_mask=np.reshape(action_mask,[n_action,1])
                reward_vec=np.reshape(reward,[1,1])
                s1_vector = np.reshape(s1, [1,4])
                _,loss=sess.run([Optimizer,Loss], feed_dict={state_ph:s_vector, newstate_ph: \
                                                s1_vector,action_mask_ph:action_mask,reward_ph:reward_vec,TQ_ph: tq}) 
                totalreward += 1
                s=s1
                if done:
                    if (i%50)==0:
                        print(str(k)+" run "+str(i) + " episode "+"Loss =  " + \
                              "{:.6f}".format(loss) + ", Reward= " + \
                              "{:.5f}".format(totalreward))
                    break
            if (i%test_gap)==0:           
                average_reward,average_q,average_discounted_q,mean_loss=run_eval(env, sess)
                episode_rd_l[test_idx]+=(average_reward/test_run)
                episode_q_l[test_idx]+=(average_q/test_run)
                episode_discounted_q[test_idx]+=(average_discounted_q/test_run)
                episode_loss[test_idx]+=(mean_loss/test_run)
                test_idx+=1
            if (i%50)==0:
                print(str(k)+" run "+str(i) + " episode "+"Test reward =  " + \
                              "{:.6f}".format(average_reward) )
        save_path = saver.save(sess, model_pathname)
        sess.close()
        print("Model saved in file: %s" % save_path)
        # real_vl_loss = sess.run(vl_loss, feed_dict={vl_state: states, vl_newvals: update_vals_vector})

    #advantages_vector = np.expand_dims(advantages, axis=1)
    #sess.run(pl_optimizer, feed_dict={pl_state: states, pl_advantages: advantages_vector, pl_actions: actions})
        lepsilon*=0.9
    return episode_rd_l,episode_q_l,episode_discounted_q,episode_loss




env = gym.make(game_id)


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, model_pathname)
print("Model is restored in file: %s" % model_pathname)
mean_length,totalreturn,discountedtotalreturn,mean_loss=run_eval(env,sess)
print("A8--\nTest average step length: "+str(mean_length)+ '\n'+'Averaged Empirical Return: ' \
      + str(totalreturn)+'\n'+"Discounted Averaged Empirical Return "+str(discountedtotalreturn))

plt.show()
