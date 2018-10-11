import gym
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.merge import Add
from keras.optimizers import Adam
from keras.models import load_model
import random
from collections import deque

score_requirement = -10
pretrain_length = 100
memory_size = pretrain_length
env = gym.make('BipedalWalker-v2')


class ActorCritic:
    def __init__(self, env, sess,learning_rate = 0.001, hidden_size = 24,batch_size = 32, epsilon = 1.0, 
                 epsilon_decay = .995, gamma = .95, tau   = .125):
        self.env  = env
        self.sess = sess
        
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.tau   = tau
        
        self.batch_size = batch_size
        self.state_size = 14
        self.action_size = 4
        self.memory = deque(maxlen=50000)
        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()

        self.actor_critic_grad = tf.placeholder(tf.float32, 
            [None, self.env.action_space.shape[0]])
        
        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output, 
            actor_model_weights, -self.actor_critic_grad) 
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)   
        self.critic_state_input, self.critic_action_input, \
            self.critic_model = self.create_critic_model()
        _, _, self.target_critic_model = self.create_critic_model()

        self.critic_grads = tf.gradients(self.critic_model.output, 
            self.critic_action_input) 
        self.sess.run(tf.initialize_all_variables())

    def create_actor_model(self):
        state_input = Input(shape=(self.state_size,))
        h1 = Dense(self.hidden_size, activation='relu')(state_input)
        h2 = Dense(self.hidden_size*2, activation='relu')(h1)
        h3 = Dense(self.hidden_size, activation='relu')(h2)
        h4 = Dense(self.hidden_size, activation='relu')(h3)
        output = Dense(self.env.action_space.shape[0], activation='relu')(h3)
        
        model = Model(input=state_input, output=output)
        adam  = Adam(self.learning_rate)
        model.compile(loss="mse", optimizer=adam)
        return state_input, model

    def create_critic_model(self):
        state_input = Input(shape = (self.state_size,))
        state_h1 = Dense(self.hidden_size, activation='relu')(state_input)
        state_h2 = Dense(self.hidden_size*2)(state_h1)
        
        action_input = Input(shape=(self.action_size,))
        action_h1    = Dense(self.hidden_size*2)(action_input)
        
        merged    = Add()([state_h2, action_h1])
        merged_h1 = Dense(self.hidden_size, activation='relu')(merged)
        merged_h2 = Dense(self.hidden_size, activation='relu')(merged_h1)
        merged_h3 = Dense(self.hidden_size, activation='relu')(merged_h2)
        output = Dense(1, activation='relu')(merged_h2)
        model  = Model(input=[state_input,action_input], output=output)
        
        adam  = Adam(self.learning_rate)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model

    def remember(self, cur_state, action, reward, new_state, done):
        self.memory.append([cur_state, action, reward, new_state, done])

    def _train_actor(self, samples):
        for sample in samples:
            cur_state, _, _, _, _ = sample
            predicted_action = self.actor_model.predict(cur_state)
            grads = self.sess.run(self.critic_grads, feed_dict={
                self.critic_state_input:  cur_state,
                self.critic_action_input: predicted_action
            })[0]

            self.sess.run(self.optimize, feed_dict={
                self.actor_state_input: cur_state,
                self.actor_critic_grad: grads
            })
            
    def _train_critic(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, done = sample
            reward = reward.reshape((-1,1))
            if not done:
                target_action = self.target_actor_model.predict(new_state)
                future_reward = self.target_critic_model.predict([new_state, target_action])[0][0]
                reward += self.gamma * future_reward
            self.critic_model.fit([cur_state, action], reward, verbose=0)
        
    def train(self):
        if len(self.memory) < self.batch_size:
            return

        samples = random.sample(self.memory, self.batch_size)
        self._train_critic(samples)
        self._train_actor(samples)

    def _update_actor_target(self):
        actor_model_weights  = self.actor_model.get_weights()
        actor_target_weights = self.target_critic_model.get_weights()
        
        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]
        self.target_critic_model.set_weights(actor_target_weights)

    def _update_critic_target(self):
        critic_model_weights  = self.critic_model.get_weights()
        critic_target_weights = self.critic_target_model.get_weights()
        
        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]
        self.critic_target_model.set_weights(critic_target_weights)        

    def update_target(self):
        self._update_actor_target()
        self._update_critic_target()

    def act(self, cur_state):
        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample().reshape((1,4))
        return self.actor_model.predict(cur_state)
    
    def save_actor(self):
        self.actor_model.save("model_actor2.h5")
        print("Saved actor model to disk")
        
    def save_critic(self):
        self.critic_model.save("model_critic2.h5")
        print("Saved critic model to disk")
        
    def load_actor(self):
        self.actor_model = load_model("model_actor2.h5")
        print("Loaded actor model from disk")
        
    def load_critic(self):
        self.critic_model = load_model("model_critic2.h5")
        print("Loaded critic model from disk")

        
class PreTrain():
    def __init__(self,env = gym.make('BipedalWalker-v2'), score_requirement = -100, memory_size = 2000):
        self.score_requirement = score_requirement
        self.memory = []
        self.env = env        
    
    def run(self,pretrain_length):
        self.env.reset()
        state, reward, done, _ = self.env.step(self.env.action_space.sample())
        state=state[:14]
        
        trwd = .0
        print('Generating pre-training experiences.')
        pbar = tqdm(total = pretrain_length)
        while(len(self.memory) < pretrain_length):
            env.render()
            next_state, reward, done, _ = self.env.step()
            
                
            next_state=next_state[:14] 
            trwd +=reward 
            if trwd < self.score_requirement:
                self.env.reset()
                trwd = 0
                continue;
            if done:
                self.env.reset()
                trwd = 0
                self.experience.append([state, action, reward, next_state, done])
                state, reward, done, _ = self.env.step(self.env.action_space.sample())
                state=state[:14]
            else:
                self.memory.append([state, action, reward, next_state,done])
                state = next_state
            pbar.update()    
        pbar.close()   
        print('Pre-Training Experienses: ', len(self.memory),'Average Reward: ', trwd/len(self.memory))
        return  self.memory
    
    def save_experiences(self):
        df = pd.DataFrame(self.memory)
        df.to_csv('pretraining_data/pre_train_data.csv', index=False, header=False)
    def load_experiences(self ): 
        df = pd.read_csv('pretraining_data/pre_train_data.csv', header=None)
        self.memory = df.reset_index().values.tolist()[0][1]
        print(self.memory)
        trwd = .0
        for reward_ind in range(len(self.memory[2])):
            trwd = trwd + self.experience[2][reward_ind]
            
        print('Pre-Training Experiences loaded from files.', self.memory.size(),'Average Reward: ', trwd/ self.memory.size())
        
        return self.memory
    
    def get_score_requirement(self):
        return self.score_requirement
    def get_memory_size(self):
        return self.memory.size()