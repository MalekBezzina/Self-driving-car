#AI for self driving car

#import libraries

import numpy as np #manipulate arrays
import random #for random samples
import os #to load the model when we save the brain and load the brain
import torch #to impliment the NN
import torch.nn as nn
import torch.nn.functional as F #to use the loss function and other function to implement the NN
import torch.optim as optim #for the optimization
import torch.autograd as autograd #to make convertion from tensors to gradients
from torch.autograd import Variable

#Creating the architecture of the Neural Network

class Network (nn.Module): #to inherite the tools to implement the NN
    
    def __init__(self,input_size,nb_action):
        super(Network,self).__init__() #to inhirite the init function
        self.input_size=input_size  #equal to 5= 3 sensors+ orientation +(-orientation) 
        self.nb_action=nb_action #3= forward , left ,  right
        self.fc1=nn.Linear(input_size,30) 
        self.fc2=nn.Linear(30,nb_action)
    
    def forward(self, state): #activate the neurones and return the Q values for each possible action
        x= F.relu(self.fc1(state))  #activates the neurons of the hidden layer
        q_values=self.fc2(x) #output neurons of our NN
        return q_values #left , right , forward  
    
#Implement Experience Relay : concider the current state and the other states before it(100 states or transitions)

class ReplayMemory(object):
    
    def __init__(self,capacity):
        self.capacity=capacity #the max nbre of transitions we want to have in our memory of events
        self.memory=[] #contains the last 100 events/ transitions

    def push (self,event): #used to append a new event(last state,new state,last action,last reward) to the memory 
        self.memory.append(event)
        if len(self.memory)>self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)  # zip(*) is like a reshape function 
                                                                  #example:  if list =((1,2,3),(4,5,6)) the zip(*list)=((1,4),(2,3),(5,6)) 


#Implement the Deep Q Learning

class Dqn():
    
    def __init__(self,input_size,nb_action, gamma):
        self.gamma=gamma
        self.reward_window=[]
        self.model=Network(input_size,nb_action)
        self.memory=ReplayMemory(100000) #100000 is the nbre of transitions
        self.optimizer=optim.Adam(self.model.parameters(), lr =0.001)#to connect our adam optimizor with our NN , lr= the learning rate
        self.last_state=torch.Tensor(input_size).unsqueeze(0)
        self.last_action= 0
        self.last_reward=0
    
    def select_action(self, state): # function that selects the right action each time
        #get best action to play while still exploring the other actions
        probs = F.softmax(self.model(Variable(state, volatile = True))*100)  #temperature  T=100 if T=0 the the AI  doesn't work
        #softmax([1,2,3])=[0.04,0.11,0.85] => softmax([1,2,3]*3)=[0,0.02,0.98]  it's about the cetrainty of which action we have to choose
        action = probs.multinomial(1)#random draw
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        #td_loss.backward(retain_variables = True)
        td_loss.backward()
        self.optimizer.step()

    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum (self.reward_window)/(len(self.reward_window)+1.)
    
    def save(self):
        torch.save({'state_dict':self.model.state_dict(),
                    'optimizer':self.optimizer.state_dict(),
                    },'last_brain.pth')
        
    def load(self):
        if os.path('last_brain.pth'):
            print("=> loading checkpoint...")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])  
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Done!")
        else:
            print("no checkpoint found...")
             
            
            
            
            
    





