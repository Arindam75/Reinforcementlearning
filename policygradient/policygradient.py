import os

import torch as T
import numpy as np
from torch import cuda, device, distributions
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

def calc_disc_return(r_t , gamma = 0.998):
    
    """
    Calculate the discounted return for a sequence of rewards.

    The discounted return (also known as return or cumulative reward) is the sum of rewards
    received over time, where future rewards are exponentially discounted by a factor of `gamma`.
    This function computes the discounted return for each time step in a trajectory, working 
    backwards from the final reward.

    Parameters:
    ----------
    r_t : list or np.ndarray
        A sequence of rewards at each time step. The length of this sequence is the length of 
        the episode or trajectory.
    gamma : float, optional
        The discount factor, which should be a value between 0 and 1. The default value is 0.998.

    Returns:
    -------
    np.ndarray
        An array of discounted returns for each time step in the input sequence. Each element in 
        the returned array represents the total discounted return starting from that time step 
        until the end of the sequence.
    
    Example:
    --------
    >>> r_t = [1, 2, 3, 4]
    >>> calc_disc_return(r_t, gamma=0.99)
    array([ 9.40703,  8.51114,  6.5556 ,  4.    ])
    """

    G_t = deque(maxlen = len(r_t))
    G_t.append(r_t[-1])

    for i in reversed(r_t[:-1]):
        disc = i + (gamma*G_t[0])
        G_t.appendleft(disc)

    return np.array(G_t)

class PolicyNet(nn.Module):
    def __init__(self, s_size, a_size, lyr1, lyr2):
        super(PolicyNet, self).__init__()
        self.lyr1 = lyr1
        self.lyr2 = lyr2
        self.linear1 = nn.Linear(s_size, self.lyr1)
        self.linear2 = nn.Linear(self.lyr1, self.lyr2)
        self.output = nn.Linear(self.lyr2, a_size)

    def forward(self, x):
        #x = T.clamp(x,-1.1,1.1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return F.softmax(self.output(x),dim = 1)
    
class ValueNet(nn.Module):
    def __init__(self, s_size, lyr1, lyr2):
        super(ValueNet, self).__init__()
        self.linear1 = nn.Linear(s_size, lyr1)
        self.linear2 = nn.Linear(lyr1, lyr2)
        self.output = nn.Linear(lyr2, 1)

    def forward(self, x):
        #x = T.clamp(x,-1.1,1.1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.output(x)

class ReInforce():
    def __init__(self,
                 s_size,
                 a_size,
                 lyr1 = 256,
                 lyr2 = 512,
                 gamma = 0.99,
                 lr = 0.001):
        
        self.state_size = s_size
        self.action_size = a_size
        self.lyr1 = lyr1
        self.lyr2 = lyr2
        self.gamma = gamma

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')             
        self.policy_net = PolicyNet(s_size, a_size, lyr1, lyr2).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr = lr )
    
    def train(self, states, rewards, actions, loss_scaler = 1):
        
        state_t = T.FloatTensor(states).to(self.device)
        action_t = T.LongTensor(actions).to(self.device).view(-1,1)
        return_t = T.FloatTensor(calc_disc_return(rewards, self.gamma)).to(self. device).view(-1,1)

        action_prob = self.policy_net(state_t).gather(1, action_t)
        loss = T.sum(-T.log(action_prob) * return_t)*loss_scaler
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
        return loss.item()
    
    def save(self, model_file):
        T.save(self.policy_net.state_dict(), model_file)
    
    def load(self, model_file):
        self.policy_net.load_state_dict(T.load(model_file))

class ReInforceBaseline():
    def __init__(self,
                 s_size,
                 a_size,
                 lyr1 = 256,
                 lyr2 = 512,
                 gamma = 0.99,
                 pi_lr = 0.001,
                 vf_lr = 0.0001):
        
        self.state_size = s_size
        self.action_size = a_size
        self.lyr1 = lyr1
        self.lyr2 = lyr2
        self.gamma = gamma

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')  
        
        self.policy_net = PolicyNet(s_size, a_size, lyr1, lyr2).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr = pi_lr )
        self.value_net = ValueNet(s_size, lyr1, lyr2).to(self.device)
        self.v_optimizer = optim.Adam(self.value_net.parameters(), lr = vf_lr)
        
    def train(self, states, rewards, actions, loss_scaler = 1):
        
        state_t = T.FloatTensor(states).to(self.device)
        action_t = T.LongTensor(actions).to(self.device).view(-1,1)
        return_t = T.FloatTensor(calc_disc_return(rewards, self.gamma)).to(self.device).view(-1,1)
        
        vf_t = self.value_net(state_t).to(self.device)
        with T.no_grad():
            adv_t = return_t - vf_t
            
        action_prob = self.policy_net(state_t).gather(1, action_t)
        loss = T.sum(-T.log(action_prob) * adv_t)*loss_scaler
    
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
        loss_fn = nn.MSELoss()
        vf_loss = loss_fn(vf_t, return_t)
        self.v_optimizer.zero_grad()
        vf_loss.backward()
        self.v_optimizer.step()
   
        return loss.item(), adv_t[0]
    
    def save(self, model_file):
        T.save({
            'policy_net_dict': self.policy_net.state_dict(),
            'value_net_dict': self.value_net.state_dict(),
            }, model_file)
        
    def load(self, model_file):
        checkpoint = T.load(model_file)
        self.policy_net.load_state_dict(checkpoint['policy_net_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_dict'])
'''
class ActorCritic():
    def __init__(self,
                 s_size,
                 a_size,
                 lyr1 = 256,
                 lyr2 = 512,
                 gamma = 0.99,
                 lr = 0.001):
        
        self.state_size = s_size
        self.action_size = a_size
        self.lyr1 = lyr1
        self.lyr2 = lyr2
        self.gamma = gamma
        self.log_prob = None

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')  
        
        self.actor = PolicyNet(s_size, a_size, lyr1, lyr2).to(self.device)
        self.ac_optimizer = optim.Adam(self.actor.parameters(), lr = lr )
        self.critic = ValueNet(s_size, lyr1, lyr2).to(self.device)
        self.cr_optimizer = optim.Adam(self.critic.parameters(), lr = lr)
    
    def learn(self, s, a ,r , s_, done):

        state_t = T.from_numpy(np.expand_dims(s,0)).to(self.device)
        state_t_ = T.from_numpy(np.expand_dims(s_,0)).to(self.device)
        probs = self.actor(state_t)
        log_prob = T.log(probs[0,a])
        
        value = self.critic(state_t)
        value_ = self.critic(state_t_)
        delta = r + self.gamma * value_*(1-int(done)) - value
        
        self.ac_optimizer.zero_grad()
        actor_loss = -log_prob * delta.item()
        actor_loss.backward()
        self.ac_optimizer.step() 

        self.cr_optimizer.zero_grad()
        critic_loss = delta**2
        critic_loss.backward()
        self.cr_optimizer.step()
'''
class ActorCritic():
    def __init__(self,
                 s_size,
                 a_size,
                 lyr1 = 256,
                 lyr2 = 512,
                 gamma = 0.99,
                 lr_ac = 0.0002,
                 lr_cr = 0.0001):
        
        self.state_size = s_size
        self.action_size = a_size
        self.lyr1 = lyr1
        self.lyr2 = lyr2
        self.gamma = gamma

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')             
        
        self.actor = PolicyNet(s_size, a_size, lyr1, lyr2).to(self.device)
        self.critic = ValueNet(s_size, lyr1, lyr2).to(self.device)
        
        self.ac_optimizer = optim.Adam(self.actor.parameters(), lr = lr_ac)
        self.cr_optimizer = optim.Adam(self.critic.parameters(), lr = lr_cr)

    def learn(self, state, action, reward, state_, done):
        
        state = T.FloatTensor(np.expand_dims(state,0)).to(self.device)
        state_ = T.FloatTensor(np.expand_dims(state_,0)).to(self.device)

        value = self.critic(state).squeeze(dim =-1)
        probs = self.actor(state).squeeze()
        
        with T.no_grad():
            value_ = self.critic(state_).squeeze(dim =-1)
        delta = (reward + self.gamma * value_*(1-int(done))) - value
        
        self.ac_optimizer.zero_grad()
        actor_loss = (-T.log(probs[action]) * delta.item())
        actor_loss.backward()
        self.ac_optimizer.step()

        self.cr_optimizer.zero_grad()
        critic_loss = T.pow(delta,2)
        critic_loss.backward()
        self.cr_optimizer.step()

        return actor_loss.item(), critic_loss.item()
    
    def save(self, model_file):
        T.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            }, model_file)
        
    def load(self, model_file):
        checkpoint = T.load(model_file)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])