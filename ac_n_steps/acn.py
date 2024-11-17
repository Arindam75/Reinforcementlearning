import torch as T
import numpy as np
from torch import cuda, device, distributions
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ActorCritic(nn.Module):
    def __init__(self, s_size, a_size, lyr1=128, lyr2=256,
                gamma = 0.99, lr = 0.0001):
        super(ActorCritic, self).__init__()
        
        self.gamma = gamma
        self.lyr1 = lyr1
        self.lyr2 = lyr2
        self.linear1 = nn.Linear(s_size, self.lyr1)
        
        self.pi_linear2 = nn.Linear(self.lyr1, self.lyr2)
        self.pi = nn.Linear(self.lyr2, a_size)
        
        self.v_linear2 = nn.Linear(self.lyr1, self.lyr2)
        self.v = nn.Linear(self.lyr2, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr = lr)
    
    def forward(self, x):
        x = self.linear1(x)
        pi = F.relu(x)
        pi = self.pi_linear2(pi)
        pi = F.relu(pi)
        pi = F.softmax(self.pi(pi), dim=1)
        
        v = F.relu(x)
        v = self.v_linear2(v)
        v = F.relu(v)
        v = self.v(v)
        return pi, v
    
    def calc_loss(self, states, actions, rewards, done, s_):
            states_t = T.tensor(states, dtype=T.float32)
            action_t = T.tensor(actions)
                    
            if done:
                #vs_ = 0
                R = 0
            else:
                _, vs_ = self(T.FloatTensor(np.expand_dims(s_,0)))
                R = vs_.item()
            
            returns = []
            for r in reversed(rewards):
                R = r + self.gamma * R
                returns.insert(0, R)
            returns = T.tensor(returns)
            
            probs , vs = self.forward(states_t)
            dist = distributions.Categorical(probs)
            log_probs = dist.log_prob(action_t)
        
            adv = returns - (vs.squeeze())
            
            actor_loss = -(log_probs * adv.detach()).mean()
            critic_loss = adv.pow(2).mean()
            loss = actor_loss + 0.5 * critic_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def save_model(self, model_file):
        T.save(self.state_dict(), model_file)
    
    def load_model(self, model_file):
        self.load_state_dict(T.load(model_file))