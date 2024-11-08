import torch as T
import numpy as np
from torch import cuda, device, distributions
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class OUActionNoise():
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x

        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
        
class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        #self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)
        self.terminal_memory = np.zeros(self.mem_size, dtype = np.int32)

    def store(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        #self.terminal_memory[index] = done
        self.terminal_memory[index] = int(done)

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


    def get_samples(self,
                    batch_size):
        
        buffer_level = min(self.counter, self.size)
        samples = np.random.choice(buffer_level, batch_size, replace=False)
        
        s = self.state_buffer[samples]
        a = self.action_buffer[samples]
        r = self.rewards_buffer[samples]
        s_ = self.new_state_buffer[samples]
        dones = self.done_buffer[samples]

        return s, a, r, s_, dones
    
class CriticNet(nn.Module):
    def __init__(self, lr, s_size, lyr1, lyr2, a_size):
        super(CriticNet, self).__init__()
        self.s_size = s_size
        self.lyr1 = lyr1
        self.lyr2 = lyr2
        self.a_size = a_size
        
        self.linear1 = nn.Linear(self.s_size, self.lyr1)
        self.linear2 = nn.Linear(self.lyr1, self.lyr2)
        self.bn1 = nn.LayerNorm(self.lyr1)
        self.bn2 = nn.LayerNorm(self.lyr2)
        
        self.action_value = nn.Linear(self.a_size, self.lyr2)
        self.q = nn.Linear(self.lyr2, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr,
                                    weight_decay=0.01)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')

        self.to(self.device)

    def forward(self, state, action):
        x = self.linear1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.bn2(x)
        #state_value = F.relu(state_value)
        #action_value = F.relu(self.action_value(action))
        y = self.action_value(action)
        q = F.relu(T.add(x, y))
        #state_action_value = T.add(state_value, action_value)
        state_action_value = self.q(q)

        return state_action_value
    
class ActorNet(nn.Module):
    def __init__(self, lr, s_size, lyr1, lyr2, a_size):
        super(ActorNet, self).__init__()
        self.s_size = s_size 
        self.lyr1 = lyr1
        self.lyr2 = lyr2
        self.a_size = a_size
        
        self.linear1 = nn.Linear(self.s_size, self.lyr1)
        self.linear2=nn.Linear(self.lyr1, self.lyr2)
        self.mu = nn.Linear(self.lyr2, self.a_size)
        
        self.bn1 = nn.LayerNorm(self.lyr1)
        self.bn2 = nn.LayerNorm(self.lyr2)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')

        self.to(self.device)
        
    def forward(self, state):
        x = self.linear1(state)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.linear2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        mu = T.tanh(self.mu(x))
        return mu
        
class Agent():
    def __init__(self, actor_lr, critic_lr, s_size, tau, a_size,
                 gamma = 0.99, max_size = 1000000, lyr1 = 32, lyr2 = 64,
                 batch_size = 64):
        
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        
        self.memory = ReplayBuffer(max_size, s_size, a_size)
        self.noise = OUActionNoise(mu=np.zeros(a_size))
        
        self.actor = ActorNet(actor_lr, s_size, lyr1, lyr2, a_size)
        self.critic = CriticNet(critic_lr, s_size, lyr1, lyr2, a_size)
        
        self.target_actor = ActorNet(actor_lr, s_size, lyr1, lyr2, a_size)
        self.target_critic = CriticNet(critic_lr, s_size, lyr1, lyr2, a_size)
        
        self.update_network_parameters(tau=1)
        
    def choose_action(self, obs):
        
        self.actor.eval()
        state = T.tensor([obs], dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(state).to(self.actor.device)
        
        mu_prime = mu + T.tensor(self.noise(), dtype = T.float).to(self.actor.device)
        self.actor.train()
        
        return mu_prime.cpu().detach().numpy()[0]
    
    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)
    
    def learn(self):
        
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, done = \
                self.memory.sample_buffer(self.batch_size)
        
        s = T.tensor(states, dtype=T.float).to(self.actor.device)
        s_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
        a = T.tensor(actions, dtype=T.float).to(self.actor.device)
        r = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        
        a_ = self.target_actor.forward(s_)
        Qs_a_ = self.target_critic.forward(s_, a_)
        Qsa = self.critic.forward(s, a)
        
        Qs_a_ = Qs_a_.view(-1)
        
        Qsa_bell = r + self.gamma*(1-done)*Qs_a_
        Qsa_bell = Qsa_bell.view(self.batch_size, 1)
        
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(Qsa_bell, Qsa)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(s, self.actor.forward(s))
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()
        
        self.update_network_parameters()
        
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                (1-tau)*target_critic_state_dict[name].clone()

        for name in actor_state_dict:
             actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                 (1-tau)*target_actor_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)
    
    def save_model(self, model_file):
        T.save(self.actor.state_dict(), model_file)
    
    def load_model(self, model_file):
        self.actor.load_state_dict(T.load(model_file))