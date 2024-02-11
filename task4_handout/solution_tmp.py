import torch
import torch.optim as optim
from torch.distributions import Normal
import torch.nn as nn
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import warnings
from typing import Union
from utils import ReplayBuffer, get_env, run_episode

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class NeuralNetwork(nn.Module):
    '''
    This class implements a neural network with a variable number of hidden layers and hidden units.
    You may use this function to parametrize your policy and critic networks.
    '''
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int, 
                                hidden_layers: int, activation: str):
        super(NeuralNetwork, self).__init__()

        # TODO: Implement this function which should define a neural network 
        # with a variable number of hidden layers and hidden units.
        # Here you should define layers which your network will use.
        self.fc_layers = nn.ModuleList()
        
        self.fc_layers.append(nn.Linear(input_dim, hidden_size))
        self.fc_layers.append(activation)

        for _ in range(hidden_layers):
            self.fc_layers.append(nn.Linear(hidden_size, hidden_size))
            self.fc_layers.append(activation)

        self.fc_layers.append(nn.Linear(hidden_size, output_dim))

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass for the neural network you have defined.
        x = torch.tensor(s)
        for layer in self.fc_layers:
            x = layer(x)
        return x
    
class Actor:
    def __init__(self,hidden_size: int, hidden_layers: int, actor_lr: float,
                state_dim: int = 3, action_dim: int = 1, device: torch.device = torch.device('cpu')):
        super(Actor, self).__init__()

        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.actor_lr = actor_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.setup_actor()

    def setup_actor(self):
        '''
        This function sets up the actor network in the Actor class.
        '''
        # TODO: Implement this function which sets up the actor network. 
        # Take a look at the NeuralNetwork class in utils.py. 
        self.actor = NeuralNetwork(self.state_dim, 2, self.hidden_size, self.hidden_layers, nn.ReLU())
        self.optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)

    def clamp_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
        '''
        :param log_std: torch.Tensor, log_std of the policy.
        Returns:
        :param log_std: torch.Tensor, log_std of the policy clamped between LOG_STD_MIN and LOG_STD_MAX.
        '''
        return torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

    def get_action_and_log_prob(self, state: torch.Tensor, 
                                deterministic: bool) -> (torch.Tensor, torch.Tensor):
        '''
        :param state: torch.Tensor, state of the agent
        :param deterministic: boolean, if true return a deterministic action 
                                otherwise sample from the policy distribution.
        Returns:
        :param action: torch.Tensor, action the policy returns for the state.
        :param log_prob: log_probability of the the action.
        '''
        assert state.shape == (3,) or state.shape[1] == self.state_dim, 'State passed to this method has a wrong shape'
        action , log_prob = torch.zeros(state.shape[0]), torch.ones(state.shape[0])
        # TODO: Implement this function which returns an action and its log probability.
        # If working with stochastic policies, make sure that its log_std are clamped 
        # using the clamp_log_std function.
        out = self.actor(state)
        if not state.shape == (3,) and state.shape[0] > 1:
            mu, sigma = out[:, 0], out[:, 1]
        else :
            mu, sigma = out[0], out[1]
        sigma = self.clamp_log_std(sigma)

        sigma = torch.exp(sigma)
        # create normal distribution
        act_dist = Normal(mu, sigma)
        # sample actions
        action = act_dist.rsample()
        log_prob = act_dist.log_prob(action) - torch.log(1 - torch.tanh(action)**2 + 1e-6)

        if deterministic:
            action = mu
            print(mu, "MU DEBUG")
            print(sigma, "SIGMA DEBUG")
        # else: 
        #     print(mu, "MU DEBUG For Testing")
        # calculate log prob
        if not state.shape == (3,) and state.shape[0] > 1:
           action = torch.tanh(action).reshape(-1, 1)
           log_prob = log_prob.reshape(-1, 1)
        else :
            action = torch.tanh(action).unsqueeze(0)
            log_prob = log_prob.unsqueeze(0)
        #print((action.shape[0],), (self.action_dim,), (log_prob.shape[0],), (self.action_dim,), "SHAPE DEBUG")
        assert (action.shape[0],) == (self.action_dim,) and (log_prob.shape[0],) == (self.action_dim,) or\
            (action.shape[0], 1) == (state.shape[0], 1) and (log_prob.shape[0], 1) == (state.shape[0], 1), 'Incorrect shape for action or log_prob.'

        return action, log_prob
    
class Critic:
    def __init__(self, hidden_size: int, 
                 hidden_layers: int, critic_lr: int, state_dim: int = 3, 
                    action_dim: int = 1,device: torch.device = torch.device('cpu')):
        super(Critic, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.critic_lr = critic_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.setup_critic()

    def setup_critic(self):
        # TODO: Implement this function which sets up the critic(s). Take a look at the NeuralNetwork 
        # class in utils.py. Note that you can have MULTIPLE critic networks in this class.
        self.critic = NeuralNetwork(self.state_dim + self.action_dim, 1, self.hidden_size, self.hidden_layers, nn.ReLU())
        self.optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr) 

class TrainableParameter:
    '''
    This class could be used to define a trainable parameter in your method. You could find it 
    useful if you try to implement the entropy temerature parameter for SAC algorithm.
    '''
    def __init__(self, init_param: float, lr_param: float, 
                 train_param: bool, device: torch.device = torch.device('cpu')):
        
        self.log_param = torch.tensor(np.log(init_param), requires_grad=train_param, device=device)
        self.optimizer = optim.Adam([self.log_param], lr=lr_param)

    def get_param(self) -> torch.Tensor:
        return torch.exp(self.log_param)

    def get_log_param(self) -> torch.Tensor:
        return self.log_param

class Agent:
    def __init__(self):
        # Environment variables. You don't need to change this.
        self.state_dim = 3  # [cos(theta), sin(theta), theta_dot]
        self.action_dim = 1  # [torque] in[-1,1]
        self.batch_size = 200
        self.min_buffer_size = 1000
        self.max_buffer_size = 100000
        # If your PC possesses a GPU, you should be able to use it for training, 
        # as self.device should be 'cuda' in that case.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(self.device))
        self.memory = ReplayBuffer(self.min_buffer_size, self.max_buffer_size, self.device)
        
        self.setup_agent()

    def setup_agent(self):
        # TODO: Setup off-policy agent with policy and critic classes. 
        # Feel free to instantiate any other parameters you feel you might need.   
        policy_hidden_size = 256
        critic_hidden_size = 256
        policy_hidden_layers = 2
        critic_hidden_layers = 2 
        policy_lr = 0.03
        critic_lr = 0.03
        temp_init = 1 # TODO check if this is correct
        temp_lr = 0.0003
        self.gamma = 0.99 # discount factor
        self.target_q1 = NeuralNetwork(self.state_dim + self.action_dim, 1, critic_hidden_size, critic_hidden_layers, nn.ReLU()) 
        self.target_q2 = NeuralNetwork(self.state_dim + self.action_dim, 1, critic_hidden_size, critic_hidden_layers, nn.ReLU())
        self.tau = 0.05 # update target network weights
                 
        self.actor = Actor(policy_hidden_size, policy_hidden_layers, policy_lr, self.state_dim, self.action_dim, self.device)
        self.critic1 = Critic(critic_hidden_size, critic_hidden_layers, critic_lr, self.state_dim, self.action_dim, self.device)
        self.critic2 = Critic(critic_hidden_size, critic_hidden_layers, critic_lr, self.state_dim, self.action_dim, self.device)
        self.alpha = TrainableParameter(temp_init, temp_lr, train_param=True, device=self.device)
        self.target_q1.load_state_dict(self.critic1.critic.state_dict())
        self.target_q2.load_state_dict(self.critic2.critic.state_dict())
        for param in self.target_q1.parameters():
            param.requires_grad = False
        for param in self.target_q2.parameters():
            param.requires_grad = False

    def get_action(self, s: np.ndarray, train: bool) -> np.ndarray:
        """
        :param s: np.ndarray, state of the pendulum. shape (3, )
        :param train: boolean to indicate if you are in eval or train mode. 
                    You can find it useful if you want to sample from deterministic policy.
        :return: np.ndarray,, action to apply on the environment, shape (1,)
        """
        # TODO: Implement a function that returns an action from the policy for the state s.
        action = np.random.uniform(-1, 1, (1,))
        # if train true => not deterministic => what we need for training
        action, _ = self.actor.get_action_and_log_prob(s, deterministic= not train)
        action = action.detach().numpy()
        assert action.shape == (1,), 'Incorrect action shape.'
        assert isinstance(action, np.ndarray), 'Action dtype must be np.ndarray' 
        return action

    @staticmethod 
    def run_gradient_update_step(object: Union[Actor, Critic], loss: torch.Tensor):
        '''
        This function takes in a object containing trainable parameters and an optimizer, 
        and using a given loss, runs one step of gradient update. If you set up trainable parameters 
        and optimizer inside the object, you could find this function useful while training.
        :param object: object containing trainable parameters and an optimizer
        '''
        object.optimizer.zero_grad()
        loss.mean().backward()
        object.optimizer.step()

    def critic_target_update(self, base_net: NeuralNetwork, target_net: NeuralNetwork, 
                             tau: float, soft_update: bool):
        '''
        This method updates the target network parameters using the source network parameters.
        If soft_update is True, then perform a soft update, otherwise a hard update (copy).
        :param base_net: source network
        :param target_net: target network
        :param tau: soft update parameter
        :param soft_update: boolean to indicate whether to perform a soft update or not
        '''
        for param_target, param in zip(target_net.parameters(), base_net.parameters()):
            if soft_update:
                param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
            else:
                param_target.data.copy_(param.data)

    def train_agent(self):
        '''
        This function represents one training iteration for the agent. It samples a batch 
        from the replay buffer,and then updates the policy and critic networks 
        using the sampled batch.
        '''
        # TODO: Implement one step of training for the agent.
        # Hint: You can use the run_gradient_update_step for each policy and critic.
        # Example: self.run_gradient_update_step(self.policy, policy_loss)

        # Batch sampling
        batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, s_prime_batch = batch

        # L1: critic optim
        # TODO: Implement Critic(s) update here.
        loss = nn.MSELoss()
        
        with torch.no_grad():
           a_prime_batch, log_prob_batch = self.actor.get_action_and_log_prob(s_prime_batch, deterministic=False)
           q_val_1 = self.critic1.critic(torch.cat([s_prime_batch, a_prime_batch], dim=1))
           q_val_2 = self.critic2.critic(torch.cat([s_prime_batch, a_prime_batch], dim=1))
           min_qf_next_target = torch.min(q_val_1, q_val_2) - self.alpha.get_param() * log_prob_batch
           next_q_value = r_batch + self.gamma * (min_qf_next_target)
        #print(self.alpha.get_param(), "self.alpha.get_param()")
        loss_Q1 = loss(self.critic1.critic(torch.cat([s_batch, a_batch], dim=1)), next_q_value)
        loss_Q2 = loss(self.critic2.critic(torch.cat([s_batch, a_batch], dim=1)), next_q_value)
        self.run_gradient_update_step(self.critic1, loss_Q1)
        for param in self.critic1.critic.parameters():
            param.requires_grad = False
        self.run_gradient_update_step(self.critic2, loss_Q2)
        for param in self.critic2.critic.parameters():
            param.requires_grad = False
        #print("Critic 1 Gradients:")
        #for param in self.critic1.critic.parameters():
            #print(param.grad)
        #print("Critic 2 Gradients:")
        #for param in self.critic2.critic.parameters():
            #print(param.grad)
        
        # L2: actor optim
        # TODO: Implement Policy update here
        actions, log_p_batch = self.actor.get_action_and_log_prob(s_batch, deterministic=False)
        for param in self.critic2.critic.parameters():
            param.requires_grad = False
        # actions = torch.tensor(actions)
        # log_p_batch = torch.tensor(log_p_batch)
        q1 = self.critic1.critic(torch.cat([s_batch, actions], dim=1))
        q2 = self.critic2.critic(torch.cat([s_batch, actions], dim=1))
        loss_actor = (torch.min(q1, q2) - self.alpha.get_param()*log_p_batch).mean(dim=0)
        self.run_gradient_update_step(self.actor, loss_actor)
        print("Actor Gradients:")
        for param in self.actor.actor.parameters():
            print(param.grad)

        # L3: temperature optim
        H_hat = -self.action_dim
        loss_temp = (-self.alpha.get_param()*(log_p_batch + H_hat)).mean(dim=0)
        self.run_gradient_update_step(self.alpha, loss_temp)

        # Update target network weights 
        self.critic_target_update(self.critic1.critic, self.target_q1, self.tau, soft_update=True)
        self.critic_target_update(self.critic2.critic, self.target_q2, self.tau, soft_update=True)

# This main function is provided here to enable some basic testing. 
# ANY changes here WON'T take any effect while grading.
if __name__ == '__main__':

    TRAIN_EPISODES = 10
    TEST_EPISODES = 300

    # You may set the save_video param to output the video of one of the evalution episodes, or 
    # you can disable console printing during training and testing by setting verbose to False.
    save_video = True
    verbose = True

    agent = Agent()
    env = get_env(g=10.0, train=True)

    for EP in range(TRAIN_EPISODES):
        run_episode(env, agent, None, verbose, train=True)

    if verbose:
        print('\n')

    test_returns = []
    env = get_env(g=10.0, train=False)

    if save_video:
        video_rec = VideoRecorder(env, "pendulum_episode.mp4")
    
    for EP in range(TEST_EPISODES):
        rec = video_rec if (save_video and EP == TEST_EPISODES - 1) else None
        with torch.no_grad():
            episode_return = run_episode(env, agent, rec, verbose, train=False)
        test_returns.append(episode_return)

    avg_test_return = np.mean(np.array(test_returns))

    print("\n AVG_TEST_RETURN:{:.1f} \n".format(avg_test_return))

    if save_video:
        video_rec.close()
