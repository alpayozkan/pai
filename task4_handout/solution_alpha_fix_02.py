import torch
import torch.optim as optim
from torch.distributions import Normal
import torch.nn as nn
import torch.nn.functional as F
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

        layers = []
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(activation)

        for i in range(hidden_layers-1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation)

        layers.append(nn.Linear(hidden_size, output_dim))

        self.net = nn.Sequential(*layers)
        
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass for the neural network you have defined.
        return self.net(s)
    
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
        activation = nn.ReLU()

        self.p = NeuralNetwork(self.state_dim, 2, self.hidden_size, self.hidden_layers, activation)
        self.p_optimizer = optim.Adam(self.p.parameters(), lr=self.actor_lr)

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
        assert tuple(state.shape) == (3,) or state.shape[1] == self.state_dim, 'State passed to this method has a wrong shape'
        action , log_prob = torch.zeros(state.shape[0]), torch.ones(state.shape[0])
        # TODO: Implement this function which returns an action and its log probability.
        # If working with stochastic policies, make sure that its log_std are clamped 
        # using the clamp_log_std function.
        
        out = self.p(state)
        mu, std = out[:,0], out[:,1] # (n,), (n,)
        # unsqueeze to obtain the same dimensionality
        mu = mu.unsqueeze(1)
        std = std.unsqueeze(1)

        std = self.clamp_log_std(std) # clamp log_std => numerical

        std = torch.exp(std)
        ndists = Normal(mu, std)
        u = ndists.rsample() # (n,) : rsample => backprop, sample => no backprop
        action = torch.tanh(u) # squash : [-1,+1]

        if deterministic:
            u = mu
            action = torch.tanh(u)
        # appendix c: enforcing action bounds: jacobian
        log_prob = ndists.log_prob(u) - torch.log(1-torch.tanh(u)**2 + 1e-6) 
        # torch.sum dim=1, broadcasting issue if action space>1

        assert ((tuple(action.shape) == (self.action_dim,)) and \
            tuple(log_prob.shape) == (self.action_dim, )) or (tuple(action.shape) == (state.shape[0], 1) and \
                                                        tuple(log_prob.shape) == (state.shape[0],1)), 'Incorrect shape for action or log_prob.'
        # assert action.shape == (state.shape[0], self.action_dim) and \
        #     log_prob.shape == (state.shape[0], self.action_dim), 'Incorrect shape for action or log_prob.'
        
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
        activation = nn.ReLU()
        input_dim = self.state_dim + self.action_dim
        # Q(s,a): (s,a) => R^1
        self.q1 = NeuralNetwork(input_dim, 1, self.hidden_size, self.hidden_layers, activation)
        self.q2 = NeuralNetwork(input_dim, 1, self.hidden_size, self.hidden_layers, activation)

        # target critics
        self.q1_trg = NeuralNetwork(input_dim, 1, self.hidden_size, self.hidden_layers, activation)
        self.q2_trg = NeuralNetwork(input_dim, 1, self.hidden_size, self.hidden_layers, activation)
        # copy weights from q-networks
        self.q1_trg.load_state_dict(self.q1.state_dict())
        self.q2_trg.load_state_dict(self.q2.state_dict())
        # no gradient calculation for target critics
        self.q1_trg.eval()
        self.q2_trg.eval()
        for param in self.q1_trg.parameters():
            param.requires_grad = False
        for param in self.q2_trg.parameters():
            param.requires_grad = False


        # Optimizers
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=self.critic_lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=self.critic_lr)

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
        self.tau = 0.005
        
        hidden_size = 256
        hidden_layers = 2
        self.gamma = 0.99

        critic_lr = 3e-4
        self.critic = Critic(hidden_size, hidden_layers, critic_lr, self.state_dim, self.action_dim, self.device)

        actor_lr = 3e-4
        self.actor = Actor(hidden_size, hidden_layers, actor_lr, self.state_dim, self.action_dim, self.device)
        
        temp_lr = 3e-4
        init_param = 1
        self.temp = TrainableParameter(init_param, temp_lr, True, self.device)

    def get_action(self, s: np.ndarray, train: bool) -> np.ndarray:
        """
        :param s: np.ndarray, state of the pendulum. shape (3, )
        :param train: boolean to indicate if you are in eval or train mode. 
                    You can find it useful if you want to sample from deterministic policy.
        :return: np.ndarray,, action to apply on the environment, shape (1,)
        """
        # TODO: Implement a function that returns an action from the policy for the state s.
        action = np.random.uniform(-1, 1, (1,))

        state = torch.tensor(s).unsqueeze(0)
        action, _ = self.actor.get_action_and_log_prob(state, not train) # if train=True => non-det => while training, setup=>False while training
        action = action.squeeze(0)
        action = action.detach().numpy() # convert to numpy, shouldnt follow grads to simulator

        assert action.shape == (1,), 'Incorrect action shape.'
        assert isinstance(action, np.ndarray ), 'Action dtype must be np.ndarray' 
        return action

    @staticmethod
    def run_gradient_update_step(optimizer, loss: torch.Tensor):
        '''
        This function takes in a object containing trainable parameters and an optimizer, 
        and using a given loss, runs one step of gradient update. If you set up trainable parameters 
        and optimizer inside the object, you could find this function useful while training.
        :param object: object containing trainable parameters and an optimizer
        '''
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

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
        
        # TODO: Make Learnable
        alpha = 0.2
        # alpha = self.temp.get_param().detach()

        # calculate targets for the q function
        with torch.no_grad():
            act_prime, logp_prime = self.actor.get_action_and_log_prob(s_prime_batch, False)

            sact_prime_batch = torch.cat([s_prime_batch, act_prime], dim=1)
            q1_prime = self.critic.q1_trg(sact_prime_batch)
            q2_prime = self.critic.q2_trg(sact_prime_batch)
            
            q1q2_prime = torch.cat([q1_prime, q2_prime], dim=1)
            q, _ = torch.min(q1q2_prime, dim=1) # take min => stated in the paper
            q = q.unsqueeze(1)
            # alpha = self.temp.get_param() # backprop avoided since alpha also learned
            y_label = r_batch + self.gamma*(q - alpha*logp_prime)
            
        # TODO: Implement Critic(s) update here.
        sact_batch = torch.cat([s_batch, a_batch], dim=1)
        loss_q1 = F.mse_loss(self.critic.q1(sact_batch), y_label)
        loss_q2 = F.mse_loss(self.critic.q2(sact_batch), y_label)

        self.run_gradient_update_step(self.critic.q1_optimizer, loss_q1)
        self.run_gradient_update_step(self.critic.q2_optimizer, loss_q2)

        # TODO: Implement Policy update here
        act_hat, logp_hat = self.actor.get_action_and_log_prob(s_batch, False)
        
        sacthat_batch = torch.cat([s_batch, act_hat], dim=1)
        q1 = self.critic.q1(sacthat_batch)
        q2 = self.critic.q2(sacthat_batch)
        
        q1q2 = torch.cat([q1, q2], dim=1)
        q, _ = torch.min(q1q2, dim=1)
        q = q.unsqueeze(1)
        loss_p = (alpha*logp_hat - q).mean()
        self.run_gradient_update_step(self.actor.p_optimizer, loss_p)

        # Alpha: temperature update
        # alpha = self.temp.get_param()
        
        # loss_alpha = -self.temp.get_param()*((logp_hat - self.action_dim).detach()) # dont backrop to policy
        # loss_alpha = loss_alpha.mean()

        # self.run_gradient_update_step(self.temp.optimizer, loss_alpha)
        # self.run_gradient_update_step(self.temp.optimizer, loss_alpha)

        # Critic Network Update
        self.critic_target_update(self.critic.q1, self.critic.q1_trg, self.tau, True)
        self.critic_target_update(self.critic.q2, self.critic.q2_trg, self.tau, True)


# This main function is provided here to enable some basic testing. 
# ANY changes here WON'T take any effect while grading.
if __name__ == '__main__':

    TRAIN_EPISODES = 50
    TEST_EPISODES = 300

    # You may set the save_video param to output the video of one of the evalution episodes, or 
    # you can disable console printing during training and testing by setting verbose to False.
    save_video = False
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
