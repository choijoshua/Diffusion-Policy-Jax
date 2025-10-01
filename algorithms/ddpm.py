import jax
import jax.numpy as jnp
import flax.linen as nn
import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from config import TrainArgs
from model.util import cosine_beta_schedule
from flax.training.train_state import TrainState


class DDPMPolicy:
    def __init__(self, args: TrainArgs, model: TrainState):
        self.args = args
        self.model = model
        self.timesteps = self.args.num_timesteps
        self.beta = cosine_beta_schedule(timesteps=self.timesteps)
        self.alpha = 1 - self.beta
        self.alpha_bar = jnp.cumprod(self.alpha)
        
    def q_mean_variance(self, x0, t):
        '''
        q(x_t | x_0) = N(mean, var)
        '''
        mean = jnp.sqrt(self.alpha_bar[t]) * x0 
        var = jnp.sqrt(1 - self.alpha_bar[t])
        return mean, var

    def q_sample(self, x0, t, noise):
        '''
        Sample from q(x_t | x_0)
        '''
        assert noise.shape == x0.shape
        mean, var = self.q_mean_variance(x0, t)

        return mean + var * noise
    
    def p_mean_variance(self, xt, t, epsilon_pred):
        '''
        p(x_{t-1} | x_t) = N(mean, var)
        x_{t-1}
        mean = 1/sqrt(a_t) * (x_t - b_t/sqrt(1-a_bar_t) * epsilon_pred)
        var = sqrt(b_t)
        '''
        mean = 1/jnp.sqrt(self.alpha[t]) * (xt - self.beta[t]/jnp.sqrt(1 - self.alpha_bar[t])*epsilon_pred) 
        var = jnp.sqrt(self.beta[t])
        return mean, var

    def p_sample(self, xt, t, epsilon_pred, noise):
        '''
        Sample from q(x_{t-1} | x_t)
        '''
        assert noise.shape == xt.shape
        mean, var = self.p_mean_variance(xt, t, epsilon_pred)
        return mean + var * noise
    
    def forward_sample(self, x0, t, noise):
        """ 
        keep function name identical with other methods for forward diffusion

        """
        return self.q_sample(x0, t, noise)
    
    def sample(self, rng, xt, agent_state, obs):
        params = agent_state.params
        for t in reversed(range(self.timesteps)):
            rng, epsilon_rng = jax.random.split(rng)
            epsilon = self.model.apply_fn(params, xt, t, obs)
            noise = jax.random.normal(key=epsilon_rng, shape=xt.shape)
            if t == 0:
                noise = jnp.zeros_like(xt)
            xt = self.p_sample(xt, t, epsilon, noise)
        
        return xt
    
    def predict(self, params, xt, t, obs):
        return self.model.apply_fn(params, xt, t, obs)
    
    def get_action(self, rng, xt, agent_state, obs):
        trajectory_pred = self.sample(xt, rng, agent_state, obs)
        return trajectory_pred[:, 0, :]
    
