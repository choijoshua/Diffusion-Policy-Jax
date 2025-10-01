import jax
import jax.numpy as jnp
import flax.linen as nn
import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from config import TrainArgs
from model.utils import cosine_beta_schedule
from flax.training.train_state import TrainState


class DDPMPolicy:
    def __init__(self, args: TrainArgs, action_dim: int, model_apply_fn=None):
        self.args = args
        self.timesteps = self.args.num_timesteps
        self.action_dim = action_dim
        self.beta = cosine_beta_schedule(timesteps=self.timesteps)
        self.alpha = 1 - self.beta
        self.alpha_bar = jnp.cumprod(self.alpha)
        self.model_apply_fn = model_apply_fn
        
    def q_mean_variance(self, x0, t):
        '''
        q(x_t | x_0) = N(mean, var)
        '''

        # Get alpha_bar values for each timestep
        alpha_bar_t = self.alpha_bar[t]  # (B,)

        if len(x0.shape) == 2:
            # Single-step: (B, D)
            alpha_bar_t = alpha_bar_t[:, None]  # (B, 1)
        elif len(x0.shape) == 3:
            # Trajectory: (B, H, D)
            alpha_bar_t = alpha_bar_t[:, None, None]  # (B, 1, 1)
        else:
            raise ValueError(f"Unsupported x0 shape: {x0.shape}")
        
        mean = jnp.sqrt(alpha_bar_t) * x0 
        var = jnp.sqrt(1 - self.alpha_bar[t])
        return mean, var

    def q_sample(self, x0, t, noise):
        '''
        Sample from q(x_t | x_0)
        '''
        assert noise.shape == x0.shape
        mean, var = self.q_mean_variance(x0, t)

        return mean + var[:, None, None] * noise
    
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
        return mean + var[:, None, None] * noise
    
    def forward_sample(self, x0, t, noise):
        """ 
        keep function name identical with other methods for forward diffusion

        """
        return self.q_sample(x0, t, noise)
    
    def sample(self, rng, train_state, xt, obs):
        params = train_state.params
        for t in reversed(range(1, self.timesteps)):
            rng, epsilon_rng = jax.random.split(rng)
            t = jnp.full((xt.shape[0],), t)
            epsilon = self.predict(params, xt, t, obs)
            noise = jax.random.normal(key=epsilon_rng, shape=xt.shape)
            xt = self.p_sample(xt, t, epsilon, noise)
        
        t = jnp.full((xt.shape[0],), 0)
        noise = jnp.zeros_like(xt)
        xt = self.p_sample(xt, t, epsilon, noise)
        return xt
    
    def predict(self, params, x, t, obs):
        return self.model_apply_fn(params, x, t, obs)
    
    def get_action(self, rng, train_state, obs):
        if obs.ndim == 1:
            obs = obs[None, :]  # (obs_dim,) -> (1, obs_dim)
            single_obs = True
        else:
            single_obs = False

        rng, rng_sample = jax.random.split(rng)
        batch_size = obs.shape[0]
        # Start from noise
        xt = jax.random.normal(
            rng_sample, 
            shape=(batch_size, self.args.horizon, self.action_dim)
        )
        trajectory_pred = self.sample(rng, train_state, xt, obs)
        
        action =  trajectory_pred[:, 0, :]
        if single_obs:
            return action[0]  # Return single action
        
        return action
    
