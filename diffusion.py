import jax
import jax.numpy as jnp
import flax.linen as nn
from config import Args
from model import UNet
from util import cosine_beta_schedule



class DiffusionPolicy:
    def __init__(self, args: Args, model: UNet, timesteps: int):
        self.args = args
        self.model = model
        self.timesteps = timesteps
        self.beta = cosine_beta_schedule(timesteps=timesteps)
        self.alpha = 1 - self.beta
        self.alpha_bar = jnp.cumprod(self.alpha)
        self.rng = jax.random.PRNGKey(seed=self.args.seed)
        
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
        Sample from q(x_{t-1} | x_t, x_0)
        '''
        assert noise.shape == xt.shape
        mean, var = self.p_mean_variance(xt, t, epsilon_pred)
        return mean + var * noise
    
    def sample(self, xt, agent_state, obs):
        params = agent_state.params
        for t in reversed(range(self.timesteps)):
            rng, epsilon_rng = jax.random.split(self.rng)
            epsilon = self.model.apply(params, xt, t, obs)
            noise = jax.random.normal(key=epsilon_rng, shape=xt.shape)
            if t == 0:
                noise = jnp.zeros_like(xt)
            xt = self.p_sample(xt, t, epsilon, noise)
        
        return xt
    
    def get_action(self, xt, agent_state, obs):
        trajectory_pred = self.sample(xt, agent_state, obs)
        return trajectory_pred[:, 0, :]


rng = jax.random.PRNGKey(seed=42)
args = Args()
model = UNet(args, 30, 17)
policy = DiffusionPolicy(args, model, 100)

x0 = jax.random.uniform(key=rng, shape=(32, 50, 17))
t = jnp.array([5] * 32)
print(policy.sample(x0))