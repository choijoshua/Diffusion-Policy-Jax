import jax
import jax.numpy as jnp
import flax.linen as nn
import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from config import Args
from model.unet import UNet



class ScoreMatchingPolicy:
    def __init__(self, args: Args, model: UNet, timesteps: int):
        self.args = args
        self.model = model
        self.rng = jax.random.PRNGKey(seed=self.args.seed)
        
    def sde_gt(self, t):
        '''
        SDE in the form of: dx = f(x, t) dt + g(t) dw
        returns g(t) = simga**t
        '''
        return self.args.sigma ** t
        
    def forward_sde_variance(self, t, sigma, num_steps, eps):
        '''
        SDE in the form of: dx = f(x, t) dt + g(t) dw
        x(t) = x(0) + integral [0, t] g(t)^2 dw
        '''
        
        return jnp.sqrt((sigma**(2 * t) - 1.) / 2. / jnp.log(sigma))
        
    def forward_sde_sample(self, x0, t, sigma, noise):
        '''
        Sample from x(t) = N(x(0), var)
        '''
        var = self.forward_sde_variance(t, sigma)
        return x0 + var*noise
    
    def euler_maruyama_sampler(self, rng, init_x, params, num_steps, eps):
        '''Generate samples from score-based models with the Euler-Maruyama solver.

        Args:
            init_x: initial x sampled from random distribution
            params: A dictionary that contains the model parameters.
            diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
            num_steps: The number of sampling steps. 
            Equivalent to the number of discretized time steps.    
            eps: The smallest time step for numerical stability.
            Reverse SDE:
                dx = [f(x, t) - g(t)^2 * s(x, t)] dt + g(t) dw
        Returns:
            Solve the SDE
            
        '''  
        x = init_x
        time_steps = jnp.linspace(1., eps, num_steps)
        step_size = time_steps[0] - time_steps[1]
        for time_step in range(time_steps):
            batch_time_step = jnp.ones(self.args.batch_size) * time_step
            g = self.sde_gt(time_step)
            mean_x = x + (g**2) * self.model.apply_fn(
                                                params,
                                                x, 
                                                batch_time_step) * step_size
            rng, step_rng = jax.random.split(rng)
            x = mean_x + jnp.sqrt(step_size) * g * jax.random.normal(step_rng, x.shape)      
        # Do not include any noise in the last sampling step.
        return mean_x
    
    def predictor_corrector_sampler(self, rng, init_x, params, num_steps, eps, snr=0.16):
        '''Generate samples from score-based models with the Predictor Corrector solver.

        Args:
            init_x: initial x sampled from random distribution
            params: A dictionary that contains the model parameters.
            num_steps: The number of sampling steps. 
            Equivalent to the number of discretized time steps.    
            eps: The smallest time step for numerical stability.
            snr: Signal-to-noise ratio for Langevin dynamics (default: 0.16)
            Reverse SDE:
                dx = [f(x, t) - g(t)^2 * s(x, t)] dt + g(t) dw
        Returns:
            Samples.
            
        '''  
        rng, step_rng = jax.random.split(rng)
        time_steps = jnp.linspace(1., eps, num_steps)
        step_size = time_steps[0] - time_steps[1]
        x = init_x  
        
        for time_step in time_steps:  # Fixed: iterate over array, not range()
            batch_time_step = jnp.ones(self.args.batch_size) * time_step
            
            # Corrector step (Langevin MCMC)
            grad = self.model.apply_fn(params, x, batch_time_step)    
            grad_norm = jnp.linalg.norm(grad.reshape(x.shape[0], x.shape[1], -1),
                                        axis=-1).mean()
            noise_norm = jnp.sqrt(jnp.prod(jnp.array(x.shape[1:])))  # Fixed: wrap in jnp.array
            langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
            rng, step_rng = jax.random.split(rng)
            z = jax.random.normal(step_rng, x.shape)
            x = x + langevin_step_size * grad + jnp.sqrt(2 * langevin_step_size) * z 

            # Predictor step (Euler-Maruyama)
            g = self.sde_gt(time_step)
            score = self.model.apply_fn(params, x, batch_time_step)
            x_mean = x + (g**2) * score * step_size
            rng, step_rng = jax.random.split(rng)
            z = jax.random.normal(step_rng, x.shape)
            x = x_mean + jnp.sqrt(g**2 * step_size) * z  
        
        # The last step does not include any noise
        return x_mean
    
    def sample(self, xt, agent_state):
        ''' Sample from target distribution based on which sampler to use
        
        Sampler Options:
            Euler Maruyama Sampler
            Predictor Corrector Sampler
            
        Return: sample from target distribution p(x)
        
        '''
        params = agent_state.params
        if self.args.sampler == "euler_maruyama":
            x_0 = self.euler_maruyama_sampler(rng, xt, params, t, self.args.sampler_timestep)
        elif self.args.sampler == "predictor_corrector":
            x_0 = self.predictor_corrector_sampler(rng, xt, params, t, self.args.sampler_timestep)
        else:
            print("PICK AN EXISTING SAMPLER: \n EULER MARUYAMA \n PREDICTOR CORRECTOR")    
        
        return x_0
    
    def get_action(self, xt, agent_state, obs):
        trajectory_pred = self.sample(xt, agent_state, obs)
        return trajectory_pred[:, 0, :]


rng = jax.random.PRNGKey(seed=42)
args = Args()
model = UNet(args, 30, 17)
policy = ScoreMatchingPolicy(args, model, 100)

x0 = jax.random.uniform(key=rng, shape=(32, 50, 17))
t = jnp.array([5] * 32)
print(policy.sample(x0, ))