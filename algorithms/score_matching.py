import jax
import jax.numpy as jnp
import flax.linen as nn
import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from config import TrainArgs


class ScoreMatchingPolicy:
    def __init__(self, args: TrainArgs, action_dim: int, model_apply_fn=None):
        self.args = args
        self.timesteps = args.timesteps
        self.action_dim = action_dim
        self.model_apply_fn = model_apply_fn
        
    def sde_gt(self, t):
        '''
        SDE in the form of: dx = f(x, t) dt + g(t) dw
        returns g(t) = simga**t
        '''
        return self.args.sigma ** t
        
    def forward_sde_variance(self, t):
        '''
        SDE in the form of: dx = f(x, t) dt + g(t) dw
        x(t) = x(0) + integral [0, t] g(t)^2 dw
        '''

        return jnp.sqrt((self.args.sigma**(2 * t) - 1.) / 2. / jnp.log(self.args.sigma))
        
    def forward_sde_sample(self, x0, t, noise):
        '''
        Sample from x(t) = N(x(0), var)
        '''
        var = self.forward_sde_variance(t)
        return x0 + var[:, None, None]*noise
    
    def euler_maruyama_sampler(self, rng, init_x, obs, params, num_steps, eps=1e-4):
        '''Generate samples from score-based models with the Euler-Maruyama solver.

        Args:
            init_x: initial x sampled from random distribution
            obs: observation condition
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
        for time_step in time_steps:
            batch_time_step = jnp.ones(self.args.batch_size) * time_step
            g = self.sde_gt(time_step)
            mean_x = x + (g**2) * self.predict(params, x, batch_time_step, obs) * step_size
            rng, step_rng = jax.random.split(rng)
            x = mean_x + jnp.sqrt(step_size) * g * jax.random.normal(step_rng, x.shape)      
        # Do not include any noise in the last sampling step.
        return mean_x
    
    def predictor_corrector_sampler(self, rng, init_x, obs, params, num_steps, eps=1e-5, snr=0.16):
        '''Generate samples from score-based models with the Predictor Corrector solver.

        Args:
            rng: 
            init_x: initial x sampled from random distribution
            obs: observation condition
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
            grad = self.predict(params, x, batch_time_step, obs)    
            grad_norm = jnp.linalg.norm(grad.reshape(x.shape[0], x.shape[1], -1),
                                        axis=-1).mean()
            noise_norm = jnp.sqrt(jnp.prod(jnp.array(x.shape[1:])))  # Fixed: wrap in jnp.array
            langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
            rng, step_rng = jax.random.split(rng)
            z = jax.random.normal(step_rng, x.shape)
            x = x + langevin_step_size * grad + jnp.sqrt(2 * langevin_step_size) * z 

            # Predictor step (Euler-Maruyama)
            g = self.sde_gt(time_step)
            score = self.predict(params, x, batch_time_step, obs)
            x_mean = x + (g**2) * score * step_size
            rng, step_rng = jax.random.split(rng)
            z = jax.random.normal(step_rng, x.shape)
            x = x_mean + jnp.sqrt(g**2 * step_size) * z  
        
        # The last step does not include any noise
        return x_mean
    
    def forward_sample(self, x0, t, noise):
        """ 
        keep function name identical with other methods for forward diffusion
        
        """
        return self.forward_sde_sample(x0, t, noise)
    
    def sample(self, rng, train_state, xt, obs):
        ''' Sample from target distribution based on which sampler to use
        
        Sampler Options:
            Euler Maruyama Sampler
            Predictor Corrector Sampler
            
        Return: sample from target distribution p(x)
        
        '''
        params = train_state.params
        if self.args.sampler == "euler_maruyama":
            x_0 = self.euler_maruyama_sampler(rng, xt, obs, params, self.timesteps, self.args.eps)
        elif self.args.sampler == "predictor_corrector":
            x_0 = self.predictor_corrector_sampler(rng, xt, obs, params, self.timesteps, self.args.eps)
        else:
            print("PICK AN EXISTING SAMPLER: \n EULER MARUYAMA \n PREDICTOR CORRECTOR")    
        
        return x_0
    
    def predict(self, params, x, t, obs):
        return self.model_apply_fn(params, x, t, obs)
    
    def get_action(self, rng, train_state, obs):
        rng, rng_sample = jax.random.split(rng)
        xt = jax.random.normal(rng_sample, shape=(self.args.batch_size, self.args.horizon, self.action_dim))
        trajectory_pred = self.sample(rng, train_state, xt, obs)
        return trajectory_pred[:, 0, :]

