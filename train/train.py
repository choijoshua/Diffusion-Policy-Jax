import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState


def create_train_state(args, rng, network, dummy_input, steps=None):
    lr = optax.cosine_decay_schedule(args.lr, steps or args.num_updates)
    return TrainState.create(
        apply_fn=network.apply,
        params=network.init(rng, *dummy_input),
        tx=optax.adam(lr, eps=1e-5),
    )


def make_train_step(args, dataset, forward_sample_fn, noise_pred_fn, mode='policy'):
    """
    Make JIT-compatible train step with dataset sampling.
    
    Args:
        args: Training arguments
        dataset: Dictionary with 'x' (actions/data) and optionally 'obs' (observations)
        forward_sample_fn: Function to add noise: (x0, t, noise) -> x_t
        noise_pred_fn: Network prediction function: (params, x_t, t, [obs]) -> noise_pred
        mode: 'policy' (conditional on observations) or 'score_matching' (unconditional)
    """
    
    # Preprocess dataset
    if mode == 'policy':
        data_x = jnp.array(dataset['x'])      # actions or trajectory data
        data_obs = jnp.array(dataset['obs'])  # observations/conditions
        dataset_size = data_x.shape[0]
    else:  # score_matching
        data_x = jnp.array(dataset['x'])
        dataset_size = data_x.shape[0]
    
    def _train_step(runner_state, unused):
        rng, train_state = runner_state
        
        # 1) Split RNG for all random operations
        rng, batch_rng, t_rng, noise_rng = jax.random.split(rng, 4)
        
        # 2) Sample batch indices and get batch data
        batch_indices = jax.random.randint(
            batch_rng, 
            (args.batch_size,), 
            0, 
            dataset_size
        )
        batch_x0 = data_x[batch_indices]
        
        if mode == 'policy':
            batch_obs = data_obs[batch_indices]
        
        # 3) Sample random timesteps for each element in batch
        t = jax.random.randint(t_rng, (args.batch_size,), 0, args.timesteps)
        
        # 4) Sample noise and create noisy samples x_t
        noise = jax.random.normal(noise_rng, batch_x0.shape)
        x_t = forward_sample_fn(batch_x0, t, noise)
        
        # 5) Define loss function
        def loss_fn(params):
            if mode == 'policy':
                # Diffusion policy: condition on observations
                eps_pred = noise_pred_fn({'params': params}, x_t, t, batch_obs)
            else:
                # Score matching: unconditional
                eps_pred = noise_pred_fn({'params': params}, x_t, t)
            
            # MSE loss on noise prediction
            loss = jnp.mean((noise - eps_pred) ** 2)
            return loss
        
        # 6) Compute gradients and update
        loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
        new_train_state = train_state.apply_gradients(grads=grads)
        
        # 7) Return new state and metrics
        new_runner_state = (rng, new_train_state)
        metrics = {'loss': loss}
        return new_runner_state, metrics
    
    # JIT compile for performance
    return jax.jit(_train_step)

