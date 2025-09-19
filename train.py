import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.linen.initializers import constant, uniform
from flax.training.train_state import TrainState


def create_train_state(args, rng, network, dummy_input, steps=None):

    lr = optax.cosine_decay_schedule(lr, steps or args.num_updates)
    return TrainState.create(
        apply_fn=network.apply,
        params=network.init(rng, *dummy_input),
        tx=optax.adam(lr, eps=1e-5),
    )

def make_train_step(self, forward_sample_fn, noise_pred_fn):
    """Make JIT-compatible agent train step, with optional model-based rollouts."""

    def _train_step(runner_state, batch_x0):
        rng, train_state, buffer = runner_state

        # 1) split RNG for everything
        rng, t_rng, noise_rng = jax.random.split(rng, 3)

        # 2) sample a random timestep for each element in the batch
        B = batch_x0.shape[0]
        t = jax.random.randint(t_rng, (B,), 0, self.timesteps)

        # 3) sample the forward noise and build x_t
        noise = jax.random.normal(noise_rng, batch_x0.shape)
        x_t   = forward_sample_fn(batch_x0, t, noise)

        # 4) define loss wrt the current parameters
        def loss_fn(params):
            # predict the noise out of the U-Net
            eps_pred = noise_pred_fn({'params': params}, x_t, t)
            # simple MSE on the noise
            return jnp.mean((noise - eps_pred)**2)

        # 5) compute gradients and update
        loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
        new_train_state = train_state.apply_gradients(grads=grads)

        # 6) return new runner state + metrics
        new_runner_state = (rng, new_train_state, buffer)
        metrics = {'loss': loss}
        return new_runner_state, metrics

    # JIT it once so you get maximum performance
    return _train_step


