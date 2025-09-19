import jax
import jax.numpy as jnp
import flax.linen as nn

def pos_embedding(t, dim):
    half_dim = dim // 2
    emb = jnp.log(10000.0) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim) * -emb)
    emb = t[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
    return emb

def cosine_beta_schedule(timesteps, s=0.008, dtype=jnp.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = jnp.linspace(0, steps, steps)
    alphas_cumprod = jnp.cos(((x / steps) + s) / (1 + s) * jnp.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = jnp.clip(betas, a_min=0, a_max=0.999)
    return jnp.array(betas_clipped, dtype=dtype)
    

# x = jnp.ones(shape=(10, 10))

# t = jnp.array([1, 2, 3, 4, 5])

# print(pos_embedding(t, 10))
# x = pos_embedding(t, 10)
# print(x.shape)