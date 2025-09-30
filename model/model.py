import jax
import jax.numpy as jnp
import flax.linen as nn
from config import Args
from util import pos_embedding


class TimeEmbedding(nn.Module):
    embed_dim: int
    @nn.compact
    def __call__(self, t):
        t = nn.Dense(features=self.embed_dim*4)(t)
        t = nn.swish(t)
        t = nn.Dense(features=self.embed_dim)(t)
        return t
    
class ObservationEmbedding(nn.Module):
    embed_dim: int
    @nn.compact
    def __call__(self, obs):
        obs = nn.Dense(features=self.embed_dim*4)(obs)
        obs = nn.swish(obs)
        obs = nn.Dense(features=self.embed_dim)(obs)
        return obs
    
class Conv1dBlock(nn.Module):
    features: int
    kernel_size: int
    num_groups: int = 8
  
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.features, kernel_size=(self.kernel_size,), strides=(1,))(x)
        x = nn.GroupNorm(num_groups=self.num_groups)(x)
        x = nn.swish(x)
        return x
    
class ResidualBlock(nn.Module):
    out_channels: int
    embed_dim: int
    kernel_size: int

    @nn.compact
    def __call__(self, x, t):
        '''
            x: [batch, horizon, in_channels]
            t: [batch, embed_dim]
        returns:
            x: [batch, horizon, out_channels]
        '''
        t = nn.Dense(self.out_channels)(nn.swish(t))
        x = Conv1dBlock(features=self.out_channels, kernel_size=self.kernel_size)(x)
        t = jnp.expand_dims(t, axis=1)
        x += t
        x = Conv1dBlock(features=self.out_channels, kernel_size=self.kernel_size)(x)
        residual = nn.Conv(self.out_channels, kernel_size=(self.kernel_size,))(x)
        return x + residual
    
# Kernel size = 3, Stride = 2, Padding = 1
class DownSample(nn.Module):
    out_channels: int
    embed_dim: int
    kernel_size: int

    @nn.compact
    def __call__(self, x, t):
        h = ResidualBlock(self.out_channels, self.embed_dim, self.kernel_size)(x, t)
        out = nn.Conv(features=self.out_channels, kernel_size=(3,), strides=(2, ))(h)
        return h, out

class UpSample(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x):
        x = nn.ConvTranspose(features=self.out_channels, kernel_size=(4, ), strides=(2, ))(x)
        return x
    
class UNet(nn.Module):
    args: Args
    obs_dim: int
    action_dim: int

    @nn.compact
    def __call__(self, x, t, obs):
        
        t = pos_embedding(t, self.args.embed_dim)
        
        time_embed = TimeEmbedding(self.args.embed_dim)(t)
        obs_embed = ObservationEmbedding(self.args.embed_dim)(obs)

        time_embed = jnp.concatenate([obs_embed, time_embed], axis=1)

        hidden = []
        for dim in self.args.dims:
            h, x = DownSample(out_channels=dim, embed_dim=self.args.embed_dim, kernel_size=3)(x, time_embed)
            hidden.append(h)
        
        x = ResidualBlock(self.args.dims[-1], self.args.embed_dim, 3)(x, time_embed)
        x = ResidualBlock(self.args.dims[-1], self.args.embed_dim, 3)(x, time_embed)
        x = UpSample(self.args.dims[-1])(x)

        for i in range(len(self.args.dims)-1, -1, -1):
            x = jnp.concatenate([x, hidden.pop()], axis=1)
            x = ResidualBlock(self.args.dims[i], self.args.embed_dim, 3)(x, time_embed)
            if i != 0: 
                x = UpSample(self.args.dims[i-1])(x)
        
        x = nn.Conv(features=self.action_dim, kernel_size=(3,))(x)

        return x

        



        



