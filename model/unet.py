import jax
import jax.numpy as jnp
import flax.linen as nn
from config import TrainArgs
from model.util import pos_embedding


class TimeEmbedding(nn.Module):
    embed_dim: int
    
    @nn.compact
    def __call__(self, t):
        t = nn.Dense(features=self.embed_dim * 4)(t)
        t = nn.swish(t)
        t = nn.Dense(features=self.embed_dim)(t)
        return t


class ObservationEmbedding(nn.Module):
    embed_dim: int
    
    @nn.compact
    def __call__(self, obs):
        obs = nn.Dense(features=self.embed_dim * 4)(obs)
        obs = nn.swish(obs)
        obs = nn.Dense(features=self.embed_dim)(obs)
        return obs


class Conv1dBlock(nn.Module):
    features: int
    kernel_size: int
    num_groups: int = 8
    
    @nn.compact
    def __call__(self, x):
        # Added padding='SAME' to preserve spatial dimensions
        x = nn.Conv(
            features=self.features, 
            kernel_size=(self.kernel_size,), 
            strides=(1,),
            padding='SAME'
        )(x)
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
        # Save input for residual connection BEFORE any operations
        residual = x
        
        # Project time embedding
        t_proj = nn.Dense(self.out_channels)(nn.swish(t))
        t_proj = jnp.expand_dims(t_proj, axis=1)  # [batch, 1, out_channels]
        
        # First conv block
        x = Conv1dBlock(features=self.out_channels, kernel_size=self.kernel_size)(x)
        
        # Add time embedding
        x = x + t_proj
        
        # Second conv block
        x = Conv1dBlock(features=self.out_channels, kernel_size=self.kernel_size)(x)
        
        # Project residual if channel dimensions don't match
        if residual.shape[-1] != self.out_channels:
            residual = nn.Conv(
                features=self.out_channels, 
                kernel_size=(1,),
                padding='SAME'
            )(residual)
        
        return x + residual


class DownSample(nn.Module):
    out_channels: int
    embed_dim: int
    kernel_size: int
    
    @nn.compact
    def __call__(self, x, t):
        # Residual block at current resolution
        h = ResidualBlock(self.out_channels, self.embed_dim, self.kernel_size)(x, t)
        
        # Downsample with proper padding
        out = nn.Conv(
            features=self.out_channels, 
            kernel_size=(3,), 
            strides=(2,),
            padding=((1, 1),)  # Explicit padding for strided conv
        )(h)
        
        return h, out


class UpSample(nn.Module):
    out_channels: int
    
    @nn.compact
    def __call__(self, x):
        # Upsample with proper padding
        x = nn.ConvTranspose(
            features=self.out_channels, 
            kernel_size=(4,), 
            strides=(2,),
            padding='SAME'
        )(x)
        return x


class UNet(nn.Module):
    args: TrainArgs
    action_dim: int
    
    @nn.compact
    def __call__(self, x, t, obs):
        '''
        x: noisy actions [batch, horizon, action_dim]
        t: timesteps [batch,]
        obs: observations [batch, obs_dim]
        
        returns: noise/score prediction [batch, horizon, action_dim]
        '''
        # Convert timesteps to float if needed
        if t.dtype in (jnp.int32, jnp.int64):
            t = t.astype(jnp.float32)
        
        # Create embeddings
        t_embed = pos_embedding(t, self.args.embed_dim)  # [batch, embed_dim]
        time_embed = TimeEmbedding(self.args.embed_dim)(t_embed)  # [batch, embed_dim]
        obs_embed = ObservationEmbedding(self.args.embed_dim)(obs)  # [batch, embed_dim]
        
        # Concatenate along feature dimension and project back
        global_cond = jnp.concatenate([obs_embed, time_embed], axis=-1)  # [batch, 2*embed_dim]
        global_cond = nn.Dense(self.args.embed_dim)(global_cond)  # [batch, embed_dim]
        
        # Initial projection to first hidden dimension
        x = nn.Conv(
            features=self.args.dims[0], 
            kernel_size=(3,), 
            padding='SAME'
        )(x)  # [batch, horizon, dims[0]]
        
        # Encoder (downsampling path)
        hidden = []
        for dim in self.args.dims:
            h, x = DownSample(
                out_channels=dim, 
                embed_dim=self.args.embed_dim, 
                kernel_size=3
            )(x, global_cond)
            hidden.append(h)
        
        # Bottleneck
        x = ResidualBlock(self.args.dims[-1], self.args.embed_dim, 3)(x, global_cond)
        x = ResidualBlock(self.args.dims[-1], self.args.embed_dim, 3)(x, global_cond)
        
        # Decoder (upsampling path)
        for i in range(len(self.args.dims) - 1, -1, -1):
            # Upsample
            x = UpSample(self.args.dims[i])(x)
            
            # Get skip connection
            skip = hidden.pop()
            
            # Match spatial dimensions if needed (handle odd/even sizes)
            if x.shape[1] != skip.shape[1]:
                min_len = min(x.shape[1], skip.shape[1])
                x = x[:, :min_len, :]
                skip = skip[:, :min_len, :]
            
            x = jnp.concatenate([x, skip], axis=-1)  # [batch, horizon, 2*dims[i]]
            
            # Residual block
            x = ResidualBlock(self.args.dims[i], self.args.embed_dim, 3)(x, global_cond)
        
        # Final projection to action dimension
        x = nn.Conv(features=self.action_dim, kernel_size=(3,), padding='SAME')(x)
        
        return x