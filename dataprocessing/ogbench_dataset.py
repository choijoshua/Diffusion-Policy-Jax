import collections 
from typing import Optional
import pickle 
import gymnasium as gym 
import ogbench 
import numpy as np
from dataprocessing.utils import concatenate_batches, calc_return_to_go


def get_ogbench_dataset(
    env,
    reward_scale:float = 1.0,
    reward_bias: float = 0.0,
    clip_action: Optional[float] = None,
    normalize: bool = False,
):
    
    env, dataset, _ = ogbench.make_env_and_datasets(env)
    
    use_terminals=False 
    if 'terminals' in dataset.keys():
        use_terminals=True

    dones_float = np.logical_or(dataset['terminals'], np.logical_not(dataset['masks']))
    
    dataset['rewards'] = dataset['rewards'] * reward_scale + reward_bias
    if normalize:
        normalize_params = {
            'obs_mean': np.mean(dataset["observations"], axis=0),
            'obs_std': np.std(dataset["observations"], axis=0) + 1e-8,
            'next_obs_mean': np.mean(dataset["next_observations"], axis=0),
            'next_obs_std': np.std(dataset["next_observations"], axis=0) + 1e-8,
        }
        
        dataset["observations"] = (
            dataset["observations"] - np.mean(dataset["observations"], axis=0)
        ) / (np.std(dataset["observations"], axis=0) + 1e-8)
        
        dataset["next_observations"] = (
            dataset["next_observations"] - np.mean(dataset["next_observations"], axis=0)
        ) / (np.std(dataset["next_observations"], axis=0) + 1e-8)
        
    else:
        normalize_params= None
    
    return dict(
        observations=dataset["observations"],
        actions=dataset["actions"],
        next_observations=dataset["next_observations"],
        rewards=dataset["rewards"],
        dones=dones_float,
        masks=dataset['masks'],
    ), normalize_params

def get_ogbench_with_mc_calculation(
    env_name,
    reward_scale: float = 1.0,
    reward_bias: float = 0.0,
    clip_action: Optional[float] = None,
    gamma:float = 0.99,
    normalize:bool=False,
    ):
    
    dataset = qlearning_dataset_and_calc_mc(env_name,
                                   reward_scale,
                                   reward_bias,
                                   clip_action,
                                   gamma)
    dataset['rewards'] = dataset['rewards'].reshape(-1)
    dataset['terminals'] = dataset['terminals'].reshape(-1)
    dataset['mc_returns'] = dataset['mc_returns'].reshape(-1)

    if normalize:
        normalize_params = {
            'obs_mean': np.mean(dataset["observations"], axis=0),
            'obs_std': np.std(dataset["observations"], axis=0) + 1e-8,
            'next_obs_mean': np.mean(dataset["next_observations"], axis=0),
            'next_obs_std': np.std(dataset["next_observations"], axis=0) + 1e-8,
        }
        
        dataset["observations"] = (
            dataset["observations"] - np.mean(dataset["observations"], axis=0)
        ) / (np.std(dataset["observations"], axis=0) + 1e-8)
        
        dataset["next_observations"] = (
            dataset["next_observations"] - np.mean(dataset["next_observations"], axis=0)
        ) / (np.std(dataset["next_observations"], axis=0) + 1e-8)
        
    else:
        normalize_params= None
    


    return dict(
        observations=dataset["observations"],
        actions=dataset["actions"],
        next_observations=dataset["next_observations"],
        rewards=dataset["rewards"],
        dones=dataset["terminals"].astype(np.float32),
        mc_returns=dataset["mc_returns"],
        masks=dataset['masks'],
        last_reward=dataset['last_reward'],
        terminals=dataset['terminals'],
        sum_of_rewards = dataset['sum_of_rewards'],
        #total_length = dataset['ep_length'],
    ), normalize_params
    
    
def qlearning_dataset_and_calc_mc(
    env,
    reward_scale,
    reward_bias,
    clip_action,
    gamma,
    dataset=None,
    terminate_on_end=False,
    ):
    
    env, dataset,_ = ogbench.make_env_and_datasets(env)
    N = dataset['observations'].shape[0]
    
    # First, identify which indices to keep
    indices_to_keep = []
    for i in range(N):
        if dataset['masks'][i] == 1:
            # Always keep mask=1 entries
            indices_to_keep.append(i)
        elif dataset['masks'][i] == 0:
            # Only keep if it's the first mask=0 in a sequence
            # (i.e., previous mask was 1 or it's the first entry)
            if i == 0 or dataset['masks'][i-1] == 1:
                indices_to_keep.append(i)
    
    # Filter the dataset
    filtered_dataset = {}
    for k in dataset.keys():
        if k in ('actions', 'next_observations','observations','rewards','terminals','timeouts','masks'):
            filtered_dataset[k] = dataset[k][indices_to_keep]
    
    # Now process the filtered dataset
    N_filtered = len(indices_to_keep)
    filtered_dataset['terminals'] = np.logical_or(
        filtered_dataset['terminals'], 
        np.logical_not(filtered_dataset['masks'])
    )
    
    data_ = collections.defaultdict(list)
    episodes_dict_list = []
    episode_step = 0
    
    for i in range(N_filtered):
        done_bool = bool(filtered_dataset['terminals'][i])
        
        for k in filtered_dataset.keys():
            if k in ('actions', 'next_observations','observations','rewards','terminals','timeouts','masks'):
                data_[k].append(filtered_dataset[k][i])
        
        episode_step += 1
        
        if done_bool or i == N_filtered - 1 and episode_step > 0:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            
            episode_data['rewards'] = (episode_data['rewards'] * reward_scale + reward_bias)
            episode_data['mc_returns'] = calc_return_to_go(
                env.spec.name,
                episode_data['rewards'],
                episode_data['masks'],
                gamma,
                reward_scale,
                reward_bias,
                infinite_horizon=True,
            )
            
            episode_data['last_reward'] = np.ones_like(episode_data['rewards'])*episode_data['rewards'][-1]
            episode_data['sum_of_rewards'] = np.ones_like(episode_data['rewards'])*np.sum(episode_data['rewards'])
            episode_data['ep_length'] = np.ones_like(episode_data['rewards'])*len(episode_data['rewards'])
            
            if clip_action is not None:
                episode_data["actions"] = np.clip(
                    episode_data["actions"], -clip_action, clip_action
                )
            
            episodes_dict_list.append(episode_data)
            data_ = collections.defaultdict(list)

    return concatenate_batches(episodes_dict_list)

def old_qlearning_dataset_and_calc_mc(
    env,
    reward_scale,
    reward_bias,
    clip_action,
    gamma,
    dataset=None,
    terminate_on_end=False,
    
    ):
    
    env, dataset,_ = ogbench.make_env_and_datasets(env)
    N = dataset['observations'].shape[0]
    data_ = collections.defaultdict(list)
    episodes_dict_list = []
    
    
    
    dataset['terminals'] = np.logical_or(dataset['terminals'], np.logical_not(dataset['masks']))
    
    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        
        for k in dataset.keys():
            if k in ('actions', 'next_observations','observations','rewards','terminals','timeouts','masks'):
                
                data_[k].append(dataset[k][i])
            
        episode_step +=1
        
            
        if done_bool or i == N - 1 and episode_step>0:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            
            episode_data['rewards'] = (episode_data['rewards'] * reward_scale + reward_bias)
            episode_data['mc_returns'] = calc_return_to_go(
                env.spec.name,
                episode_data['rewards'],
                episode_data['masks'],
                gamma,
                reward_scale,
                reward_bias,
                infinite_horizon=True,
            )
            
            episode_data['last_reward'] = np.ones_like(episode_data['rewards'])*episode_data['rewards'][-1]
            
            episode_data['sum_of_rewards'] = np.ones_like(episode_data['rewards'])*np.sum(episode_data['rewards'])
            episode_data['ep_length'] = np.ones_like(episode_data['rewards'])*len(episode_data['rewards'])
            if clip_action is not None:
                episode_data["actions"] = np.clip(
                    episode_data["actions"], -clip_action, clip_action
                )
            episodes_dict_list.append(episode_data)
            data_ = collections.defaultdict(list)

    return concatenate_batches(episodes_dict_list)