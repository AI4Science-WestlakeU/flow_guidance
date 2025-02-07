from collections import namedtuple
import os
import numpy as np
import torch
import pdb

import tqdm

from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer


Batch = namedtuple('Batch', 'trajectories conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')


class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env='hopper-medium-replay', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, seed=None):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env_name = env
        self.env = env = load_environment(env)
        self.env.seed(seed)
        self.horizon = horizon
        self.max_path_length = max_path_length
        itr = sequence_dataset(env, self.preprocess_fn)

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()
        
        # cache normalizer
        normalizer_cache_path = os.path.join('logs', self.env_name, 'normalizer_cache', f'normalizer_H{horizon}.pth')
        try:
            self.normalizer = torch.load(normalizer_cache_path)
            print(f'[ datasets/sequence ] Normalizer loaded from {normalizer_cache_path}')
        except Exception as e:
            print(f'[ datasets/sequence ] Normalizer not found at {normalizer_cache_path}, calculating...')
            self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
            os.makedirs(os.path.dirname(normalizer_cache_path), exist_ok=True)
            torch.save(self.normalizer, normalizer_cache_path)
            print(f'[ datasets/sequence ] Normalizer saved at {normalizer_cache_path}')

        print('[ datasets/sequence ] Normalizers acquired')
        self.indices = self.make_indices(fields.path_lengths, horizon)
        print('[ datasets/sequence ] Indices made')

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()
        print('[ datasets/sequence ] Normalized')

        print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        print("Normalizing...")
        for key in tqdm.tqdm(keys):
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):

        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        conditions = self.get_conditions(observations) # (T, C_state)
        trajectories = np.concatenate([actions, observations], axis=-1) # (T, C_action + C_state)
        batch = Batch(trajectories, conditions)
        return batch


class GoalDataset(SequenceDataset):

    def get_conditions(self, observations):
        '''
            condition on both the current observation and the last observation in the plan
        '''
        return {
            0: observations[0],
            self.horizon - 1: observations[-1],
        }


class ValueDataset(SequenceDataset):
    '''
        adds a value field to the datapoints for training the value function
    '''

    def __init__(self, *args, discount=0.99, normed=False, inf_horizon=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:,None]
        self.inf_horizon = inf_horizon
        self.normed = False
        if normed:
            cache_path = os.path.join('logs', self.env_name, 'normalizer_cache', f'value_bounds_H{self.horizon}.pth')
            try:
                print(f'[ datasets/sequence ] Value bounds loaded from {cache_path}')
                self.vmin, self.vmax = torch.load(cache_path)
            except Exception as e:
                print(f'[ datasets/sequence ] Value bounds not found, calculating... ')
                self.vmin, self.vmax = self._get_bounds()
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                torch.save((self.vmin, self.vmax), cache_path)
            self.normed = True

    def _get_bounds(self):
        print('[ datasets/sequence ] Getting value dataset bounds...', end=' ', flush=True)
        vmin = np.inf
        vmax = -np.inf
        for i in tqdm.tqdm(range(len(self.indices))):
            value = self.__getitem__(i).values.item()
            vmin = min(value, vmin)
            vmax = max(value, vmax)
        print('✓')
        return vmin, vmax

    def normalize_value(self, value):
        ## [0, 1]
        normed = (value - self.vmin) / (self.vmax - self.vmin)
        ## [-1, 1]
        normed = normed * 2 - 1
        return normed

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        path_ind, start, end = self.indices[idx]
        if self.inf_horizon:
            rewards = self.fields['rewards'][path_ind, start:]
        else:
            rewards = self.fields['rewards'][path_ind, start: end] # Only rewards in the horizon is predicted! Why does Diffuser use a diffusion model to learn this?
        discounts = self.discounts[:len(rewards)]
        value = (discounts * rewards).sum()
        if self.normed:
            value = self.normalize_value(value)
        value = np.array([value], dtype=np.float32)
        value_batch = ValueBatch(*batch, value)
        return value_batch
