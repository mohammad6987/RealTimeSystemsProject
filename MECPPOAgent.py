from MECEnvironement import MECEnvironment
import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt
class MECPPOAgent:
    def __init__(self, num_users):
        self.num_users = num_users
        self.env = MECEnvironment(num_users)
        
        # Policy network architecture
        policy_kwargs = dict(
            activation_fn=torch.nn.ReLU,
            net_arch=[dict(pi=[128, 128], vf=[128, 128])]
        )
        
        self.model = PPO(
            "MlpPolicy",
            self.env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01
        )
    
    def train(self, total_timesteps=100000):
        self.model.learn(total_timesteps=total_timesteps)
    
    def evaluate(self, num_episodes=10):
        rewards = []
        energies = []
        qos_scores = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_energy = 0
            episode_qos = 0
            
            done = False
            while not done:
                action, _ = self.model.predict(state, deterministic=True)
                state, reward, done, info = self.env.step({
                    'offload': action[:self.num_users],
                    'spectrum': action[self.num_users:]
                })
                
                episode_reward += reward
                episode_energy += info['energy_consumption']
                episode_qos += info['qos']
            
            rewards.append(episode_reward)
            energies.append(episode_energy)
            qos_scores.append(episode_qos)
        
        return {
            'rewards': rewards,
            'energies': energies,
            'qos_scores': qos_scores
        }