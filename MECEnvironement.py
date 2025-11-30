import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt

class MECEnvironment(gym.Env):
    def __init__(self, num_users=5):
        super(MECEnvironment, self).__init__()
        
        self.num_users = num_users
        self.current_step = 0
        self.max_steps = 1000
        
        # Action space: [offload_decision, spectrum_ratio] for each user
        # offload_decision: 0 (local) or 1 (offload)
        # spectrum_ratio: continuous value between 0 and 1
        self.action_space = spaces.Dict({
            'offload': spaces.MultiBinary(num_users),
            'spectrum': spaces.Box(low=0, high=1, shape=(num_users,), dtype=np.float32)
        })
        
        # State space: task features + user features + system state
        state_dim = num_users * 5 + 3  # 5 features per user + 3 system features
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(state_dim,), dtype=np.float32)
        
        # System parameters from Project20.pdf
        self.server_frequency = 6.0  # GHz
        self.server_power = 5.0  # W
        self.spectrum_bandwidth = 20.0  # MHz
        
        # Initialize users and tasks
        self.reset()
    
    def reset(self):
        self.current_step = 0
        self.total_energy_consumption = 0
        self.total_qos = 0
        self.offloaded_tasks_schedule = []
        
        # Initialize user positions and capabilities
        self.user_distances = np.random.uniform(10, 50, self.num_users)
        self.user_frequencies = np.random.uniform(0.5, 1.5, self.num_users)
        
        # Generate initial tasks
        self.generate_tasks()
        
        return self._get_state()
    
    def generate_tasks(self):
        # Generate tasks according to Project20.pdf specifications
        self.task_data_sizes = np.random.uniform(16, 80, self.num_users)  # Mbit
        self.task_required_cycles = np.random.uniform(1.5, 3.5, self.num_users)  # GHz (converted to Giga cycles)
        self.task_deadlines = np.random.uniform(2, 4, self.num_users)  # seconds
    
    def _get_state(self):
        # State includes: task features, user features, and system state
        state = []
        
        # Task features for each user
        for i in range(self.num_users):
            state.extend([
                self.task_data_sizes[i],
                self.task_required_cycles[i],
                self.task_deadlines[i],
                self.user_frequencies[i],
                self.user_distances[i]
            ])
        
        # System state
        state.extend([
            self.total_energy_consumption / (self.current_step + 1),  # Average energy
            self.total_qos / (self.current_step + 1),  # Average QoS
            len(self.offloaded_tasks_schedule)  # Number of offloaded tasks
        ])
        
        return np.array(state, dtype=np.float32)
    
    def calculate_local_execution(self, user_idx):
        """Calculate local execution time and energy"""
        execution_time = self.task_required_cycles[user_idx] / self.user_frequencies[user_idx]
        
        # Energy consumption based on paper's model
        energy = self.task_required_cycles[user_idx] * 1e-3  # Simplified model
        
        return execution_time, energy
    
    def calculate_offload_execution(self, user_idx, spectrum_ratio):
        """Calculate offload execution time and energy"""
        # Transmission time (simplified)
        transmission_rate = spectrum_ratio * self.spectrum_bandwidth * np.log2(1 + 1/self.user_distances[user_idx])
        transmission_time = self.task_data_sizes[user_idx] / transmission_rate
        
        # Server execution time
        server_execution_time = self.task_required_cycles[user_idx] / self.server_frequency
        
        # Total time
        total_time = transmission_time + server_execution_time
        
        # Energy consumption (transmission + server)
        transmission_energy = 0.1 * transmission_time  # Simplified transmission energy
        server_energy = self.server_power * server_execution_time * spectrum_ratio
        
        total_energy = transmission_energy + server_energy
        
        return total_time, total_energy
    
    def calculate_qos(self, execution_time, deadline):
        """Calculate QoS based on Project20.pdf formula"""
        if execution_time <= deadline:
            return 1.0
        elif execution_time <= 2 * deadline:
            return 1.0 - (execution_time - deadline) / deadline
        else:
            return 0.0
    
    def step(self, action):
        offload_decisions = action['offload']
        spectrum_allocations = action['spectrum']
        
        total_energy = 0
        total_qos = 0
        step_offloaded_tasks = []
        
        # Normalize spectrum allocations for offloaded users only
        offloaded_mask = offload_decisions.astype(bool)
        if np.any(offloaded_mask):
            total_spectrum = np.sum(spectrum_allocations[offloaded_mask])
            if total_spectrum > 1.0:
                spectrum_allocations[offloaded_mask] /= total_spectrum
        
        for i in range(self.num_users):
            if offload_decisions[i] == 0:  # Local execution
                exec_time, energy = self.calculate_local_execution(i)
            else:  # Offload execution
                exec_time, energy = self.calculate_offload_execution(i, spectrum_allocations[i])
                step_offloaded_tasks.append({
                    'user': i,
                    'deadline': self.task_deadlines[i],
                    'completion_time': exec_time
                })
            
            qos = self.calculate_qos(exec_time, self.task_deadlines[i])
            
            total_energy += energy
            total_qos += qos
        
        self.total_energy_consumption += total_energy
        self.total_qos += total_qos
        self.offloaded_tasks_schedule.extend(step_offloaded_tasks)
        
        # Reward: maximize QoS and minimize energy consumption
        reward = total_qos - 0.1 * total_energy  # Weight can be adjusted
        
        self.current_step += 1
        self.generate_tasks()  # Generate new tasks for next step
        
        done = self.current_step >= self.max_steps
        
        info = {
            'energy_consumption': total_energy,
            'qos': total_qos,
            'offloaded_tasks': step_offloaded_tasks
        }
        
        return self._get_state(), reward, done, info
    
    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Total Energy: {self.total_energy_consumption:.2f}, Total QoS: {self.total_qos:.2f}")