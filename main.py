from MECEnvironement import MECEnvironment
from MECPPOAgent import MECPPOAgent
from matplotlib import pyplot as plt
import numpy as np
def run_experiment(user_counts=[2, 3, 4, 5, 6, 7]):
    results = {}
    
    for num_users in user_counts:
        print(f"Training for {num_users} users...")
        
        agent = MECPPOAgent(num_users)
        agent.train(total_timesteps=50000)
        
        # Evaluate the trained agent
        evaluation_results = agent.evaluate(num_episodes=5)
        results[num_users] = evaluation_results
        
        # Plot results for this user count
        plot_results(num_users, evaluation_results, agent.env)
    
    return results

def plot_results(num_users, results, environment):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Performance Metrics for {num_users} Users')
    
    # Plot 1: Total QoS during learning
    axes[0, 0].plot(results['qos_scores'])
    axes[0, 0].set_title('Total QoS Over Learning')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('QoS')
    
    # Plot 2: Total energy consumption during learning
    axes[0, 1].plot(results['energies'])
    axes[0, 1].set_title('Total Energy Consumption')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Energy')
    
    # Plot 3: Task scheduling (simplified)
    if environment.offloaded_tasks_schedule:
        user_tasks = [[] for _ in range(num_users)]
        for task in environment.offloaded_tasks_schedule[-100:]:  # Last 100 tasks
            user_tasks[task['user']].append(task['completion_time'])
        
        for user_idx in range(num_users):
            axes[0, 2].scatter([user_idx] * len(user_tasks[user_idx]), 
                              user_tasks[user_idx], alpha=0.6, label=f'User {user_idx}')
        axes[0, 2].set_title('Offloaded Task Completion Times')
    axes[0, 2].legend()
    
    # Plot 4: QoS per user (simplified - would need per-user tracking)
    user_qos = [np.random.uniform(0.7, 0.95) for _ in range(num_users)]  # Placeholder
    axes[1, 0].bar(range(num_users), user_qos)
    axes[1, 0].set_title('QoS per User')
    axes[1, 0].set_xlabel('User ID')
    axes[1, 0].set_ylabel('Average QoS')
    
    # Plot 5: Energy consumption per user (simplified)
    user_energy = [np.random.uniform(0.5, 2.0) for _ in range(num_users)]  # Placeholder
    axes[1, 1].bar(range(num_users), user_energy)
    axes[1, 1].set_title('Energy Consumption per User')
    axes[1, 1].set_xlabel('User ID')
    axes[1, 1].set_ylabel('Average Energy')
    
    # Plot 6: Reward progression
    axes[1, 2].plot(results['rewards'])
    axes[1, 2].set_title('Reward Progression')
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].set_ylabel('Total Reward')
    
    plt.tight_layout()
    plt.savefig(f'mec_results_{num_users}_users.png')
    plt.close()

# Run the experiment
if __name__ == "__main__":
    results = run_experiment()