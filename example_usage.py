import gym
from env.multi_channel import JCoNaREnv
from simulator import preprocessing
import networkx as nx
import secrets

def load_data(data_path, merchants_path, local_size, n_channels, local_heads_number, max_capacity):
    \"\"\"Load the Lightning Network data.\"\"\"
    directed_edges = preprocessing.get_directed_edges(data_path)
    providers = preprocessing.get_providers(merchants_path)
    
    # Create data structure needed for the environment
    data = {
        'nodes': preprocessing.get_nodes(directed_edges),
        'src': preprocessing.get_src(directed_edges),
        'providers': providers,
        'local_heads_number': local_heads_number,
        'n_channels': n_channels
    }
    return data

def make_env(data, max_capacity, max_episode_length, counts, amounts, epsilons, capacity_upper_scale_bound, data_path):
    \"\"\"Create the JCoNaR gym environment.
    
    Args:
        data: Dictionary containing LN network data
        max_capacity: Maximum channel capacity
        max_episode_length: Maximum steps per episode
        counts: List of transaction counts per type
        amounts: List of transaction amounts per type
        epsilons: List of merchant ratios per type
        capacity_upper_scale_bound: Upper bound for capacity scaling
        data_path: Path to the Lightning Network data
    
    Returns:
        Gym environment instance
    \"\"\"
    directed_edges = preprocessing.get_directed_edges(data_path)
    providers = data['providers']
    
    # Create the Lightning Network graph
    G = preprocessing.make_LN_graph(directed_edges, providers)
    
    # Create and return the environment
    env = JCoNaREnv(
        data=data,
        max_capacity=max_capacity,
        max_episode_length=max_episode_length,
        number_of_transaction_types=len(counts),
        counts=counts,
        amounts=amounts,
        epsilons=epsilons,
        capacity_upper_scale_bound=capacity_upper_scale_bound,
        model=\"custom\",  # Set to your model type
        LN_graph=G,
        seed=secrets.randbelow(1000000)
    )
    
    return env

# Example usage
if __name__ == \"__main__\":
    # Environment parameters
    env_params = {
        'data_path': 'data/data.json',
        'merchants_path': 'data/merchants.json',
        'local_size': 50,
        'n_channels': 5,
        'local_heads_number': 10,
        'max_capacity': 1e7,
        'max_episode_length': 10,
        'counts': [10, 10, 10],
        'amounts': [10000, 50000, 100000],
        'epsilons': [0.6, 0.6, 0.6],
        'capacity_upper_scale_bound': 25
    }
    
    # Load data
    data = load_data(
        env_params['data_path'],
        env_params['merchants_path'],
        env_params['local_size'],
        env_params['n_channels'],
        env_params['local_heads_number'],
        env_params['max_capacity']
    )
    
    # Create environment
    env = make_env(
        data,
        env_params['max_capacity'],
        env_params['max_episode_length'],
        env_params['counts'],
        env_params['amounts'],
        env_params['epsilons'],
        env_params['capacity_upper_scale_bound'],
        env_params['data_path']
    )
    
    print(\"Environment created successfully!\")
    print(f\"Observation space: {env.observation_space}\")
    print(f\"Action space: {env.action_space}\")
    
    # Example: Run one episode
    obs = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Your model would predict action here
        action = env.action_space.sample()  # Random action for example
        obs, reward, done, info = env.step(action)
        total_reward += reward
    
    print(f\"Episode finished with total reward: {total_reward}\")
