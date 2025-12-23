# Aligning Incentives and Resilience: Joint Node Selection and Resource Allocation in the Lightning Network - Gym Environment

[![Made with Python](https://img.shields.io/badge/Python->=3.8-red?logo=python&logoColor=white)](https://python.org)
[![Gym - >=0.26](https://img.shields.io/static/v1?label=Gym&message=>%3D0.26&color=black)](https://github.com/openai/gym)
[![NetworkX - ~3.2](https://img.shields.io/static/v1?label=NetworkX&message=~3.2&color=brightgreen)](https://networkx.org/)

## Overview

This repository provides a **Gym Environment** for Lightning Network channel and capacity selection research. This is a standalone environment package that allows researchers and developers to integrate their own reinforcement learning models for joint node selection and resource allocation in the Lightning Network. This environment is provided by the research paper team "Aligning Incentives and Resilience: Joint Node Selection and Resource Allocation in the Lightning Network".

**Key Features:**
- OpenAI Gym-compatible environment for Lightning Network channel opening
- Configurable transaction patterns and resource allocation scenarios
- Network simulation with realistic Lightning Network topology
- Support for custom RL models and algorithms
- Built-in Lightning Network data and transaction generators

## What's Included

This package includes:
- **Gym Environment** (`env/multi_channel.py`): The main JCoNaREnv environment
- **Simulator** (`simulator/`): Lightning Network simulation and transaction generation
- **Example Usage** (`example_usage.py`): Complete example showing how to use the environment
- **Data**: Sample Lightning Network topology and merchant data

## What's NOT Included

This is a **model-agnostic environment**. It does NOT include:
- Pre-trained models
- Training scripts for specific algorithms
- Model architectures
- Evaluation and baseline scripts

**You bring your own model!** This allows you to experiment with any RL algorithm or neural network architecture.

## Installation

### Install Dependencies

```bash
git clone https://github.com/incentivus/lightning-network-gym.git
cd lightning-network-gym
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
import gym
from env.multi_channel import JCoNaREnv
from simulator import preprocessing
import secrets

# Load Lightning Network data
def load_data(data_path, merchants_path, local_size, n_channels, local_heads_number, max_capacity):
    directed_edges = preprocessing.get_directed_edges(data_path)
    providers = preprocessing.get_providers(merchants_path)
    
    data = {
        'nodes': preprocessing.get_nodes(directed_edges),
        'src': preprocessing.get_src(directed_edges),
        'providers': providers,
        'local_heads_number': local_heads_number,
        'n_channels': n_channels
    }
    return data

# Create environment
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

data = load_data(
    env_params['data_path'],
    env_params['merchants_path'],
    env_params['local_size'],
    env_params['n_channels'],
    env_params['local_heads_number'],
    env_params['max_capacity']
)

# Initialize environment
env = JCoNaREnv(
    data=data,
    max_capacity=env_params['max_capacity'],
    max_episode_length=env_params['max_episode_length'],
    number_of_transaction_types=len(env_params['counts']),
    counts=env_params['counts'],
    amounts=env_params['amounts'],
    epsilons=env_params['epsilons'],
    capacity_upper_scale_bound=env_params['capacity_upper_scale_bound'],
    model="your_model_name",
    LN_graph=preprocessing.make_LN_graph(
        preprocessing.get_directed_edges(env_params['data_path']),
        data['providers']
    ),
    seed=secrets.randbelow(1000000)
)

# Use with your RL agent
obs = env.reset()
done = False

while not done:
    # Your model predicts action here
    action = your_model.predict(obs)
    obs, reward, done, info = env.step(action)
```

See `example_usage.py` for a complete working example.

## Environment Details

### Observation Space

The environment provides a `gym.spaces.Dict` observation space containing:
- Network topology information
- Node features and attributes
- Current channel states
- Transaction history

### Action Space

The environment uses a `gym.spaces.MultiDiscrete` action space for:
- Selecting nodes for channel opening
- Allocating channel capacities

### Reward Function

During the reward function process we consider:
- Transaction success rates
- Network centralization metrics
- Resource utilization efficiency
- Channel balance and liquidity

Reward function is generated based on:
- Total revenue achieved from transaction throughput

## Integrating Your Model

### With Stable-Baselines3

```python
from stable_baselines3 import PPO

model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=200000)
```

### With Custom PyTorch/TensorFlow Models

```python
# Implement your custom training loop
for episode in range(num_episodes):
    obs = env.reset()
    done = False
    
    while not done:
        action = your_model.forward(obs)
        obs, reward, done, info = env.step(action)
        your_model.update(obs, action, reward)
```

### With Graph Neural Networks

The environment provides network graph structures compatible with PyTorch Geometric, DGL, and other GNN libraries.

## Environment Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_capacity` | Maximum channel capacity | 1e7 |
| `max_episode_length` | Maximum steps per episode | 10 |
| `counts` | Transaction counts per type | [10, 10, 10] |
| `amounts` | Transaction amounts (satoshi) | [10000, 50000, 100000] |
| `epsilons` | Merchant ratios | [0.6, 0.6, 0.6] |
| `capacity_upper_scale_bound` | Capacity scaling upper bound | 25 |

## Data Format

The environment expects:
- **data.json**: Lightning Network graph structure
- **merchants.json**: Merchant node information

Sample data is provided in the `data/` directory.

## Original Research

This environment is derived from the research project: [Lightning-Network-Centralization](https://github.com/incentivus/Lightning-Network-Centralization)

## License

This work is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International](http://creativecommons.org/licenses/by-nc-nd/4.0/) license.

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a>

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For questions and issues, please open an issue on the GitHub repository.
