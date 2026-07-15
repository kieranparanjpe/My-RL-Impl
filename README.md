# My-RL-Impl

## Project Description
The goal of this project was to learn about reinforcement learning by implementing PPO myself. I have attempted 
similar projects twice before. The [first RL project](https://github.com/kieranparanjpe/GridworldRL) I made was a 
simple implementation of SARSA applied to a gridworld, and 
[the second](https://github.com/kieranparanjpe/BipedalRL) was an attempt to implement REINFORCE with eligibility traces applied to bipedal walking. The first 
project was a success, the second was too ambitious.  As a result, I decided to make this project, with the goal of 
being able to easily implement different algorithms and try them on different tasks.

This project contains 4 key sections: the training loop, the MDP, the algorithms, and the logging. It is all 
decoupled such that we can use the same training loop to train on different tasks with different algorithms, 
making it easy to iterate and learn. 

A more detailed report of my implementation and results, can be found [here](https://kieranparanjpe.github.io/My-RL-Impl/report/PPO_Report.pdf).

### MDP support

Right now there is MDP support for [gymnasium](https://gymnasium.farama.org/index.html) tasks. I've tried training 
policies on CartPole-v1, LunarLander-v3, HalfCheetah-v5, Walker2d-v5 and Humanoid-v5. Note that these are a variety of 
continous/discrete action and observation spaces.

### Algorithms

I implemented proximal policy optimization (PPO) with support for both discrete and continuous action spaces. The 
same implementation of PPO can be used for each type of policy, as this is further abstracted.

#### Policies

I implemented 3 different policies: `categorical_policy.py`, `single_beta_policy.py`, `single_gaussian_policy.py`. 
They work by running the forward pass of a neural network to estimate the parameters of the associated 
distribution, which can then be sampled from. Note: the gaussian policy may not work as expected; I've been using 
the beta policy more for continuous tasks because it automatically clamps outputs to \[0, 1\].

### Logging

I use Weights & Biases to log training. The dashboard I used can be found [here](https://wandb.ai/kieranparanjpe-mcgill-university/RL_Project1/runs). I also record videos 5 times during 
training and save them locally, as well as upload them to W&B.

## Running Locally

### Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

**1. Install uv** (if you don't have it):
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2. Clone the repository:**
```
git clone https://github.com/kieranparanjpe/My-RL-Impl.git
cd My-RL-Impl
```

**3. Create the virtual environment and install dependencies:**
```
uv sync
```
This automatically creates a `.venv` and installs all pinned dependencies.

**4. Activate the virtual environment:**
```
source .venv/bin/activate
```

### Running

There are two files to run: `train.py` and `evaluate.py`.

#### Training
```
uv run python -m src.train \
    --environment "<gymnasium environment id>" \
    --policy "[categorical, single_beta, single_gaussian]" \
    --algorithm "ppo" \
    --hyperparameters "hyperparameters/half-cheetah-1.json" \
    --grid "hyperparameters/half-cheetah-grid1.json" \
    -r -s -l
```
`--hyperparameters` will load hyperparams from a json file, and `--grid` will create a grid of hyperparameters to
search over, and spawn n runs to try each combination. Do not use both at the same time.

The json file is split into four sections. All sections are optional and fall back to defaults if omitted:

```json
{
  "algorithm": {
    "n_timesteps": 2000000,
    "lr": 0.0003,
    "value_lr": 0.0003,
    "gamma": 0.99,
    "lamda": 0.95,
    "importance_ratio_clip": 0.2,
    "batch_size": 64,
    "buffer_size": 2048,
    "gradient_epochs": 10,
    "entropy_loss_weight": 0.0001
  },
  "policy": {
    "hidden_sizes": [64, 64],
    "activation": "relu",
    "action_range": [-0.4, 0.4]
  },
  "value_fn": {
    "hidden_sizes": [64, 64],
    "activation": "relu"
  },
  "mdp": {
    "normalise_obs": true,
    "normalise_reward": true,
    "reward_norm_gamma": 0.99
  }
}
```

`action_range` is only used by `single_beta`. Supported activations are `relu`, `tanh`, and `leaky_relu`.

For grid search, any value that is a list becomes a grid dimension. All grid dimensions across all sections are
combined into a cartesian product. Example files can be found in `/hyperparameters`.

Use `-r` to enable recording, `-s` to enable policy saving, and `-l` to enable logging. To enable logging, you will 
need to log in with 
```
wandb login
```

#### Evaluating
This will display a trained policy visually. 
```
uv run python -m src.evaluate \
    --environment "<gymnasium environment id>" \
    --policy "[categorical, single_beta, single_gaussian]" \
    --weights "path/to/saved/weights.pth"
```

