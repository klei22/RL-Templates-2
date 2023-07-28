# README: Custom Environment RL Template

This Python script is a template for training a reinforcement learning agent to optimize a target function, with the help of Stable Baselines3 library. It contains a custom Gym environment, in which the agent learns to interact.

The target function, by default, is a complex 2D function with local minima and maxima. The agent's task is to find the global maximum.

## Setup

Ensure you have Python (version 3.7 or newer) installed in your system. You'll also need to install the required libraries.

This can be done via pip:

```bash
pip install requirements.txt
```

## How to use

You can configure the agent training process via command-line arguments:

```
    --model - The model to use, either 'ppo' or 'a2c'. Default is 'ppo'.
    --max_timesteps - The maximum number of timesteps to train for. Default is 100,000.
    --initial_x - The initial value of the x coordinate of the environment's state. Default is 0.0.
    --initial_y - The initial value of the y coordinate of the environment's state. Default is 0.0.
```

You can run the script with custom settings like so:

```bash
python rl_template.py --model a2c --max_timesteps 200000 --initial_x -1.0 --initial_y 1.0
```

This will train an A2C model for 200,000 timesteps, and the environment's state will be initialized at (-1.0, 1.0).
Modifying the Environment

You can modify the CustomEnv class and the target_function according to your needs:

```
    target_function: This is the function the agent is trying to optimize. You can replace it with any function you like, but make sure that it's a function of the environment's state.

    CustomEnv: This is a custom Gym environment where the agent learns to interact. You can modify its action and observation spaces, as well as the reward function (currently, the reward is the value of the target function).
```

## Output

The trained model will be saved in the current directory under the name "agent_model.zip". You can load this model later to visualize its performance or further training.

Additionally, the final values of the optimized parameters (the state of the environment) will be printed on the console.

Tensorboard logs are saved in the respective directories ("./ppo_tensorboard/" or "./a2c_tensorboard/"), you can visualize them using:

```bash
tensorboard --logdir ./ppo_tensorboard/
```

or

```bash
tensorboard --logdir ./a2c_tensorboard/
```
