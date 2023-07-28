import gym
from gym import spaces
import numpy as np
import argparse
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Define the function you want to optimize
# This represents a 3D parabolic surface
def target_function(x):
    return -((x[0] - 3) **2 + (x[1] + 3)**2)

# Define your custom environment
class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))

        # Initialize state
        self.state = np.zeros(2)

    def step(self, action):
        self.state += action
        reward = target_function(self.state)
        done = False

        return self.state, reward, done, {}

    def reset(self):
        self.state = np.zeros(2)
        return self.state

def get_model(model_name, env):
    if model_name.lower() == "ppo":
        return PPO('MlpPolicy', env, verbose=1, tensorboard_log="./ppo_tensorboard/")
    elif model_name.lower() == "a2c":
        return A2C('MlpPolicy', env, verbose=1, tensorboard_log="./a2c_tensorboard/")
    else:
        raise ValueError(f"Unknown model name {model_name}, please choose either 'ppo' or 'a2c'")

def main():
    parser = argparse.ArgumentParser(description="Train an agent to maximize a function.")
    parser.add_argument("--model", help="The model to use, either 'ppo' or 'a2c'", type=str, default="ppo")
    parser.add_argument("--max_timesteps", help="The maximum number of timesteps to train for", type=int, default=60000)
    args = parser.parse_args()

    # Create the environment
    env = CustomEnv()

    # Make it compatible with Stable Baselines3
    env = DummyVecEnv([lambda: env])

    # Instantiate the agent
    model = get_model(args.model, env)

    # Train the agent
    model.learn(total_timesteps=args.max_timesteps)

    # Print the final values of the parameters
    print("Final values of the parameters: ", env.envs[0].state)

    # Test the trained agent
    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        # print(action)
        obs, _, _, _ = env.step(action)
        # print(obs)
        # print()

    # Save the model
    model.save("agent_model")

if __name__ == "__main__":
    main()

