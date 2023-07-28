import gym
from gym import spaces
from stable_baselines3 import PPO
import numpy as np
import argparse
import os

class FunctionMaximizationEnv(gym.Env):
    def __init__(self):
        super(FunctionMaximizationEnv, self).__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.current_step = 0
        self.MAX_STEPS = 1000

    def step(self, action):
        self.current_step += 1
        self.state += action
        reward = np.sin(self.state[0]) * np.sin(self.state[1])  # 2D Sinusoidal function
        done = self.current_step >= self.MAX_STEPS
        return self.state, reward, done, {}

    def reset(self):
        self.state = np.random.uniform(-1.0, 1.0, size=(2,))
        self.current_step = 0
        return self.state

def main(args):
    log_dir = "./tmp/"
    os.makedirs(log_dir, exist_ok=True)

    env = FunctionMaximizationEnv()

    model = PPO('MlpPolicy', env, ent_coef=0.1, verbose=1)  # Removed action_noise
    model.learn(total_timesteps=args.max_iter)
    model.save("ppo_func_max")


    # Test the trained agent
    obs = env.reset()
    obs_for_render = env_for_render.reset()  # Reset the rendering environment

    # Create a list to store the states
    states = []

    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)

        # Add the state to the list
        states.append(np.array(obs))

        # Render using the rendering environment
        obs_for_render, reward, done, *rest = env_for_render.step(action[0])  # Take the first element of action
        env_for_render.render()
        if done:
            obs_for_render = env_for_render.reset()
            break  # End the loop when the episode is done

    # Print the states that led to the end of the episode
    for i, state in enumerate(states):
        print(f"State {i + 1}: {state}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_iter', type=int, default=30000, help="Maximum number of iterations")
    args = parser.parse_args()
    main(args)

