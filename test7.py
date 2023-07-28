import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Define the function you want to optimize
def target_function(x):
    return -np.sum(np.square(x - 2))

# Define your custom environment
class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,))

        # Initialize state
        self.state = np.zeros(4)

    def step(self, action):
        self.state += action
        reward = target_function(self.state)
        done = False

        return self.state, reward, done, {}

    def reset(self):
        self.state = np.zeros(4)
        return self.state

# Create the environment
env = CustomEnv()

# Make it compatible with Stable Baselines3
env = DummyVecEnv([lambda: env])  

# Specify the TensorBoard log directory
log_dir = './tb_logs/'

# Instantiate the agent
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)

# Train the agent with tensorboard callback
model.learn(total_timesteps=10000)

# Print the final values of the parameters
print("Final values of the parameters: ", env.envs[0].state)

# Test the trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)

# Save the model
model.save("ppo_custom")

