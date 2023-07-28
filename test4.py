import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Create environments
env = gym.make('CartPole-v1', render_mode="human")
env = DummyVecEnv([lambda: env])  # This one is for training

env_for_render = gym.make('CartPole-v1')  # This one is for rendering

# Instantiate the agent
model = PPO('MlpPolicy', env, verbose=1)

# Train the agent
model.learn(total_timesteps=10000)

# Save the agent
model.save("ppo_cartpole")

# Load the trained agent
model = PPO.load("ppo_cartpole")

# # Test the trained agent
# obs = env.reset()
# obs_for_render = env_for_render.reset()  # Reset the rendering environment
# for i in range(1000):
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)

#     # Render using the rendering environment
#     obs_for_render, _, _, _ = env_for_render.step(action)
#     env_for_render.render()

# n,
# n,
# # ...
# ...

# Test the trained agent
obs = env.reset()
obs_for_render = env_for_render.reset()  # Reset the rendering environment
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)

    # Render using the rendering environment
    package = env_for_render.step(action[0])  # Take the first element of action
    env_for_render.render()

