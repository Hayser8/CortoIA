import gymnasium as gym
import numpy as np
import random

env = gym.make("FrozenLake-v1", is_slippery=True)

n_states = env.observation_space.n  
n_actions = env.action_space.n      
Q = np.zeros((n_states, n_actions))

alpha = 0.1         
gamma = 0.99         
epsilon = 1.0        
epsilon_decay = 0.995  
epsilon_min = 0.01   
num_episodes = 2000  

for episode in range(num_episodes):
    state, info = env.reset()
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        best_next_action = np.argmax(Q[next_state])
        td_target = reward + gamma * Q[next_state, best_next_action] * (1 - int(done))
        Q[state, action] += alpha * (td_target - Q[state, action])

        state = next_state

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

print("Entrenamiento finalizado.")

num_test_episodes = 5
for episode in range(num_test_episodes):
    state, info = env.reset()
    done = False
    print(f"\nEpisodio de evaluación {episode + 1}:")
    env.render()
    
    while not done:
        action = np.argmax(Q[state])
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()
    
    print("Recompensa obtenida:", reward)
