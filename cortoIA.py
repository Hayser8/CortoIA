import gymnasium as gym
import numpy as np
import random
from gym.envs.toy_text.frozen_lake import generate_random_map

alpha = 0.1        
gamma = 0.99        
epsilon = 1.0       
epsilon_decay = 0.995  
epsilon_min = 0.01  
num_episodes = 2000  

n_states = 16  
n_actions = 4      
Q = np.zeros((n_states, n_actions))

def create_env():
    """
    Crea un entorno de FrozenLake con un mapa aleatorio.
    generate_random_map genera un tablero de tamaño 4x4 con probabilidad p de tener
    un piso congelado (F) y (1-p) de tener un agujero (H), garantizando que el mapa sea resoluble.
    """
    random_map = generate_random_map(size=4, p=0.8)
    env = gym.make("FrozenLake-v1", desc=random_map, is_slippery=True)
    return env, random_map

for episode in range(num_episodes):
    env, _ = create_env()  
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
    env.close()

print("Entrenamiento finalizado.")

num_test_episodes = 5
for episode in range(num_test_episodes):
    env, random_map = create_env()
    print(f"\nEpisodio de evaluación {episode + 1}:")
    print("Mapa generado:")
    for row in random_map:
        print("".join(row))
    
    state, info = env.reset()
    done = False
    env.render()
    
    while not done:
        action = np.argmax(Q[state])
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()
    
    print("Recompensa obtenida:", reward)
    env.close()
