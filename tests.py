import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


'''fig, ax = plt.subplots()
matrice = np.random.rand(10, 10)
im = ax.imshow(matrice, cmap='viridis')

def update(frame):
    # Ici, tu modifies la matrice à chaque étape
    nouvelle_matrice = np.random.rand(10, 10)  # exemple : matrice aléatoire
    im.set_array(nouvelle_matrice)
    return [im]

ani = FuncAnimation(fig, update, frames=100, interval=10, blit=True)
plt.show()'''

#A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#print(f"Divergence non nulle détectée en ({i},{j}): {div}")
dt = 0.5
n_t = 3
test1 = np.linspace(dt, dt*n_t, n_t)
test2 = np.arange(dt, dt*n_t + dt, dt)
print(test1)
print(test2)
#print(np.arange(1, 1+3*2, 2))
