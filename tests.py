import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


fig, ax = plt.subplots()
matrice = np.random.rand(10, 10)
im = ax.imshow(matrice, cmap='viridis')

def update(frame):
    # Ici, tu modifies la matrice à chaque étape
    nouvelle_matrice = np.random.rand(10, 10)  # exemple : matrice aléatoire
    im.set_array(nouvelle_matrice)
    return [im]

ani = FuncAnimation(fig, update, frames=100, interval=10, blit=True)
plt.show()