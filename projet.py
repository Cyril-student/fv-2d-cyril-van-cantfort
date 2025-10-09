import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

#Fonction de vérification de la divergence nulle des matrices de vitesse
def Verif_divergence_nulle(U, V, n_x, n_y, dx, dy):
    for i in range(n_y):
        for j in range(n_x):
            #Calcul de la divergence en (i,j) : div = du/dx + dv/dy
            div = (np.roll(U, -1, axis=1)[i,j] - U[i,j]) / dx + (np.roll(V, -1, axis=0)[i,j] - V[i,j]) / dy
            if abs(div) > 1e-10:
                print(f"Divergence non nulle détectée en ({i},{j}): {div}")
                return False
    print("Divergence nulle vérifiée pour toutes les cellules.")
    return True

#Fonction de reconstruction constante des concentrations aux faces des volumes finis
def construction_constante(C, U, V, i, j):
    if U[i,j] > 0:
        c_l = np.roll(C, 1, axis=1)[i,j]
    else:
        c_l = C[i,j]
    if np.roll(U, -1, axis=1)[i,j] > 0:
        c_r = C[i,j]
    else:
        c_r = np.roll(C, -1, axis=1)[i,j]
    if V[i,j] > 0:
        c_b = np.roll(C, 1, axis=0)[i,j]
    else:
        c_b = C[i,j]
    if np.roll(V, -1, axis=0)[i,j] > 0:
        c_t = C[i,j]
    else:
        c_t = np.roll(C, -1, axis=0)[i,j]
    return c_l, c_r, c_b, c_t

#Fonction de reconstruction linéaire des concentrations aux faces des volumes finis
def construction_linéaire(C, U, V, i, j):
    #reconstruction stable si le flux est décentré amont
    if U[i,j] > 0:
        #décalage de C avec np.roll pour bénéficier des conditions aux limites périodiques
        c_l = np.roll(C, 1, axis=1)[i,j] + (C[i,j] - np.roll(C, 2, axis=1)[i,j]) / (2*dx) * dx/2
    else:
        c_l = C[i,j] + (np.roll(C, -1, axis=1)[i,j] - np.roll(C, 1, axis=1)[i,j]) / (2*dx) * (-dx/2)
    if np.roll(U, -1, axis=1)[i,j] > 0:
        c_r = C[i,j] + (np.roll(C, -1, axis=1)[i,j] - np.roll(C, 1, axis=1)[i,j]) / (2*dx) * dx/2
    else:
        c_r = np.roll(C, -1, axis=1)[i,j] + (np.roll(C, -2, axis=1)[i,j] - C[i,j]) / (2*dx) * (-dx/2)
    if V[i,j] > 0:
        c_b = np.roll(C, 1, axis=0)[i,j] + (C[i,j] - np.roll(C, 2, axis=0)[i,j]) / (2*dy) * dy/2
    else:
        c_b = C[i,j] + (np.roll(C, -1, axis=0)[i,j] - np.roll(C, 1, axis=0)[i,j]) / (2*dy) * (-dy/2)
    if np.roll(V, -1, axis=0)[i,j] > 0:
        c_t = C[i,j] + (np.roll(C, -1, axis=0)[i,j] - np.roll(C, 1, axis=0)[i,j]) / (2*dy) * dy/2
    else:
        c_t = np.roll(C, -1, axis=0)[i,j] + (np.roll(C, -2, axis=0)[i,j] - C[i,j]) / (2*dy) * (-dy/2)
    return c_l, c_r, c_b, c_t

#résolution du problème de transport avec un schéma d'Euler explicite
def Euler_explicite(F_construction, C, U, V, dt, n_t, dx, dy, n_x, n_y):
    #Liste pour stocker les états successifs de la matrice de concentration
    C_evolution = []
    C_evolution.append(C.copy())

    #Vérification de la stabilité du schéma
    if dt * (max_u/dx + max_v/dy) > 1:
        print("SCHEMA INSTABLE !")
    else:
        print("Condition de stabilité vérifiée.")
    
    Verif_divergence_nulle(U, V, n_x, n_y, dx, dy)

    for t in np.arange(dt, dt*n_t, dt):
        for i in range(n_y):
            for j in range(n_x):
                #vitesses aux faces du volume fini
                u_l = U[i,j]
                v_b = V[i,j]
                u_r = np.roll(U, -1, axis=1)[i,j]
                v_t = np.roll(V, -1, axis=0)[i,j]
                #concentrations aux faces du volume fini
                c_l, c_r, c_b, c_t = F_construction(C, U, V, i, j)
                #mise à jour de la concentration
                C[i,j] = C[i,j] - dt/dx * (c_r*u_r - c_l*u_l) - dt/dy * (c_t*v_t - c_b*v_b)
        # Sauvegarde de l'état courant de la matrice de concentration
        C_evolution.append(C.copy())
    return C_evolution

#Fonction d'animation
def animate_concentration(C_evolution, interval=50, save=False, filename= "animation.mp4"):
    fig, ax = plt.subplots()
    im = ax.imshow(C_evolution[0], cmap='viridis', origin='lower')
    plt.colorbar(im, ax=ax)
    ax.set_title('Evolution de la concentration')

    def update(frame):
        im.set_array(C_evolution[frame])
        ax.set_xlabel(f"Step {frame}")
        return [im]

    anim = FuncAnimation(fig, update, frames=len(C_evolution), interval=interval, blit=True)
    if save:
        anim.save(filename)
    else:
        plt.show()
    return anim

#nombre de points selon x et y
n_x = 10
n_y = 10
#pas selon x et y
dx = 1
dy = 1

#matrice des concentrations au centre des volumes finis
C = np.zeros((n_y, n_x))
#matrice des vitesses aux faces verticales des volumes finis
U = np.zeros((n_y, n_x))
#matrice des vitesses aux faces horizontales des volumes finis
V = np.ones((n_y, n_x))

#nombre de courant, nombre de pas, pas de temps
nbre_courant = 1
n_t = 2
max_u = np.max(np.abs(U))
max_v = np.max(np.abs(V))
dt = nbre_courant/(max_u/dx + max_v/dy)
print(dt, np.linspace(0, 2, 2))

#condition initiale sur la concentration
#C[int(n_y/2)-1,:] = 1
C[int(n_y/3)-1:int(2*n_y/3)-1,int(n_x/2)-1] = 1
#condition initiale sur la vitesse horizontale
U = np.ones((n_y, n_x))
#condition initiale sur la vitesse verticale
V = np.zeros((n_y, n_x))

#exécution du code
if __name__ == "__main__":
    C_evolution = Euler_explicite(construction_constante, C, U, V, dt, n_t, dx, dy, n_x, n_y)
    animate_concentration(C_evolution, interval=1000/n_t, save=False)