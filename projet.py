import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

#dimensions du domaine de calcul
L_x = 50
L_y = 50
#nombre de points selon x et y
n_x = 50
n_y = 50
#pas selon x et y
dx = L_x / n_x
dy = L_y / n_y
#durée totale, pas de temps
T = 100
dt = 0.5

#matrice des concentrations au centre des volumes finis
C = np.zeros((n_y, n_x))
#matrice des vitesses aux faces verticales des volumes finis
U = np.ones((n_y, n_x))
#matrice des vitesses aux faces horizontales des volumes finis
V = np.zeros((n_y, n_x))

def Verif_stabilité():
    max_u = np.max(np.abs(U))
    max_v = np.max(np.abs(V))
    if dt * (max_u/dx + max_v/dy) > 1:
        print("Attention, condition de stabilité non vérifiée !")
    else:
        print("Condition de stabilité vérifiée.")

Verif_stabilité()

#condition initiale sur la concentration
C[:,10:20] = 1

#Liste pour stocker les états successifs de la matrice de concentration
C_evolution = []

#Fonction de vérification de la divergence nulle des matrices de vitesse
def Verif_divergence_nulle():
    for i in range(n_y):
        for j in range(n_x):
            #conditions aux limites périodiques
            if i == n_x-1:
                v_t = V[0,j]
            else:
                v_t = V[i+1,j]
            if j == n_y-1:
                u_r = U[i,0]
            else:
                u_r = U[i,j+1]
            u_l = U[i,j]
            v_b = V[i,j]
             
            if i==n_y-1 and j==n_x-1:
                if (U[i,0] - U[i,j])/dx + (V[0,j]- V[i,j])/dy != 0:
                    print("Divergence non nulle !")
            elif i==n_y-1:
                if (U[i,j+1] - U[i,j])/dx + (V[0,j]- V[i,j])/dy != 0:
                    print("Divergence non nulle !")
            elif j==n_x-1:
                if (U[i,0] - U[i,j])/dx + (V[i+1,j]- V[i,j])/dy != 0:
                    print("Divergence non nulle !")
            else:
                if (U[i,j+1] - U[i,j])/dx + (V[i+1,j]- V[i,j])/dy != 0:
                    print("Divergence non nulle !")

Verif_divergence_nulle()

C_evolution.append(C.copy())

def construction_constante():
    for t in np.arange(1, T, dt):
        for i in range(n_y):
            for j in range(n_x):
                #mise à jour de la concentration avec la dans un schéma Euler explicite
                C[i,j] = C[i,j] - dt/dx * (c_r*u_r - c_l*u_l) - dt/dy * (c_t*v_t - c_b*v_b)
#pas fini

def construction_lineaire():
    for t in np.arange(1, T, dt):
        for i in range(n_y):
            for j in range(n_x):
                #conditions aux limites périodiques
                if i == n_x-1:
                    v_t = V[0,j]
                else:
                    v_t = V[i+1,j]
                if j == n_y-1:
                    u_r = U[i,0]
                else:
                    u_r = U[i,j+1]
                u_l = U[i,j]
                v_b = V[i,j]

                #faces haut et bas
                #reconstruction constante des concentrations aux bornes du domaine aux faces des volumes finis
                if i == 0:
                    if v_b > 0:
                        c_b = C[n_y-1,j]
                    else:
                        c_b = C[i,j]
                    if v_t > 0:
                        c_t = C[i,j]
                    else:
                        c_t = C[i+1,j]

                elif i == n_y-1:
                    if v_b > 0:
                        c_b = C[i-1,j]
                    else:
                        c_b = C[i,j]
                    if v_t > 0:
                        c_t = C[i,j]
                    else:
                        c_t = C[0,j]
                #reconstruction linéaire à l'intérieur du domaine aux faces des volume finis
                else:
                    if v_b < 0:
                        c_b = C[i-1,j] + (C[i,j] - C[i-2,j]) / (2*dy) * dy/2
                    else:
                        c_b = C[i,j] + (C[i+1,j] - C[i-1,j]) / (2*dy) * (-dy/2)

                    if v_t > 0:
                        c_t = C[i,j] + (C[i+1,j] - C[i-1,j]) / (2*dy) * dy/2
                    else:
                        if i == n_y-2:
                            c_t = C[i,j]
                        else:
                            c_t = C[i+1,j] + (C[i+2,j] - C[i,j]) / (2*dy) * (-dy/2)

                #faces gauche et droite
                if j == 0:
                    if u_l > 0:
                        c_l = C[i,n_x-1]
                    else:
                        c_l = C[i,j]
                    if u_r > 0:
                        c_r = C[i,j] + (C[i,j+1] - C[i,j-1]) / (2*dx) * dx/2
                    else:
                        c_r = C[i,j+1] + (C[i,j+2] - C[i,j]) / (2*dx) * (-dx/2)
                    

                elif j == n_x-1:
                    if u_r > 0:
                        c_r = C[i,j]
                    else:
                        c_r = C[i,0]
                    if u_l > 0:
                        c_l = C[i,j-1] + (C[i,j] - C[i,j-2]) / (2*dx) * dx/2
                    else:
                        c_l = C[i,j] + (C[i,j+1] - C[i,j-1]) / (2*dx) * (-dx/2)
                
                else:
                    if u_l > 0:
                        c_l = C[i,j-1] + (C[i,j] - C[i,j-2]) / (2*dx) * dx/2
                    else:
                        c_l = C[i,j] + (C[i,j+1] - C[i,j-1]) / (2*dx) * (-dx/2)

                    if u_r > 0:
                        c_r = C[i,j] + (C[i,j+1] - C[i,j-1]) / (2*dx) * dx/2
                    else:
                        if j == n_x-2:
                            c_r = C[i,j]
                        else:
                            c_r = C[i,j+1] + (C[i,j+2] - C[i,j]) / (2*dx) * (-dx/2)


                #mise à jour de la concentration avec la dans un schéma Euler explicite
                C[i,j] = C[i,j] - dt/dx * (c_r*u_r - c_l*u_l) - dt/dy * (c_t*v_t - c_b*v_b)
        # Sauvegarde de l'état courant de la matrice de concentration
        C_evolution.append(C.copy())

# Fonction d'animation
def animate_concentration(C_evolution, interval=50, save=False, filename="animation.mp4"):
    fig, ax = plt.subplots()
    im = ax.imshow(C_evolution[0], cmap='viridis', origin='lower')
    ax.set_title('Evolution de la concentration')

    def update(frame):
        im.set_array(C_evolution[frame])
        ax.set_xlabel(f"Step {frame}")
        return [im]

    anim = FuncAnimation(fig, update, frames=len(C_evolution), interval=interval, blit=True)
    if save:
        anim.save(filename, writer='ffmpeg')
    else:
        plt.show()
    return anim

#construction_constante()

construction_lineaire()


# l'animation se lance si le script est exécuté directement
if __name__ == "__main__":
    animate_concentration(C_evolution, interval=50, save=False)