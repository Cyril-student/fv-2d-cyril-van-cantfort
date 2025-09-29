import matplotlib.pyplot as plt
import numpy as np

L_x = 50
L_y = 50
#dimensions du domaine de calcul
n_x = 50
n_y = 50
#nombre de points selon x et y
dx = L_x / n_x
dy = L_y / n_y
#pas selon x et y
T = 100
n_t = 200
dt = T / n_t
#durée totale, nombre de pas de temps, pas de temps

C = np.zeros((n_y, n_x))
#matrice des concentrations
U = np.ones((n_y, n_x))
#matrice des vitesses aux faces verticales des volumes finis
V = U
#matrice des vitesses aux faces horizontales des volumes finis

def Verif_divergence_nulle():
    for i in range(n_y):
        for j in range(n_x):
            if i == n_x-1:
                i = 0
            if j == n_y-1:
                j = 0
            
            if (U[i,j+1] - U[i,j])/dx + (V[i,j] - V[i+1,j])/dy != 0:
                print("Divergence non nulle !")
    #vérification de la divergence nulle des matrices de vitesse

Verif_divergence_nulle()


for t in np.linspace(1, T, n_t):
    for i in range(n_y):
        for j in range(n_x):
            if i == n_x-1:
                i = 0
            if j == n_y-1:
                j = 0
            #conditions aux limites périodiques

            #c_l =
            #c_r =
            #c_t =
            #c_b =
            u_l = U[i,j]
            u_r = U[i,j+1]
            v_t = V[i,j]
            v_b = V[i+1,j]
            #concentrations aux faces du volume fini

            C[i,j] = C[i,j] - dt/dx * (c_r*U[i,j+1] - c_l*U[i,j]) - dt/dy * (c_t*V[i,j] - c_b*V[i+1,j])
            #mise à jour de la concentration avec la méthode des volumes finis
    
    plt.imshow(C, cmap='viridis')
    plt.colorbar() 
    plt.show()
    plt.close()  
    # affiche la concentration à chaque pas de temps