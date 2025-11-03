import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import time

#Fonction de vérification de la divergence nulle des matrices de vitesse
def Verif_divergence_nulle(U, V, dx, dy):
    div = (np.roll(U, -1, axis=1) - U) / dx + (np.roll(V, -1, axis=0) - V) / dy
    #print(div)
    if np.any(np.abs(div) > 1e-10):
        indices = np.argwhere(np.abs(div) > 1e-10)
        for i, j in indices:
            print(f"Divergence non nulle détectée en ({i},{j}): {div[i,j]}")
        return False
    print("Divergence nulle vérifiée pour toutes les cellules.")
    return True


#Flag pour éviter de vérifier plusieurs fois la stabilité dans une même exécution
flag_stability_checked = False

#Fonction de vérification de la stabilité du schéma
def stabilite_schema(borne, nbre_courant):
    global flag_stability_checked
    if not flag_stability_checked:
        if nbre_courant > borne:
            print("SCHEMA INSTABLE !")
        else:
            print("Condition de stabilité vérifiée.")
        flag_stability_checked = True




#Fonction de reconstruction constante des concentrations aux faces des volumes finis
def construction_constante(C, U, V, dx, dy):
    C_bords_x = np.where(U > 0, np.roll(C, 1, axis=1), C)
    C_bords_y = np.where(V > 0, np.roll(C, 1, axis=0), C)
    return C_bords_x, C_bords_y


#Fonction de reconstruction linéaire des concentrations aux faces des volumes finis
def construction_linéaire(C, U, V, dx, dy):
    C_bords_x = np.where(U > 0,
        np.roll(C, 1, axis=1) + (C - np.roll(C, 2, axis=1)) / (2*dx) * (dx/2),
        C - (np.roll(C, -1, axis=1) - np.roll(C, 1, axis=1)) / (2*dx) * (dx/2)
    )
    C_bords_y = np.where(V > 0,
        np.roll(C, 1, axis=0) + (C - np.roll(C, 2, axis=0)) / (2*dy) * (dy/2),
        C - (np.roll(C, -1, axis=0) - np.roll(C, 1, axis=0)) / (2*dy) * (dy/2)
    )
    return C_bords_x, C_bords_y




#fonction résolution du problème de transport avec un schéma d'Euler explicite
def Euler_explicite(F_construction, C, C_RHS, U, V, dt, dx, dy, nbre_courant):

    #vérification de la stabilité du schéma
    stabilite_schema(1, nbre_courant)

    #matrices des concentrations aux bords
    C_bords_x, C_bords_y = F_construction(C_RHS, U, V, dx, dy)

    #mise à jour de la concentration
    C_new = C - dt* ((np.roll(U * C_bords_x, -1, axis=1) - U * C_bords_x)/dx + (np.roll(V * C_bords_y, -1, axis=0) - V * C_bords_y)/dy)
    return C_new


#fonction résolution du problème de transport avec un schéma de Runge-Kutta d'ordre 2
def Runge_Kutta_2(F_construction, C, C_RHS, U, V, dt, dx, dy, nbre_courant):

    #vérification de la stabilité du schéma
    stabilite_schema(1, nbre_courant)

    #Calcul de C_temp1
    C_temp1 = Euler_explicite(F_construction, C, C, U, V, dt, dx, dy, nbre_courant)

    #Calcul de C_temp2
    C_temp2 = Euler_explicite(F_construction, C, C_temp1, U, V, dt, dx, dy, nbre_courant)

    #mise à jour de la concentration avec la moyenne de C_temp1 et C_temp2 en prenant la moitié du correcteur et du prédicteur pour avoir le 2nd ordre de précision
    C_new = (C_temp1 + C_temp2)/2
    return C_new


def rotation_probleme(C, U, V, angle_degrees):
    if angle_degrees == 90:
        C_rotated = np.rot90(C, 3)
        U_rotated = np.rot90(U, 3)
        V_rotated = np.rot90(V, 3)
        U_rotated, V_rotated = -V_rotated, U_rotated
        return C_rotated, U_rotated, V_rotated
    elif angle_degrees == 180:
        C_rotated = np.rot90(C, 2)
        U_rotated = np.rot90(U, 2)
        V_rotated = np.rot90(V, 2)
        U_rotated, V_rotated = -U_rotated, -V_rotated
        return C_rotated, U_rotated, V_rotated
    elif angle_degrees == 270:
        C_rotated = np.rot90(C, 1)
        U_rotated = np.rot90(U, 1)
        V_rotated = np.rot90(V, 1)
        U_rotated, V_rotated = V_rotated, -U_rotated
        return C_rotated, U_rotated, V_rotated
    else:
        print("Angle de rotation non supporté. Utilisez 90, 180 ou 270 degrés.")
        return C, U, V


#Fonction de création d'un domaine
def domaine(n_x, n_y, dx, dy):
    x = np.linspace(-n_x*dx/2, n_x*dx/2, n_x)
    y = np.linspace(-n_y*dy/2, n_y*dy/2, n_y)
    X, Y = np.meshgrid(x, y)
    return X, Y


#Fonction pour créer un champ de vitesse uniforme
def vitesse_uniforme(n_x, n_y, u_value, v_value):
    U = np.full((n_y, n_x), u_value)
    V = np.full((n_y, n_x), v_value)
    return U, V


#Fonction de création d'un champ de vitesse à partir de la fonction de courant sin(x)
def vitesse_moyenne_bords(fct_courant, n_x, n_y, dx, dy):
    X, Y = domaine(n_x, n_y, dx, dy)
    U_moyen_bords = (fct_courant(X - dx/2, Y + dy/2) - fct_courant(X - dx/2, Y - dy/2))/dy
    V_moyen_bords = -(fct_courant(X + dx/2, Y - dy/2) - fct_courant(X - dx/2, Y - dy/2))/dx
    return U_moyen_bords, V_moyen_bords


def courant_sincos(x, y):
    return np.sin(x)*np.cos(y)


def courant_carrés(x, y):
    return -((x)**2 + (y)**2)/2


#Fonctions de création des conditions initiales de concentration
def concentration_uniforme(n_x, n_y, value):
    C = np.full((n_y, n_x), value)
    return C


def point_concentration_centre(n_x, n_y, value):
    C = np.zeros((n_y, n_x))
    C[int((n_y-1)/2),int((n_x-1)/2)] = value
    return C


def concentration_gaussienne(n_x, n_y, dx, dy, x0, y0, sigma):
    X, Y = domaine(n_x, n_y, dx, dy)
    C = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
    return C




#Fonction d'animation
def animate_concentration(C_evolution, dt, n_x, n_y, dx, dy, interval = 100, save=False, filename="animation.mp4"):
    fig, ax = plt.subplots()
    im = ax.imshow(
        C_evolution[0],
        cmap='viridis',
        origin='lower',
        extent=(-(n_x*dx)/2, (n_x*dx)/2, -(n_y*dy)/2, (n_y*dy)/2)
    )
    plt.colorbar(im, ax=ax)
    ax.set_title(f'Evolution de la concentration — Temps : 0.00')

    #ax.set_xlim(-(n_x*dx)/2, n_x*dx/2)
    #ax.set_ylim(-n_y*dy/2, n_y*dy/2)
    def update(frame):
        im.set_array(C_evolution[frame])
        time = frame * dt
        ax.set_title(f"Evolution de la concentration — Temps : {time:.2f}")
        return [im]

    anim = FuncAnimation(fig, update, frames=len(C_evolution), interval = interval, blit=False)
    if save:
        anim.save(filename)
    else:
        plt.show()
    return anim




#exécution du code
def execution(dx, dy, C, U, V, nbre_courant, n_t, F_construction, F_schéma):

    start = time.time()  # Démarre le chrono

    #Vérification de la divergence nulle du champ de vitesse
    Verif_divergence_nulle(U, V, dx, dy)

    #Calcul du dt à partir du nombre de courant
    matrice_dt = nbre_courant / (np.abs(U)/dx + np.abs(V)/dy)
    dt = np.min(matrice_dt)

    #Liste pour stocker les états successifs de la matrice de concentration
    C_evolution = []
    C_evolution.append(C.copy())

    #itération sur le pas de temps pour faire évoluer la concentration
    for t in np.linspace(dt, n_t*dt, n_t):
        C = F_schéma(F_construction, C, C, U, V, dt, dx, dy, nbre_courant)
        C_evolution.append(C.copy())
    
    end = time.time()     # Arrête le chrono
    execution_time = end - start
    print(f"Temps d'exécution : {execution_time:.5f} secondes")

    #Animation de l'évolution de la concentration
    temps_animation = 6 # Durée totale de l'animation en secondes
    interval = (temps_animation / len(C_evolution)) * 1000  # Intervalle entre les frames en millisecondes
    animate_concentration(C_evolution, dt, C.shape[1], C.shape[0], dx, dy, interval=interval)

    #réinitialisation du flag de stabilité pour un nouvel appel à la fonction d'exécution
    global flag_stability_checked
    flag_stability_checked = False



if __name__ == "__main__":

    #nombre de points selon x et y
    n_x, n_y= 21, 41
    #pas selon x et y
    dx, dy = 1, 1

    #matrice des concentrations au centre des volumes finis
    C = np.zeros((n_y, n_x))
    #matrice des vitesses aux faces verticales des volumes finis
    U = np.zeros((n_y, n_x))
    #matrice des vitesses aux faces horizontales des volumes finis
    V = np.zeros((n_y, n_x))


    #condition initiale sur la concentration
    #C = point_concentration_centre(n_x, n_y, 3)
    C[5,5], C[30,5] = 1, 1
    
    #condition initiale sur les vitesses
    U, V = vitesse_uniforme(n_x, n_y, 1, 1)

    """
    C = concentration_gaussienne(n_x, n_y, dx, dy, 0, 0, 50)
    plt.imshow(C, cmap='viridis', origin='lower')
    plt.colorbar()
    plt.title("Condition initiale de la concentration")
    plt.show()

    #vitesses à divergence nulle
    U, V = vitesse_moyenne_bords(courant_sincos, n_x, n_y, dx, dy)
    X, Y = domaine(n_x, n_y, dx, dy)
    plt.figure()
    plt.quiver(X, Y, U, V, scale=50)
    plt.title("Champ de vitesses")
    plt.show()
    """


    #nombre de courant, nombre de pas, pas de temps
    nbre_courant = 1
    n_t = 20

    #Lancement de l'exécution du code avec les fonctions choisies
    execution(dx, dy, C, U, V, nbre_courant, n_t, construction_constante, Runge_Kutta_2)

    #rotation du problème de 90, 180 et 270 degrés
    C, U, V = rotation_probleme(C, U, V, 90)
    execution(dx, dy, C, U, V, nbre_courant, n_t, construction_constante, Runge_Kutta_2)

    C, U, V = rotation_probleme(C, U, V, 90)
    execution(dx, dy, C, U, V, nbre_courant, n_t, construction_constante, Runge_Kutta_2)

    C, U, V = rotation_probleme(C, U, V, 90)
    execution(dx, dy, C, U, V, nbre_courant, n_t, construction_constante, Runge_Kutta_2)

    #à faire :
    #- faire que la divergence soit nulle pour le champ de vitesse
    #- faire une fonction qui donne la précision du schéma utilisé en comparant avec la solution exacte (cas test)

