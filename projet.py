import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import time



#Fonction de vérification de la divergence nulle des matrices de vitesse
def Verif_divergence_nulle(U, V, dx, dy):
    div = (np.roll(U, -1, axis=1) - U)/dx + (np.roll(V, -1, axis=0) - V)/dy
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



#Fonction de rotation du problème de 90°
def rotation_90_degre_probleme(C, U, V):

    # rotation antihoraire 90°
    C_rot = np.rot90(C, 1)
    U_rot = np.rot90(-V, 1)
    V_rot = np.rot90(U, 1)

    # réalignement des faces
    U_rot = np.roll(U_rot, -1, axis=0)
    return C_rot, U_rot, V_rot



#Fonction de création d'un domaine
def domaine_periodique(x_min, x_max, y_min, y_max, n_x, n_y):
    # endpoint=False : on ne répète pas x_max (utile pour domaine périodique)
    x = np.linspace(x_min, x_max, n_x, endpoint= False)
    y = np.linspace(y_min, y_max, n_y, endpoint= False)
    X, Y = np.meshgrid(x, y)
    return X, Y


#Fonction pour créer un champ de vitesse uniforme
def vitesse_uniforme(n_x, n_y, u_value, v_value):
    U = np.full((n_y, n_x), u_value)
    V = np.full((n_y, n_x), v_value)
    return U, V


#Fonctions de création des champs de vitesse aux bords périodiques et à divergence nulle
def vitesse_sincos(x, y, dx, dy):
    x_face = x - dx/2
    y_face = y - dy/2
    U = np.sin(x_face) * np.cos(y)
    V = -np.cos(x) * np.sin(y_face)
    return U, V

def vitesse_carrés(x, y, dx, dy):
    U = -y
    V = x
    return U, V


#Fonctions de création de champs de concentration
def point_concentration(X, Y, x0, y0, value):
    C = np.zeros_like(X)
    # Trouve les indices du point de grille le plus proche de (x0, y0)
    i = np.argmin(np.abs(Y[:, 0] - y0))
    j = np.argmin(np.abs(X[0, :] - x0))
    C[i, j] = value
    return C

def concentration_gaussienne(x, y, x_0, y_0, sigma):
    C = np.exp(-((x-x_0)**2 + (y-y_0)**2) / (2 * sigma**2))
    return C




#Fonction d'animation
def animate_concentration(C_evolution, dt, n_x, n_y, dx, dy, interval, save=False, filename="animation.mp4"):
    fig, ax = plt.subplots()
    im = ax.imshow(
        C_evolution[0],
        cmap='viridis',
        origin='lower',
        extent=(0, n_x*dx, 0, n_y*dy)
    )
    plt.colorbar(im, ax=ax)
    ax.set_title(f'Evolution de la concentration — Temps : 0.00')


    def update(frame):
        im.set_array(C_evolution[frame])
        time = frame * dt
        ax.set_title(f"Évolution de la concentration — Temps : {time:.2f}")
        return [im]

    anim = FuncAnimation(fig, update, frames=len(C_evolution), interval = interval, blit=False)
    if save:
        anim.save(filename)
    else:
        plt.show()
    return anim


def solution_analytique_point(X, Y, t, u0, v0, x0, y0, value, Lx, Ly):
    # Déplacement du pic avec conditions périodiques
    x_c = (x0 + u0 * t) % Lx
    y_c = (y0 + v0 * t) % Ly
    
    # On place la même amplitude au point de grille le plus proche
    C_ref = np.zeros_like(X)
    i = np.argmin(np.abs(Y[:,0] - y_c))
    j = np.argmin(np.abs(X[0,:] - x_c))
    C_ref[i, j] = value
    return C_ref


def residu(C_evolution, X, Y, U, V, dt, solution_analytique):
    n_t = len(C_evolution)
    residus = np.zeros(n_t)
    temps = np.arange(n_t) * dt

    for i, C_num in enumerate(C_evolution):
        C_ref = solution_analytique(X, Y, temps[i])
        residus[i] = np.sqrt(np.mean((C_num - C_ref)**2))

    

    # tracé de l'évolution du résidu
    plt.figure(figsize=(6,4))
    plt.plot(temps, residus, label='Résidu L2')
    plt.xlabel('Temps')
    plt.ylabel('Erreur L2')
    plt.title('Évolution du résidu par rapport à la solution analytique')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return residus




#exécution du code
def execution(dx, dy, C, U, V, nbre_courant, n_t, F_construction, F_schéma):
    # Démarre le chrono
    start = time.time()

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
    
    #Arrête le chrono
    end = time.time() 
    execution_time = end - start
    print(f"Temps d'exécution : {execution_time:.5f} secondes")

    #Animation de l'évolution de la concentration
    temps_animation = 10 # Durée totale de l'animation en secondes
    interval = (temps_animation / len(C_evolution)) * 1000  # Intervalle entre les frames en millisecondes
    animate_concentration(C_evolution, dt, C.shape[1], C.shape[0], dx, dy, interval=interval)

    #réinitialisation du flag de stabilité pour un nouvel appel à la fonction d'exécution
    global flag_stability_checked
    flag_stability_checked = False

    return C_evolution, dt



if __name__ == "__main__":

    #limites du domaine
    x_min, x_max, y_min, y_max = 0, 20, 0, 20
    #nombres de points du domaine
    n_x, n_y = 20, 20
    #pas selon x et y
    dx = (x_max-x_min)/n_x
    dy = (y_max-y_min)/n_y
    #nombre de courant et nombre de pas de temps
    nbre_courant = 1
    n_t = 100

    #création du domaine périodique
    X, Y = domaine_periodique(x_min, x_max, y_min, y_max, n_x, n_y)
    #création d’un champ de vitesse uniforme (transport vers la droite)
    u_value=1.0
    v_value=0.0
    U, V = vitesse_uniforme(n_x, n_y, u_value, v_value)
    # Création du point de concentration à la position choisie (centre du domainre ici)
    x0, y0 = 0.5 * (x_max - x_min), 0.5 * (y_max - y_min)
    c_value = 1.0
    C0 = point_concentration(X, Y, x0, y0, c_value)

    #définition de la solution analytique
    def sol_point(X, Y, t):
        return solution_analytique_point(X, Y, t,
                                         u_value, v_value,
                                         x0, y0,
                                         c_value,
                                         Lx=x_max-x_min, Ly=y_max-y_min)


    #démonstration d'un cas simple de déplacement d'un point de concentration
    print()
    print("nombre de courant = 1, construction constante et schéma Euler explicite :")
    C_evolution_point, dt = execution(dx, dy, C0, U, V, nbre_courant, n_t, construction_constante, Euler_explicite)

    #rotation des matrices de concentrations et vitesses prouvant que la solution est la même
    print()
    print("nombre de courant = 1, construction constante et schéma Euler explicite :")
    C90, U90, V90 =  rotation_90_degre_probleme(C0, U, V)
    C_evolution_point90, dt90 = execution(dx, dy, C90, U90, V90, nbre_courant, n_t, construction_constante, Euler_explicite)

    print()
    print("nombre de courant = 1, construction constante et schéma Euler explicite :")
    C180, U180, V180 =  rotation_90_degre_probleme(C90, U90, V90)
    C_evolution_point180, dt180 = execution(dx, dy, C180, U180, V180, nbre_courant, n_t, construction_constante, Euler_explicite)

    print()
    print("nombre de courant = 1, construction constante et schéma Euler explicite :")
    C270, U270, V270 =  rotation_90_degre_probleme(C180, U180, V180)
    C_evolution_point270, dt270 = execution(dx, dy, C270, U270, V270, nbre_courant, n_t, construction_constante, Euler_explicite)

    residu(C_evolution_point, X, Y, U, V, dt, sol_point)
    #résidu = 0 appuyant le fait que résolution du cas simple donne la soltution réelle si nbre_courant = 1
    #Par ailleurs, la solution est exacte car, à nombre de courant = 1, les erreurs spatiales et temporelles s'annulent ensemble (dans un cas de champ de vitesse uniforme)

    #construction linéaire et schéma Euler explicite
    print()
    print("nombre de courant = 1, construction linéaire et schéma Euler explicite :")
    C_evolution_point, dt = execution(dx, dy, C0, U, V, nbre_courant, n_t, construction_linéaire, Euler_explicite)
    residu(C_evolution_point, X, Y, U, V, dt, sol_point)
    #solution instable
    #raisons : 
    # - oscillations dû à la non présence de limiteur + concentration au départ non diffuse (point de concentration unique) augmentant encore les oscillations.
    # - différence d'ordre de précision entre les schémas spatiaux (ordre 2) et temporels (ordre 1).
    # - on est à la limite de stabilité = 1 du nombre de courant. Si le nombre de courant est diminuée, cela va restreindre les oscillations.

    print()
    print("nombre de courant = 1, construction constante et schéma Runge Kutta à 2 pas :")
    C_evolution_point, dt = execution(dx, dy, C0, U, V, nbre_courant, n_t, construction_constante, Runge_Kutta_2)
    residu(C_evolution_point, X, Y, U, V, dt, sol_point)
    #oscillations également présentes car :
    # - cas particulier de point de concentration unique
    # - différence différence d'ordre de précision, ici inversé par rapport au cas précédent, ordre 1 en spatial et 2 en temporel
    # - nombre de courant = 1


    print()
    print("nombre de courant = 1, construction linéaire et schéma Runge Kutta à 2 pas :")
    C_evolution_point, dt = execution(dx, dy, C0, U, V, nbre_courant, n_t, construction_linéaire, Runge_Kutta_2)
    residu(C_evolution_point, X, Y, U, V, dt, sol_point)
    # meilleure solution qu'en euler explicite mais tout de même présence d'oscillations. Un limiteur est requis pour atténuer grandement ces dernières.




    #cas ou nbre_courant = 0.5 et temps doublé pour avoir le même temps total qu'à nombre de courant = 1
    n_t = 200
    nbre_courant = 0.5
    print()
    print("nombre de courant = 0.5, construction constante et schéma Euler explicite :")
    C_evolution_point, dt = execution(dx, dy, C0, U, V, nbre_courant, n_t, construction_constante, Euler_explicite)
    residu(C_evolution_point, X, Y, U, V, dt, sol_point)
    #plus une solution exacte mais une diffusion de la solution dû aux termes d'erreurs au carré introduisant cette diffusion.

    print()
    print("nombre de courant = 0.5, construction linéaire et schéma Euler explicite :")
    C_evolution_point, dt = execution(dx, dy, C0, U, V, nbre_courant, n_t, construction_linéaire, Euler_explicite)
    residu(C_evolution_point, X, Y, U, V, dt, sol_point)
    #les instabilités apparaisent moins vite qu'à nombre de courant = 1 mais elles sont présentes

    print()
    print("nombre de courant = 0.5, construction constante et schéma Runge Kutta à 2 pas :")
    C_evolution_point, dt = execution(dx, dy, C0, U, V, nbre_courant, n_t, construction_constante, Runge_Kutta_2)
    residu(C_evolution_point, X, Y, U, V, dt, sol_point)
    #Solution diffuse n'oscillant plus. Le résidu est tout de même moins bon qu'en Euler explicite montrant une moins bonne précision quand on mélange les ordres de précision
    
    print()
    print("nombre de courant = 0.5, construction linéaire et schéma Runge Kutta à 2 pas :")
    C_evolution_point, dt = execution(dx, dy, C0, U, V, nbre_courant, n_t, construction_linéaire, Runge_Kutta_2)
    residu(C_evolution_point, X, Y, U, V, dt, sol_point)
    #Solution présentant la meilleure précision étant donné que le résidu augmente moins vite ce qui semble logique pour un 2e ordre en spatial et temporel.




    #Concentration et Champs de vitesses non-triviaux

    #limites du domaine
    x_min, x_max, y_min, y_max = 0, 2*np.pi, 0, 2*np.pi
    #nombres de points du domaine
    n_x, n_y = 200, 200
    #pas selon x et y
    dx = (x_max-x_min)/n_x
    dy = (y_max-y_min)/n_y
    #nombre de courant et nombre de pas de temps
    nbre_courant = 1
    n_t = 1000

    #création du domaine périodique
    X, Y = domaine_periodique(x_min, x_max, y_min, y_max, n_x, n_y)
    #création des champs de vitesses qui doivent être périodiques sur les limites du domaine
    U, V = vitesse_sincos(X, Y, dx, dy)
    #création de la matrice de concentration initiale
    C = concentration_gaussienne(X, Y, np.pi, np.pi, sigma = 20)

    #comparaison entre résolutions d'ordre 1 et d'ordre 2
    print()
    print("nombre de courant = 1, construction constante et Euler explicite :")
    C_evolution, dt = execution(dx, dy, C, U, V, nbre_courant, n_t, construction_constante, Euler_explicite)

    print()
    print("nombre de courant = 1, construction linéaire et Runge-kutta Runge Kutta à 2 pas:")
    C_evolution, dt = execution(dx, dy, C, U, V, nbre_courant, n_t, construction_linéaire, Runge_Kutta_2)
    #la solution d'ordre 2 semble en effet plus précise comme il y a moins de diffusion
