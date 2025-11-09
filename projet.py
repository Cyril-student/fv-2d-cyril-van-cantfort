import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import time



#Fonction de vérification de la divergence nulle des matrices de vitesse
def Verif_divergence_nulle(U, V, dx, dy):
    div = (np.roll(U, -1, axis=1) - U)/dx + (np.roll(V, -1, axis=0) - V)/dy #calcul de la divergence des mailles du domaine
    if np.any(np.abs(div) > 1e-10):
        indices = np.argwhere(np.abs(div) > 1e-10)
        for i, j in indices:
            print(f"Divergence non nulle détectée en ({i},{j}): {div[i,j]}") #indique les emplacements des mailles où la divergence est non-nulle
        return False 
    print("Divergence nulle vérifiée pour toutes les cellules.")
    return True

#Flag pour éviter de vérifier plusieurs fois la stabilité dans une même exécution du code
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




#fonction de résolution du problème de transport avec un schéma d'Euler explicite
def Euler_explicite(F_construction, C, C_RHS, U, V, dt, dx, dy, nbre_courant):

    #vérification de la stabilité du schéma
    stabilite_schema(1, nbre_courant)

    #matrices des concentrations aux bords bas et gauches des mailles
    C_bords_x, C_bords_y = F_construction(C_RHS, U, V, dx, dy)

    #mise à jour de la concentration
    C_new = C - dt* ((np.roll(U * C_bords_x, -1, axis=1) - U * C_bords_x)/dx + (np.roll(V * C_bords_y, -1, axis=0) - V * C_bords_y)/dy)
    return C_new


#fonction résolution du problème de transport avec un schéma de Runge-Kutta d'ordre 2
def Runge_Kutta_2(F_construction, C, C_RHS, U, V, dt, dx, dy, nbre_courant):

    #vérification de la stabilité du schéma
    stabilite_schema(1, nbre_courant)

    #Calcul du prédicteur
    C_temp1 = Euler_explicite(F_construction, C, C, U, V, dt, dx, dy, nbre_courant)

    #Calcul de correcteur
    C_temp2 = Euler_explicite(F_construction, C, C_temp1, U, V, dt, dx, dy, nbre_courant)

    #mise à jour de la concentration avec la moyenne de C_temp1 et C_temp2 en prenant la moitié du correcteur et du prédicteur pour avoir le 2nd ordre de précision
    C_new = (C_temp1 + C_temp2)/2
    return C_new



#Fonction de rotation du problème de 90°
def rotation_90_degre_probleme(C, U, V):

    # rotation antihoraire 90°
    C_rot = np.rot90(C, -1)
    U_rot = np.rot90(-V, -1)
    V_rot = np.rot90(U, -1)


    #réalignement des vitesses aux faces pour conserver la divergence nulle
    V_rot = np.roll(V_rot, -1, axis=1)
    return C_rot, U_rot, V_rot



#Fonction de création d'un domaine
def domaine_periodique(x_min, x_max, y_min, y_max, n_x, n_y):
    #endpoint=False : on ne répète pas x_max (utile pour domaine périodique)
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
    #Trouve les indices du point de grille le plus proche de (x0, y0)
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
        origin = 'lower',
        cmap='viridis',
        extent=(0, n_x*dx, 0, n_y*dy)
    )
    plt.colorbar(im, ax=ax)
    ax.set_title(f'Evolution de la concentration — Temps : 0.00')

    #fonction de mise à jour d'image nécessaire pour FuncAnimation
    def update(frame):
        im.set_array(C_evolution[frame])
        time = frame * dt
        ax.set_title(f"Évolution de la concentration — Temps : {time:.2f}") #affichage du temps réel dans l'animation
        return [im]

    anim = FuncAnimation(fig, update, frames=len(C_evolution), interval = interval, blit=False)
    if save:
        anim.save(filename)
    else:
        plt.show()
    return anim



#fonction permettant de visualiser les différences de précision entre les schémas spatiaux
def precision():

    #création des arrays pour le plot des erreurs
    abscisse_dx = np.zeros(5)
    residus_const = np.zeros(5) #erreur construction constante
    residus_lin = np.zeros(5) #erreur construction linéaire

    #limites du domaine
    x_min, x_max, y_min, y_max = 0, 160, 0, 160
    #nombres de points du domaine
    n_x, n_y = x_max, y_max
    #pas selon x et y. On prend dx et dy = 1 ici
    dx = (x_max-x_min)/n_x
    dy = (y_max-y_min)/n_y
    print()
    print("dx, dy :" , dx, dy)
    #nombre de courant et nombre de pas de temps. On choisit dt et n_t de sorte à ce que la concentration de fin soit celle de départ donc la concentration fait juste un "tour" dû au transport.
    nbre_courant = 1
    n_t = n_x

    #création du domaine périodique
    X, Y = domaine_periodique(x_min, x_max, y_min, y_max, n_x, n_y)

    #création d’un champ de vitesse uniforme (transport vers la droite).
    u_value=1.0
    v_value=0.0
    U, V = vitesse_uniforme(n_x, n_y, u_value, v_value)

    #création de la matrice de concentration initiale. Une gaussienne est choisie pour éviter d'introduire des erreurs liés à des changements de concentration trop importants d'une cellule à l'autre
    C = concentration_gaussienne(X, Y, x_max/2, y_max/2, sigma = 30)

    #calcul du transport de la concentration pour comparer avec la solution exacte avec un schéma Runge kutta à 2 pas pour avoir la précision d'ordre 2 en temps et ainsi comparer les précision en spatial sans risque d'instabilité
    #construction constante
    C_evolution_const, dt, anim = execution(dx, dy, C, U, V, nbre_courant, n_t, construction_constante, Runge_Kutta_2)
    #construction linéaire
    C_evolution_lin, dt, anim = execution(dx, dy, C, U, V, nbre_courant, n_t, construction_linéaire, Runge_Kutta_2)

    #sauvegarde du dx et de la moyenne de l'erreur en L2
    abscisse_dx[0] = dx
    residus_const[0] = np.mean(((C_evolution_const[len(C_evolution_const) - 1] - C))**2)
    residus_lin[0] = np.mean((C_evolution_lin[len(C_evolution_lin) - 1] - C)**2)


    #dx et dy = 2
    n_x, n_y = int(n_x/2), int(n_y/2)
    dx = (x_max-x_min)/n_x
    dy = (y_max-y_min)/n_y
    print()
    print("dx, dy :" , dx, dy)

    n_t = n_x

    X, Y = domaine_periodique(x_min, x_max, y_min, y_max, n_x, n_y)

    u_value=1.0
    v_value=0.0
    U, V = vitesse_uniforme(n_x, n_y, u_value, v_value)

    C = concentration_gaussienne(X, Y, x_max/2, y_max/2, sigma = 30)

    C_evolution_const, dt, anim = execution(dx, dy, C, U, V, nbre_courant, n_t, construction_constante, Runge_Kutta_2)
    C_evolution_lin, dt, anim = execution(dx, dy, C, U, V, nbre_courant, n_t, construction_linéaire, Runge_Kutta_2)

    abscisse_dx[1] = dx
    residus_const[1] = np.mean(((C_evolution_const[len(C_evolution_const) - 1] - C))**2)
    residus_lin[1] = np.mean((C_evolution_lin[len(C_evolution_lin) - 1] - C)**2)


    #dx et dy = 4
    n_x, n_y = int(n_x/2), int(n_y/2)
    dx = (x_max-x_min)/n_x
    dy = (y_max-y_min)/n_y
    print()
    print("dx, dy :" , dx, dy)

    n_t = n_x


    X, Y = domaine_periodique(x_min, x_max, y_min, y_max, n_x, n_y)

    u_value=1.0
    v_value=0.0
    U, V = vitesse_uniforme(n_x, n_y, u_value, v_value)

    C = concentration_gaussienne(X, Y, x_max/2, y_max/2, sigma = 30)

    C_evolution_const, dt, anim = execution(dx, dy, C, U, V, nbre_courant, n_t, construction_constante, Runge_Kutta_2)
    C_evolution_lin, dt, anim = execution(dx, dy, C, U, V, nbre_courant, n_t, construction_linéaire, Runge_Kutta_2)

    abscisse_dx[2] = dx
    residus_const[2] = np.mean(((C_evolution_const[len(C_evolution_const) - 1] - C))**2)
    residus_lin[2] = np.mean((C_evolution_lin[len(C_evolution_lin) - 1] - C)**2)


    #dx et dy = 8
    n_x, n_y = int(n_x/2), int(n_y/2)
    dx = (x_max-x_min)/n_x
    dy = (y_max-y_min)/n_y
    print()
    print("dx, dy :" , dx, dy)
    n_t = n_x

    X, Y = domaine_periodique(x_min, x_max, y_min, y_max, n_x, n_y)

    u_value=1.0
    v_value=0.0
    U, V = vitesse_uniforme(n_x, n_y, u_value, v_value)

    C = concentration_gaussienne(X, Y, x_max/2, y_max/2, sigma = 30)

    C_evolution_const, dt, anim = execution(dx, dy, C, U, V, nbre_courant, n_t, construction_constante, Runge_Kutta_2)
    C_evolution_lin, dt, anim = execution(dx, dy, C, U, V, nbre_courant, n_t, construction_linéaire, Runge_Kutta_2)

    abscisse_dx[3] = dx
    residus_const[3] = np.mean(((C_evolution_const[len(C_evolution_const) - 1] - C))**2)
    residus_lin[3] = np.mean((C_evolution_lin[len(C_evolution_lin) - 1] - C)**2)


    #dx et dy = 16
    print(n_x/2)
    n_x, n_y = int(n_x/2), int(n_y/2)
    dx = (x_max-x_min)/n_x
    dy = (y_max-y_min)/n_y
    print()
    print("dx, dy :" , dx, dy)
    n_t = n_x

    X, Y = domaine_periodique(x_min, x_max, y_min, y_max, n_x, n_y)

    u_value=1.0
    v_value=0.0
    U, V = vitesse_uniforme(n_x, n_y, u_value, v_value)

    C = concentration_gaussienne(X, Y, x_max/2, y_max/2, sigma = 30)

    C_evolution_const, dt, anim = execution(dx, dy, C, U, V, nbre_courant, n_t, construction_constante, Runge_Kutta_2)
    C_evolution_lin, dt, anim = execution(dx, dy, C, U, V, nbre_courant, n_t, construction_linéaire, Runge_Kutta_2)

    abscisse_dx[4] = dx
    residus_const[4] = np.mean(((C_evolution_const[len(C_evolution_const) - 1] - C))**2)
    residus_lin[4] = np.mean((C_evolution_lin[len(C_evolution_lin) - 1] - C)**2)


    #plot de l'erreur des constructions constante et linéaire
    fig = plt.figure()
    plt.plot(abscisse_dx, residus_const, label = "construction constante")
    plt.plot(abscisse_dx, residus_lin, label = "construction linéaire")
    plt.xlabel('pas spatial dx')
    plt.ylabel('Erreur L2 moyenne')
    plt.title('Évolution de l\'erreur de la solution calculée')
    plt.grid(True)
    plt.legend(loc = 'best')
    plt.tight_layout()

    return fig



#exécution du code d'évolution de la concentration
def execution(dx, dy, C, U, V, nbre_courant, n_t, F_construction, F_schéma):
    #Démarre le chrono pour connaitre le temps de calcul de la solution
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
    
    #Arrête le chrono afin de montrer le temps necessaire au calcul de l'évolution de la matrice de concentration sans prendre en compte l'animation
    end = time.time() 
    execution_time = end - start
    print(f"Temps d'exécution : {execution_time:.5f} secondes")

    #Animation de l'évolution de la concentration
    temps_animation = 10 # Durée totale de l'animation en secondes
    interval = (temps_animation / len(C_evolution)) * 1000  #Intervalle entre les frames en millisecondes
    anim = animate_concentration(C_evolution, dt, C.shape[1], C.shape[0], dx, dy, interval=interval)

    #réinitialisation du flag de stabilité pour un nouvel appel à la fonction d'exécution
    global flag_stability_checked
    flag_stability_checked = False

    return C_evolution, dt, anim



if __name__ == "__main__":

    #limites du domaine
    x_min, x_max, y_min, y_max = 0, 10, 0, 10
    #nombres de points du domaine
    n_x, n_y = 10, 10
    #pas selon x et y
    dx = (x_max-x_min)/n_x
    dy = (y_max-y_min)/n_y
    print()
    print("dx, dy :" , dx, dy)
    #nombre de courant et nombre de pas de temps
    nbre_courant = 1
    n_t = 100

    #création du domaine
    X, Y = domaine_periodique(x_min, x_max, y_min, y_max, n_x, n_y)
    #création d’un champ de vitesse uniforme (transport vers la droite)
    u_value=1.0
    v_value=0.0
    U, V = vitesse_uniforme(n_x, n_y, u_value, v_value)
    # Création du point de concentration à la position choisie (centre du domaine ici ou la maille la plus proche du centre)
    x0, y0 = 0.5 * (x_max - x_min), 0.5 * (y_max - y_min)
    c_value = 1.0
    C0 = point_concentration(X, Y, x0, y0, c_value)

    #démonstration d'un cas simple de déplacement d'un point de concentration
    print()
    print("nombre de courant = 1, construction constante et schéma Euler explicite :")
    C_evolution_point, dt, anim = execution(dx, dy, C0, U, V, nbre_courant, n_t, construction_constante, Euler_explicite)
    #La solution du problème de transport ici est exacte comme on est en construction constante avec un schéma Euler explicite et à nombre de courant = 1 dans un champ de vitesse uniforme
    #dans ce cas précis, les erreurs spatiales et temporelles s'annulent entre elles nous retournant cette solution sans erreur

    #rotation des matrices de concentrations et vitesses prouvant que la solution est la même quand le problème est tourné
    print()
    print("nombre de courant = 1, construction constante et schéma Euler explicite :")
    C90, U90, V90 =  rotation_90_degre_probleme(C0, U, V)
    C_evolution_point90, dt90, anim90 = execution(dx, dy, C90, U90, V90, nbre_courant, n_t, construction_constante, Euler_explicite)

    print()
    print("nombre de courant = 1, construction constante et schéma Euler explicite :")
    C180, U180, V180 =  rotation_90_degre_probleme(C90, U90, V90)
    C_evolution_point180, dt180, anim180 = execution(dx, dy, C180, U180, V180, nbre_courant, n_t, construction_constante, Euler_explicite)

    print()
    print("nombre de courant = 1, construction constante et schéma Euler explicite :")
    C270, U270, V270 =  rotation_90_degre_probleme(C180, U180, V180)
    C_evolution_point270, dt270, anim270 = execution(dx, dy, C270, U270, V270, nbre_courant, n_t, construction_constante, Euler_explicite)

    #changement du nombre de courant pour montrer qu'il doit être inférieur à 1 pour la stabilité du schéma en Euler explicite
    nbre_courant = 1.1
    print()
    print("nombre de courant = 1.1, construction constante et schéma Euler explicite :")
    C_evolution_point, dt, anim= execution(dx, dy, C0, U, V, nbre_courant, n_t, construction_constante, Euler_explicite)

    #en construction linéaire, la solution est instable en schéma Euler explicite même à nombre de courant < 1
    #l'absence de limiteur, la concentration initale étant une impulsion en un point du domaine et le mélange des précisions d'ordre 1 et 2 peuvent expliquer ce phénomène
    nbre_courant = 0.5
    print()
    print("nombre de courant = 0.5, construction linéaire et schéma Euler explicite :")
    C_evolution_point, dt, anim = execution(dx, dy, C0, U, V, nbre_courant, n_t, construction_linéaire, Euler_explicite)

    #Le schéma temporel Runge Kutta à 2 pas est également stable pour des nombres de courant ne dépassant pas 1
    nbre_courant = 1
    print()
    print("nombre de courant = 1, construction linéaire et schéma Runge Kutta à 2 pas :")
    C_evolution_point, dt, anim = execution(dx, dy, C0, U, V, nbre_courant, n_t, construction_linéaire, Runge_Kutta_2)

    #ceci montre l'instabilité du schéma pour un nombre de courant supérieur à 1
    nbre_courant = 1.1
    print()
    print("nombre de courant = 1.1, construction linéaire et schéma Runge Kutta à 2 pas :")
    C_evolution_point, dt, anim = execution(dx, dy, C0, U, V, nbre_courant, n_t, construction_linéaire, Runge_Kutta_2)



    precision()
    #le plot des erreurs L2 moyennes montre bien le comportement relativement linéaire et parabollique des erreurs des constructions constante et linéaire respectivement en fonction du pas spatial. 
    #Pour rappel, la construction constante est d'ordre 1 (erreur linéaire) et la construction linéaire est d'ordre 2 (erreur parabollique)
    #Le graphique démontre donc bien que le 2e ordre donne des résultats avec moins d'erreurs pour des tailles de mailles dx raisonnables par rapport à la taille du domaine.


    #Concentration initiale et Champ de vitesse non-triviaux
    x_min, x_max, y_min, y_max = 0, 2*np.pi, 0, 2*np.pi
    n_x, n_y = 200, 200
    dx = (x_max-x_min)/n_x
    dy = (y_max-y_min)/n_y
    print()
    print("dx, dy :" , dx, dy)
    nbre_courant = 1
    n_t = 1000

    #création du domaine permettant la périodicité du champ de vitesse afin d'obtenir la divergence nulle aux bords du domaine
    X, Y = domaine_periodique(x_min, x_max, y_min, y_max, n_x, n_y)
    #création des champs de vitesses qui doivent être périodiques sur les limites du domaine
    U, V = vitesse_sincos(X, Y, dx, dy)
    #création de la matrice de concentration gaussienne
    C = concentration_gaussienne(X, Y, np.pi, np.pi, sigma = 2)


    #Affichage de la concentration initiale et du champ de vitesse
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    #concentration
    im1 = axes[0].imshow(C, extent=[x_min, x_max, y_min, y_max],
                        origin='lower', cmap='viridis')
    axes[0].set_title("Concentration initiale")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im1, ax=axes[0], label="Concentration")
    #champ de vitesse
    im2 = axes[1].quiver(X[::4, ::4], Y[::4, ::4], U[::4, ::4], V[::4, ::4], np.sqrt(U[::4, ::4]**2 + V[::4, ::4]**2), cmap='plasma', scale = 50) #seul un point de vitesse sur 4 est affiché pour la lisibilité du graphe
    axes[1].set_title("Champ de vitesse")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].axis('equal')
    fig.colorbar(im2, ax=axes[1], label="Norme de la vitesse")
    plt.show

    #comparaison entre résolutions d'ordre 1 et d'ordre 2 en temps et en espace à nombre de courant = 1
    print()
    print("nombre de courant = 1, construction constante et Euler explicite :")
    C_evolution, dt, anim = execution(dx, dy, C, U, V, nbre_courant, n_t, construction_constante, Euler_explicite)

    print()
    print("nombre de courant = 1, construction linéaire et Runge-kutta Runge Kutta à 2 pas:")
    C_evolution, dt, anim = execution(dx, dy, C, U, V, nbre_courant, n_t, construction_linéaire, Runge_Kutta_2)
    print("Temps total :", n_t*dt)
    #la solution d'ordre 2 semble en effet plus précise comme on remarque que elle se diffuse moins qu'en ordre 1
