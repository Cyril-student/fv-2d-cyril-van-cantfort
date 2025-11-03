import matplotlib.pyplot as plt
import numpy as np

"""
Fonction de courant exemple 1

$PSI = sin(x) * cos(y)$
"""
def champ_courant(x, y):
    """ Fonction de courant """
    return np.sin(x) * np.cos(y)

def champ_vitesse(x, y):
    """ Vitesse dérivée de la fonction de courant """
    u = -np.sin(x) * np.sin(y)
    v = -np.cos(x) * np.cos(y)

    return u, v

def int_u(x, y, dy):
    """ Intégrale de u sur un bord """
    return champ_courant(x, y+dy/2) - champ_courant(x, y-dy/2)

def int_v(x, y, dx):
    """ Intégrale de u sur un bord """
    return -(champ_courant(x+dx/2, y) - champ_courant(x-dx/2, y))

def vitesse_au_centre_du_bord_x(x, y):
    """ Calcule la vitesse au centre du bord selon X """
    u,v = champ_vitesse(x, y)

    return u

def vitesse_au_centre_du_bord_y(x, y):
    """ Calcule la vitesse au centre du bord selon Y """
    u,v = champ_vitesse(x, y)

    return v

def vitesse_moyenne_bord_x(x, y, dy):
    """ Calcule la vitesse moyenne sur le bord selon X """
    return int_u(x,y,dy) / dy

def vitesse_moyenne_bord_y(x, y, dx):
    """ Calcule la vitesse moyenne sur le bord selon Y """
    return int_v(x,y,dx) / dx

def compare_vitesses(x, y, dx, dy):
    """ Compare la vitesse au centre et la vitesse moyenne sur le bord """

    u_centre = vitesse_au_centre_du_bord_x(x, y)
    u_moyen  = vitesse_moyenne_bord_x(x, y, dy)
    v_centre = vitesse_au_centre_du_bord_y(x, y)
    v_moyen  = vitesse_moyenne_bord_y(x, y, dx)

    print('Vitesse au centre du bord selon X :', u_centre)
    print('Vitesse moyenne sur le bord selon X :', u_moyen)
    print('Vitesse au centre du bord selon Y :', v_centre)
    print('Vitesse moyenne sur le bord selon Y :', v_moyen)

def bilan_vitesse(x,y,dx,dy):
    """ Calcul du bilan sur base des valeurs centreées et moyennes exactes """
    u_gauche = vitesse_au_centre_du_bord_x(x-dx/2, y)
    u_droite = vitesse_au_centre_du_bord_x(x+dx/2, y)
    v_bas    = vitesse_au_centre_du_bord_y(x, y-dy/2)
    v_haut   = vitesse_au_centre_du_bord_y(x, y+dy/2)

    divergence_centre = (u_droite - u_gauche) * dy + (v_haut -v_bas) * dx

    u_moyen_gauche = vitesse_moyenne_bord_x(x-dx/2, y, dy)
    u_moyen_droite = vitesse_moyenne_bord_x(x+dx/2, y, dy)
    v_moyen_bas    = vitesse_moyenne_bord_y(x, y-dy/2, dx)
    v_moyen_haut   = vitesse_moyenne_bord_y(x, y+dy/2, dx)

    divergence_moyenne = (u_moyen_droite - u_moyen_gauche) * dy + (v_moyen_haut - v_moyen_bas) * dx

    return divergence_centre - divergence_moyenne

"""
Fonction de courant exemple 2 : rotation solide sens anti-horaire

$PSI = -(x^2 + y^2)/2$

"""

def champ_courant_rotation(x, y):
    """ Fonction de courant pour une rotation solide """
    return -(x**2 + y**2)/2

def champ_vitesse_rotation(x, y):
    """ Pure rotation solide """
    u = -y
    v = x

    return u, v

def int_u_rotation(x, y, dy):
    """ Intégrale de u sur un bord pour une rotation solide """
    return champ_courant_rotation(x, y+dy/2) - champ_courant_rotation(x, y-dy/2)

def int_v_rotation(x, y, dx):
    """ Intégrale de u sur un bord pour une rotation solide """
    return -(champ_courant_rotation(x+dx/2, y) - champ_courant_rotation(x-dx/2, y))

def vitesse_au_centre_du_bord_x_rotation(x, y):
    """ Calcule la vitesse au centre du bord selon X pour une rotation solide """
    u,v = champ_vitesse_rotation(x, y)
    return u

def vitesse_au_centre_du_bord_y_rotation(x, y):
    """ Calcule la vitesse au centre du bord selon Y pour une rotation solide """
    u,v = champ_vitesse_rotation(x, y)
    return v

def vitesse_moyenne_bord_x_rotation(x, y, dy):
    """ Calcule la vitesse moyenne sur le bord selon X pour une rotation solide """
    return int_u_rotation(x,y,dy) / dy

def vitesse_moyenne_bord_y_rotation(x, y, dx):
    """ Calcule la vitesse moyenne sur le bord selon Y pour une rotation solide """
    return int_v_rotation(x,y,dx) / dx

def bilan_vitesse_rotation(x,y,dx,dy):
    """ Calcul du bilan sur base des valeurs centreées et moyennes exactes """

    u_gauche = vitesse_au_centre_du_bord_x_rotation(x-dx/2, y)
    u_droite = vitesse_au_centre_du_bord_x_rotation(x+dx/2, y)
    v_bas    = vitesse_au_centre_du_bord_y_rotation(x, y-dy/2)
    v_haut   = vitesse_au_centre_du_bord_y_rotation(x, y+dy/2)

    divergence_centre = (u_droite - u_gauche) * dy + (v_haut -v_bas) * dx

    u_moyen_gauche = vitesse_moyenne_bord_x_rotation(x-dx/2, y, dy)
    u_moyen_droite = vitesse_moyenne_bord_x_rotation(x+dx/2, y, dy)
    v_moyen_bas    = vitesse_moyenne_bord_y_rotation(x, y-dy/2, dx)
    v_moyen_haut   = vitesse_moyenne_bord_y_rotation(x, y+dy/2, dx)

    divergence_moyenne = (u_moyen_droite - u_moyen_gauche) * dy + (v_moyen_haut - v_moyen_bas) * dx

    return divergence_centre - divergence_moyenne

""" Divers """

def domaine(xmin, xmax, ymin, ymax, nx, ny):
    """ Crée un domaine de calcul """
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)

    return X, Y

def plot_champ_vitesse(X, Y, u, v):
    """ Affiche le champ de vitesse """
    fig, axes = plt.subplots(1,2)
    axes[0].contour(X, Y, champ_courant(X, Y), 20)
    axes[0].set_title('Lignes de courant')
    axes[1].quiver(X, Y, u, v, scale=20)
    axes[1].set_title('Champ de vitesse')

    for ax in axes:
        ax.set_xlim(-2*np.pi, 2*np.pi)
        ax.set_ylim(-2*np.pi, 2*np.pi)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    fig.show()

def plot_champ_vitesse_rotation(X, Y, u, v):
    """ Affiche le champ de vitesse pour une rotation solide """
    fig, axes = plt.subplots(1,2)
    axes[0].contour(X, Y, champ_courant_rotation(X, Y), 20)
    axes[0].set_title('Lignes de courant')
    axes[1].quiver(X, Y, u, v, scale=100)
    axes[1].set_title('Champ de vitesse')

    for ax in axes:
        ax.set_xlim(-2*np.pi, 2*np.pi)
        ax.set_ylim(-2*np.pi, 2*np.pi)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    fig.show()


def main():
    # Création du domaine
    xmin, xmax, ymin, ymax = -2*np.pi, 2*np.pi, -2*np.pi, 2*np.pi
    nx, ny = 200, 200
    X, Y = domaine(xmin, xmax, ymin, ymax, nx, ny)

    # Calcul du champ de vitesse
    u, v = champ_vitesse(X, Y)

    # Affichage du champ de vitesse
    plot_champ_vitesse(X, Y, u, v)

    # Calcul du champ de vitesse pour une rotation solide
    u, v = champ_vitesse_rotation(X, Y)

    # Affichage du champ de vitesse pour une rotation solide
    plot_champ_vitesse_rotation(X, Y, u, v)

if __name__ == '__main__':
    compare_vitesses(5., 0., 1, 1)
    print('Différence de bilan entre valeurs centrales et valeurs exactes :', bilan_vitesse(5, 2, .1, .1))
    print('Différence de bilan entre valeurs centrales et valeurs exactes :', bilan_vitesse_rotation(5, 2, .1, .1))
    main()
    pass