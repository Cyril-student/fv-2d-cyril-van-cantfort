import numpy as np
import time


C = np.array([[0,0,1],
              [0,0,0],
                [1,0,1]])

U, V = np.zeros((3,3)), np.zeros((3,3))

def rotation_probleme(C, U, V, angle):
    if angle == 90:
        C_new = np.rot90(C)
        U_new = np.rot90(U)
        V_new = np.rot90(V)
        U_new, V_new = V_new, -U_new
    elif angle == 180:
        C_new = np.rot90(C, 2)
        U_new = np.rot90(U, 2)
        V_new = np.rot90(V, 2)
        U_new, V_new = -U_new, -V_new
    elif angle == 270:
        C_new = np.rot90(C, 3)
        U_new = np.rot90(U, 3)
        V_new = np.rot90(V, 3)
        U_new, V_new = -V_new, U_new
    else:
        print("Angle de rotation non supporté. Utilisez 90, 180 ou 270 degrés.")
        return C, U, V
    return C_new, U_new, V_new

print("Matrice originale C :\n", C)

C_rotated, U_rotated, V_rotated = rotation_probleme(C, U, V, 90)

print("Matrice C après rotation de 90 degrés :\n", C_rotated)

C_rotated, U_rotated, V_rotated = rotation_probleme(C, U, V, 180)

print("Matrice C après rotation de 180 degrés :\n", C_rotated)

C_rotated, U_rotated, V_rotated = rotation_probleme(C, U, V, 270)

print("Matrice C après rotation de 270 degrés :\n", C_rotated)