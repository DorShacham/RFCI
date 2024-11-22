#%%
import numpy as np 
import matplotlib.pyplot as plt

#%%
A = 1
a = 1
def normalize(vec):
    return np.array(vec) / np.linalg.norm(vec)

def grad(fn, x,y, epsilon = 1e-6):
    return ((fn(x + epsilon / 2, y ) - fn(x - epsilon / 2, y)) / epsilon), ((fn(x, y + epsilon / 2) - fn(x, y - epsilon / 2)) / epsilon)


def w_xy(kx, ky, B):
    kx -=  1e5 / 100
    d_hat = lambda kx,ky: normalize(A * np.array([1 + np.cos(kx) + np.cos(ky), np.sin(kx) + np.sin(ky), B + 2 * np.sin(kx)]))
    d_hat_dkx, d_hat_dky = grad(d_hat, kx, ky)
    return 1 / 2 * np.dot(d_hat(kx,ky) , np.cross(d_hat_dkx, d_hat_dky))




#%%
Nx = 100
Ny = 100
N = Nx * Ny

Kx =  np.pi / a * np.linspace(-1,1,num=Nx,endpoint=False)
Ky =  np.pi / a * np.linspace(-1,1,num=Ny,endpoint=False)
B_array = np.linspace(-2,6,9 + 1)

indices_to_delete = np.where(np.isin(B_array, [0,2,4]))[0]
B_array = np.delete(B_array, indices_to_delete)

C_array = []

for B in  B_array:
    w = np.zeros((Nx,Ny))
    C = 0
    for i in range(len(Kx)):
        for j in range(len(Ky)):
            w[j,i] = w_xy(Kx[i], Ky[j], B)
            C += 2 * np.pi * w[j,i] / N

    C_array.append(C)

    if B in [ - 1, 1, 3, 5]:
        print(f"B = {B}: C = {C}")
        K_ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
        # Create the heatmap
        plt.figure(dpi=300)
        plt.imshow(w, extent=[-np.pi, np.pi, -np.pi, np.pi], origin='lower', aspect='auto')

        # Set the ticks for the x and y axes
        plt.xticks(ticks=K_ticks, labels=[r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
        plt.yticks(ticks=K_ticks, labels=[r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])

        # Add axis labels
        plt.xlabel(r'$K_x$')
        plt.ylabel(r'$K_y$')

        # Add a color bar to indicate the colors
        plt.colorbar(label='Intensity')
        plt.title(str(r"$\omega^{+}_{xy}(k_x,k_y)$ for $B=$" + str(B)))
        # plt.savefig(str(f"figs/q2_B_{B}.jpg"))

plt.figure(dpi=300)
plt.plot(B_array,C_array,".")
plt.grid()
plt.xlabel(r'$B$')
plt.ylabel(r'$C$')
plt.title(r"The Chern number as a function of B")
# plt.savefig("figs/q2_C-B_plot.jpg")

# %%

#%% Question 1.2.a

import numpy as np

# Define the coefficient matrix A
A = np.array([
    [ 1, -1,  0,  0,  0],
    [ 0,  1, -1,  0,  0],
    [ 0,  0,  1, -1,  0],
    [ 0,  0,  0,  1, -1],
    [-1,  0,  0,  0,  1]
])

# Define the right-hand side vector b
# We'll use a symbolic 'c' to represent h*I_12/(e^2)
b = np.array([1, -1, 0, 0, 0])
# Solve the system
x = np.linalg.pinv(A) @ b

print("Least-squares solution:")
for i, value in enumerate(x):
    print(f"x{i+1} = {value:.6f}a")

residual = A @ x - b
print("Residual:", residual)


#%% Question 1.2.b

import numpy as np

# Define the coefficient matrix A
A = np.array([
    [ 1, -1,  0,  0,  0],
    [ 0,  1, -1,  0,  0],
    [ 0,  0,  1, -1,  0],
    [ 0,  0,  0,  1, -1],
    [-1,  0,  0,  0,  1]
])

# Define the right-hand side vector b
# We'll use a symbolic 'c' to represent h*I_12/(e^2)
b = np.array([1, 0, -1, 0, 0])
# Solve the system
x = np.linalg.pinv(A) @ b

print("Least-squares solution:")
for i, value in enumerate(x):
    print(f"x{i+1} = {value:.6f}a")

residual = A @ x - b
print("Residual:", residual)

#%% Question 1.3.a

import numpy as np

# Define the coefficient matrix A
A = np.array([
    [ 2, -1,  0,  0, -1],
    [-1,  2, -1,  0,  0],
    [ 0, -1,  2, -1,  0],
    [ 0,  0, -1,  2, -1],
    [-1,  0,  0, -1,  2]
])

# Define the right-hand side vector b
# We'll use a symbolic 'c' to represent h*I_12/(e^2)
b = np.array([1, -1, 0, 0, 0])
# Solve the system
x = np.linalg.pinv(A) @ b

print("Least-squares solution:")
for i, value in enumerate(x):
    print(f"x{i+1} = {value:.6f}a")

residual = A @ x - b
print("Residual:", residual)


#%% Question 1.2.b

import numpy as np

# Define the coefficient matrix A
A = np.array([
    [ 2, -1,  0,  0, -1],
    [-1,  2, -1,  0,  0],
    [ 0, -1,  2, -1,  0],
    [ 0,  0, -1,  2, -1],
    [-1,  0,  0, -1,  2]
])

# Define the right-hand side vector b
# We'll use a symbolic 'c' to represent h*I_12/(e^2)
b = np.array([1, 0, -1, 0, 0])
# Solve the system
x = np.linalg.pinv(A) @ b

print("Least-squares solution:")
for i, value in enumerate(x):
    print(f"x{i+1} = {value:.6f}a")

residual = A @ x - b
print("Residual:", residual)