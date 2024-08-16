#%% testing 

# parametrs of the model
Nx = 20
Ny = 20

N = Nx * Ny
t1 = 1
sub = 2
# Building the single particle hamiltonian (h2)
# need to check if the gauge transformation is needed to adress (Natanel said no)
Kx = np.linspace(0, 2 * np.pi,num=Nx,endpoint=False) 
Ky = np.linspace(0, 2 * np.pi,num=Ny,endpoint=False) 
X = np.array(range(Nx)) 
Y = np.array(range(Ny)) 

# Kx, Ky = np.meshgrid(kx,ky)
def build_h2(kx, ky):
    return -2 * t1 * (np.cos(kx) + np.cos(ky)) * np.eye(sub)

H_k_list = []
for kx in Kx:
    for ky in Ky:
        H_k_list.append(build_h2(kx,ky))
        
# creaing a block diagonal H_k matrix and dft to real space

H_k = block_diag(*H_k_list)

# dft matrix as a tensor protuct of dft in x and y axis and idenity in the sublattice
dft_matrix = np.kron(dft(Nx),(np.kron(dft(Ny),np.eye(sub)))) / np.sqrt(N)
H_real_space =np.matmul(np.matmul(dft_matrix.T.conjugate(),H_k), dft_matrix)
# eig_val, eig_vec = np.linalg.eigh(H_real_space)

index = lambda x,y,A=0: sub * (Ny * x + y) + A
vectors = np.eye(N * sub)
H_real_space_2 = np.zeros((N * sub, N * sub))
for i in range(Nx):
    for j in range(Ny):
        H_real_space_2 += -t1 * np.outer(vectors[:,index(i,j)], vectors[:,index((i+1)%Nx,j)] + vectors[:,index(i,(j+1)%Ny)] +  vectors[:,index((i-1)%Nx,j)] + vectors[:,index(i,(j-1)%Ny)])
        H_real_space_2 += -t1 * np.outer(vectors[:,index(i,j,1)], vectors[:,index((i+1)%Nx,j,1)] + vectors[:,index(i,(j+1)%Ny,1)] +  vectors[:,index((i-1)%Nx,j,1)] + vectors[:,index(i,(j-1)%Ny,1)])
plt.matshow(np.abs(H_real_space))
plt.matshow(np.abs(H_real_space_2))
print(np.diag(np.matmul(np.matmul(dft_matrix,H_real_space_2), dft_matrix.T.conjugate())).real)
print(H_k_list)
# print(H_real_space_2)
state = np.zeros((N * sub))
x = 1
y = 1
index = sub * (Ny * x + y) 
state[index] =1
# print_sb_state(state,Nx,Ny)
# print_sb_state(H_real_space[:,index],Nx,Ny)