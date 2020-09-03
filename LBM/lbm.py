"""
Created on Wed Jul 15 17:20:09 2020

@author: suraj
"""
import numpy as np
import matplotlib.pyplot as plt

#%%
nx = 200 		# number of intervals in x
lx = 20.0		# spatial domain lenght
tm = 450.0  	# max time
dt = 0.001	# time step

du = 1.0	    # diffusion coefficient for u
dv = 4.0		# diffusion coefficient for v

alpha = 1.0

#  model paramaters
a1 = 2.0	
a0 =-0.03
ep = 0.01		# kinetic bifurcation paramater

# ploting
ns = 100 		# number of snapshots to store/plot

# spatial grid size
dx = lx/nx

# data snapshot interval time
ds = tm/ns

# max number of time for numerical solution
nt = int(tm/dt)

# check for viscous stability condition
cu = dt*du/(dx*dx)
cv = dt*dv/(dx*dx)

#%%
#-----------------------------------------------------------------------------!
# Compute lattice Boltzmann numerical solutions: D1Q3 lattice
# 1/3, 1/3, 1/3 weights are used
# At the boundaries, we impose homogeneous Neumann (i.e. no-flux) boundary conditions 
#-----------------------------------------------------------------------------!

u, v = np.zeros((nx,ns+1)), np.zeros((nx,ns+1))
u0, v0 = np.zeros(nx), np.zeros(nx)
fu, fv = np.zeros((nx,3)), np.zeros((nx,3))
fequ, feqv = np.zeros((nx+1,3)), np.zeros((nx+1,3))
ru, rv = np.zeros((nx+1,3)), np.zeros((nx+1,3))
ou, ov = np.zeros((nx+1,3)), np.zeros((nx+1,3))
un, vn = np.zeros((nx+3)), np.zeros((nx+3))
fun, fvn = np.zeros((nx+3,3)), np.zeros((nx+3,3))

fun_all, fvn_all = np.zeros((nx+3,3,nt+1)), np.zeros((nx+3,3,nt+1))

# D1 refers to one-dimensional
# Q3 refers to lattice (p=-1, 0, +1)

# weights: 
w = (1.0/3.0)*np.ones(3)

# initial conditions: 
xc = 0.5*lx
xs = 0.25
dd = 0.1

# initial conditions: 
#for i in range(nx):
#    x = -0.5*dx + (i+1)*dx
#    u0[i] = np.tanh(x-xc)
#    v0[i] = dd*(1.0 + np.tanh(xs*(x-0.5*lx)))

xall = np.linspace(0.5*dx,lx-0.5*dx,nx)
u0 = alpha*np.tanh(xall - xc)
v0 = dd*(1.0 + np.tanh(xs*(xall - 0.5*lx)))

# initial conditions for particle densities:
# equally distributed weights
#for i in range(nx):
#    for p in range(3):
#        fu[i,p] = w[p]*u0[i]
#        fv[i,p] = w[p]*v0[i]

fu = np.dot(u0.reshape(-1,1),w.reshape(1,-1))
fv = np.dot(v0.reshape(-1,1),w.reshape(1,-1))

# store macroscopic variables 
u[:,0] = u0
v[:,0] = v0

# time integration for particle density distributions
k = 0
freq = int(nt/ns)

omegau = 2.0/(1.0 + 3.0*du*dt/(dx*dx))
omegav = 2.0/(1.0 + 3.0*dv*dt/(dx*dx))

fun_all[:,:,0] = fun
fvn_all[:,:,0] = fvn
    
for j in range(1,nt+1):
    
    # sample u and v (D1Q3)
    un[2:nx+2] = np.sum(fu, axis = 1)
    vn[2:nx+2] = np.sum(fv, axis = 1)

    # equilibrium (all latices have 1/3 coefficient)
    fequ = np.dot(un[2:nx+2].reshape(-1,1),w.reshape(1,-1))
    feqv = np.dot(vn[2:nx+2].reshape(-1,1),w.reshape(1,-1))
    
    # BGK collision term
    ou = -omegau*(fu - fequ)
    ov = -omegav*(fv - feqv)
    
    # reaction term (same weights, they are all 1/3)
    # ru = (u - u**3 - v)
    # rv = ep*(u - a1*v - a0) !in the ref paper, there is a typo in Eq.20, it should be linear in v, not cubic
    rhs_u = un[2:nx+2] - un[2:nx+2]**3 - vn[2:nx+2]
    rhs_v = ep*(un[2:nx+2] - a1*vn[2:nx+2] - a0)   
    
    ru = dt*np.dot(rhs_u.reshape(-1,1),w.reshape(1,-1))
    rv = dt*np.dot(rhs_v.reshape(-1,1),w.reshape(1,-1))
    
    # update
    fun[2:nx+2,1] = fu[:,1] + ou[:,1] + ru[:,1]
    fvn[2:nx+2,1] = fv[:,1] + ov[:,1] + rv[:,1]
    
    fun[3:nx+2,2] = fu[:nx-1,2] + ou[:nx-1,2] + ru[:nx-1,2]
    fvn[3:nx+2,2] = fv[:nx-1,2] + ov[:nx-1,2] + rv[:nx-1,2]
    # the halfway bounce-back scheme (left boundary)
    fun[2,2] = fu[0,0]
    fvn[2,2] = fv[0,0] 
    
    fun[2:nx+1,0] = fu[1:,0] + ou[1:,0] + ru[1:,0]
    fvn[2:nx+1,0] = fv[1:,0] + ov[1:,0] + rv[1:,0]
    # the halfway bounce-back scheme (right boundary)
    fun[nx+1,0] = fu[nx-1,2]
    fvn[nx+1,0] = fv[nx-1,2]
    
    fu[:,:] = fun[2:nx+2,:]
    fv[:,:] = fvn[2:nx+2,:]
    
    fun_all[:,:,j] = fun
    fvn_all[:,:,j] = fvn
    
    if j%freq == 0:
        print(j*dt)
        un[2:nx+2] = np.sum(fu, axis = 1)
        vn[2:nx+2] = np.sum(fv, axis = 1)
        
        k = k+1
        u[:,k] = un[2:nx+2]
        v[:,k] = vn[2:nx+2]



#%%
# validation with Nx = 100
if nx == 100: 
    un_f_lbm = np.loadtxt('u_450_lbm.txt')
    
    aan = u[:,-1] - un_f_lbm[:,1]
    
    plt.plot(u[:,0],label='Initial condition')
    plt.plot(u[:,-1],label='Python')
    plt.plot(un_f_lbm[:,1],label='Fortran')
    plt.legend()
    plt.show()
            
    
    vn_f_lbm = np.loadtxt('v_450_lbm.txt')
    
    bbn = v[:,-1] - vn_f_lbm[:,1]
    
    plt.plot(v[:,0],label='Initial condition')
    plt.plot(v[:,-1],label='Python')
    plt.plot(vn_f_lbm[:,1],label='Fortran')
    plt.legend()
    plt.show()

#%%
x = np.linspace(0.5*dx,lx-0.5*dx,nx)
t = np.linspace(0,tm,ns+1)

fig, ax = plt.subplots(1,2,figsize=(10,4),sharex=True)
axs = ax.flat
vmin = -1.0
vmax = 1.0

cs = axs[0].contourf(t,x,u,60,cmap='jet',vmin=vmin,vmax=vmax,zorder=-9)
axs[0].set_rasterization_zorder(-1)
axs[0].set_xlabel('$T$')
axs[0].set_ylabel('$X$')
axs[0].set_title('$U$')
cbaxes = fig.add_axes([0.1, -0.075, 0.35, 0.05]) 
cbar = fig.colorbar(cs, cax=cbaxes,orientation='horizontal')
cbar.set_ticks([-0.9,0.0,0.9])
cbar.set_ticklabels([-0.9,0.0,0.9])

vmin = -0.2
vmax = 0.2

cs = axs[1].contourf(t,x,v,30,cmap='jet')
axs[1].set_xlabel('$T$')
axs[1].set_ylabel('$X$')
axs[1].set_title('$V$')
cbaxes = fig.add_axes([0.6, -0.075, 0.35, 0.05]) 
cbar = fig.colorbar(cs, cax=cbaxes,orientation='horizontal')
cbar.set_ticks([-0.18,0.0,0.18])
cbar.set_ticklabels([-0.18,0.0,0.18])

fig.tight_layout()
plt.show()
fig.savefig(f'lbm_{nx}_{alpha}.pdf',dpi=300)
fig.savefig(f'lbm_{nx}_{alpha}.png',dpi=300)

np.savez(f'solution_lbm_{nx}_{tm}_{alpha}' , x=x, t=t, u=u, v =v, fun_all=fun_all, fvn_all=fvn_all)

#%%

plt.plot(u[:,0])
plt.plot(u[:,-1])
plt.show()

plt.plot(v[:,0])
plt.plot(v[:,-1])
plt.show()







