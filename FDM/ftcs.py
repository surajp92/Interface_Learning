"""
Created on Thu Jul 16 17:32:36 2020

@author: suraj
"""
import numpy as np
import matplotlib.pyplot as plt

#%%
#-----------------------------------------------------------------------------!
# compute rhs for numerical solutions
#
#  	ru = du*u'' + u - u**3 - v
#  
#	rv = dv*v'' + ep*(u - a1*v - a0)	
#-----------------------------------------------------------------------------!
def rhs(du,dv,ep,a1,a0,un,vn,nx,dx):
    
    ru = du*(un[2:nx+2] - 2.0*un[1:nx+1] + un[0:nx])/(dx**2) + \
         un[1:nx+1] - un[1:nx+1]**3 - vn[1:nx+1]
         
    rv = dv*(vn[2:nx+2] - 2.0*vn[1:nx+1] + vn[0:nx])/(dx**2) + \
         ep*(un[1:nx+1] - a1*vn[1:nx+1] - a0)
         
    return ru, rv

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
a0 = -0.03
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
u, v = np.zeros((nx,ns+1)), np.zeros((nx,ns+1))
un, vn = np.zeros((nx+2)), np.zeros((nx+2))
ru, rv = np.zeros((nx)), np.zeros((nx))

un_all, vn_all = np.zeros((nx+2,nt+1)), np.zeros((nx+2,nt+1))

# initial conditions: 
xc = 0.5*lx
xs = 0.25
dd = 0.1

alpha = 0.7

xall = np.linspace(0.5*dx,lx-0.5*dx,nx)
u0 = alpha*np.tanh(xall - xc)
v0 = dd*(1.0 + np.tanh(xs*(xall - 0.5*lx)))

#%%
plt.plot(uo_1)
plt.plot(u0)

#%%
un[1:nx+1] = u0
vn[1:nx+1] = v0

# neuman BC
un[0] = un[1]
un[nx+1] = un[nx]
vn[0] = vn[1]
vn[nx+1] = vn[nx]

k = 0
freq = int(nt/ns)

u[:,k] = u0
v[:,k] = v0

for j in range(1,nt+1):
    
    ru, rv = rhs(du,dv,ep,a1,a0,un,vn,nx,dx)
    
    un[1:nx+1] = un[1:nx+1] + dt*ru
    vn[1:nx+1] = vn[1:nx+1] + dt*rv
    
    # neuman BC
    un[0] = un[1]
    un[nx+1] = un[nx]
    vn[0] = vn[1]
    vn[nx+1] = vn[nx]
    
    un_all[:,j] = un
    vn_all[:,j] = vn
    
    if j%freq == 0:
        k = k+1
        u[:,k] = un[1:nx+1]
        v[:,k] = vn[1:nx+1]
        print(j*dt)


#%%
# validation with Nx = 100
if nx == 100:         
    un_f_ftcs = np.loadtxt('u_450_ftcs.txt')
    
    aan = u[:,-1] - un_f_ftcs[:,1]
    
    plt.plot(u[:,0],label='Initial condition')
    plt.plot(u[:,-1],label='Python')
    plt.plot(un_f_ftcs[:,1],label='Fortran')
    plt.legend()
    plt.show()
            
    
    vn_f_ftcs = np.loadtxt('v_450_ftcs.txt')
    
    bbn = v[:,-1] - vn_f_ftcs[:,1]
    
    plt.plot(v[:,0],label='Initial condition')
    plt.plot(v[:,-1],label='Python')
    plt.plot(vn_f_ftcs[:,1],label='Fortran')
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
fig.savefig(f'ftcs_{nx}_{alpha}.pdf',dpi=300)
fig.savefig(f'ftcs_{nx}_{alpha}.png',dpi=300)

np.savez(f'solution_fdm_{nx}_{tm}_{alpha}' , x=x, t=t, u=u, v =v, un_all=un_all, vn_all=vn_all)

#%%
plt.plot(u[:,:])
plt.show()

plt.plot(u[:,0])
plt.plot(u[:,-1])
plt.show()

plt.plot(v[:,0])
plt.plot(v[:,-1])
plt.show()





