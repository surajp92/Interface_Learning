"""
Created on Fri Jul 17 15:53:51 2020

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt
import time

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
nx_fdm = 100		# number of intervals in x
lx_fdm = 10.0		# spatial domain lenght
nx_lbm = 100
lx_lbm = 10.0

nx = nx_fdm + nx_lbm
lx = lx_fdm + lx_lbm

tm = 450  	# max time
dt = 0.001	# time step

du = 1.0	    # diffusion coefficient for u
dv = 4.0		# diffusion coefficient for v

#  model paramaters
a1 = 2.0	
a0 =-0.03
ep = 0.01		# kinetic bifurcation paramater

# ploting
ns = 100 		# number of snapshots to store/plot

# spatial grid size
dx_fdm = lx_fdm/nx_fdm
dx_lbm = lx_lbm/nx_lbm

# data snapshot interval time
ds = tm/ns

# max number of time for numerical solution
nt = int(tm/dt)

# check for viscous stability condition
cu = dt*du/(dx_fdm*dx_fdm)
cv = dt*dv/(dx_fdm*dx_fdm)

#%%

# weights: 
w = (1.0/3.0)*np.ones(3)

# initial conditions: 
xc = 0.5*lx
xs = 0.25
dd = 0.1

k = 0
freq = int(nt/ns)

xall_fdm = np.linspace(0.5*dx_fdm,lx_fdm - 0.5*dx_fdm,nx_fdm)
xall_lbm = np.linspace(lx_lbm+0.5*dx_lbm,lx_fdm+lx_lbm-0.5*dx_lbm,nx_lbm)

u0_fdm = np.tanh(xall_fdm - xc)
u0_lbm = np.tanh(xall_lbm - xc)

v0_fdm = dd*(1.0 + np.tanh(xs*(xall_fdm - 0.5*lx)))
v0_lbm = dd*(1.0 + np.tanh(xs*(xall_lbm - 0.5*lx)))


u_fdm, v_fdm = np.zeros((nx_fdm,ns+1)), np.zeros((nx_fdm,ns+1))
un_fdm, vn_fdm = np.zeros((nx_fdm+2)), np.zeros((nx_fdm+2))
ru_fdm, rv_fdm = np.zeros((nx_fdm)), np.zeros((nx_fdm))

u_fdm[:,k] = u0_fdm
v_fdm[:,k] = v0_fdm

# weights: 
w = (1.0/3.0)*np.ones(3)

fu, fv = np.zeros((nx_lbm,3)), np.zeros((nx_lbm,3))
fequ, feqv = np.zeros((nx_lbm,3)), np.zeros((nx_lbm,3))
ru_lbm, rv_lbm = np.zeros((nx_lbm,3)), np.zeros((nx_lbm,3))
ou, ov = np.zeros((nx_lbm,3)), np.zeros((nx_lbm,3))
un_lbm, vn_lbm = np.zeros((nx_lbm+3)), np.zeros((nx_lbm+3))
fun, fvn = np.zeros((nx_lbm+3,3)), np.zeros((nx_lbm+3,3))

fu = np.dot(u0_lbm.reshape(-1,1),w.reshape(1,-1))
fv = np.dot(v0_lbm.reshape(-1,1),w.reshape(1,-1))

u_lbm, v_lbm = np.zeros((nx_lbm,ns+1)), np.zeros((nx_lbm,ns+1))

# store macroscopic variables 
u_lbm[:,k] = u0_lbm
v_lbm[:,k] = v0_lbm

# sample u and v (D1Q3)
un_lbm[2:nx_lbm+2] = np.sum(fu, axis = 1)
vn_lbm[2:nx_lbm+2] = np.sum(fv, axis = 1)

omegau = 2.0/(1.0 + 3.0*du*dt/(dx_lbm*dx_lbm))
omegav = 2.0/(1.0 + 3.0*dv*dt/(dx_lbm*dx_lbm))

un_fdm[1:nx_fdm+1] = u0_fdm
vn_fdm[1:nx_fdm+1] = v0_fdm

# neuman BC
un_fdm[0] = un_fdm[1]
un_fdm[nx_fdm+1] = un_lbm[2] #0.25*(un_lbm[2]+un_lbm[3]+un_lbm[4]+un_lbm[5]) # ? LBM
vn_fdm[0] = vn_fdm[1]
vn_fdm[nx_fdm+1] = vn_lbm[2] #0.25*(vn_lbm[2]+vn_lbm[3]+vn_lbm[4]+vn_lbm[5]) # ? LBM


#%%
# temporal integration
start = time.time()
for j in range(1,nt+1):
        
    # sample u and v (D1Q3)
    un_lbm[2:nx_lbm+2] = np.sum(fu, axis = 1)
    vn_lbm[2:nx_lbm+2] = np.sum(fv, axis = 1)

    # equilibrium (all latices have 1/3 coefficient)
    fequ = np.dot(un_lbm[2:nx_lbm+2].reshape(-1,1),w.reshape(1,-1))
    feqv = np.dot(vn_lbm[2:nx_lbm+2].reshape(-1,1),w.reshape(1,-1))
    
    # BGK collision term
    ou = -omegau*(fu - fequ)
    ov = -omegav*(fv - feqv)
    
    # reaction term (same weights, they are all 1/3)
    # ru = (u - u**3 - v)
    # rv = ep*(u - a1*v - a0) !in the ref paper, there is a typo in Eq.20, it should be linear in v, not cubic
    rhs_u = un_lbm[2:nx_lbm+2] - un_lbm[2:nx_lbm+2]**3 - vn_lbm[2:nx_lbm+2]
    rhs_v = ep*(un_lbm[2:nx_lbm+2] - a1*vn_lbm[2:nx_lbm+2] - a0)   
    
    ru_lbm = dt*np.dot(rhs_u.reshape(-1,1),w.reshape(1,-1))
    rv_lbm = dt*np.dot(rhs_v.reshape(-1,1),w.reshape(1,-1))
    
    # update
    # i = 0
    fun[2:nx_lbm+2,1] = fu[:,1] + ou[:,1] + ru_lbm[:,1]
    fvn[2:nx_lbm+2,1] = fv[:,1] + ov[:,1] + rv_lbm[:,1]
    
    # i = 1
    fun[3:nx_lbm+2,2] = fu[:nx_lbm-1,2] + ou[:nx_lbm-1,2] + ru_lbm[:nx_lbm-1,2]
    fvn[3:nx_lbm+2,2] = fv[:nx_lbm-1,2] + ov[:nx_lbm-1,2] + rv_lbm[:nx_lbm-1,2]
    # the halfway bounce-back scheme (left boundary)
    ufdm = un_fdm[nx_fdm]
    vfdm = vn_fdm[nx_fdm]
    
    # first order
    futemp = ufdm/3.0 - (1.0/(6.0*omegau))*(un_lbm[2] - un_fdm[nx_fdm-1])  # fu[0,0] #
    fvtemp = vfdm/3.0 - (1.0/(6.0*omegav))*(vn_lbm[2] - vn_fdm[nx_fdm-1]) # fv[0,0] #
    
    rutemp = ufdm - ufdm**3 - vfdm
    rvtemp = ep*(ufdm - a1*vfdm - a0) 
    
    fustar = (1.0-omegau)*futemp + (omegau/3.0)*ufdm + (dt/3.0)*rutemp
    fvstar = (1.0-omegav)*fvtemp + (omegav/3.0)*vfdm + (dt/3.0)*rvtemp
        
    fun[2,2] = fustar #utemp #fustar
    fvn[2,2] = fvstar #vtemp #fvstar
    
    # i = -1
    fun[2:nx_lbm+1,0] = fu[1:,0] + ou[1:,0] + ru_lbm[1:,0]
    fvn[2:nx_lbm+1,0] = fv[1:,0] + ov[1:,0] + rv_lbm[1:,0]
    # the halfway bounce-back scheme (right boundary)
    fun[nx_lbm+1,0] = fu[nx_lbm-1,2]
    fvn[nx_lbm+1,0] = fv[nx_lbm-1,2]
    
    fu[:,:] = fun[2:nx_lbm+2,:]
    fv[:,:] = fvn[2:nx_lbm+2,:]
    
    # sample u and v (D1Q3) for setting BC for FDM right boundary
    un_lbm[2:nx_lbm+2] = np.sum(fu, axis = 1)
    vn_lbm[2:nx_lbm+2] = np.sum(fv, axis = 1)
    
    # FDM domain
    ru_fdm, rv_fdm = rhs(du,dv,ep,a1,a0,un_fdm,vn_fdm,nx_fdm,dx_fdm)
    
    un_fdm[1:nx_fdm+1] = un_fdm[1:nx_fdm+1]+ dt*ru_fdm
    vn_fdm[1:nx_fdm+1] = vn_fdm[1:nx_fdm+1]+ dt*rv_fdm
    
    # neuman BC
    un_fdm[0] = un_fdm[1]
    un_fdm[nx_fdm+1] = un_lbm[2] #0.25*(un_lbm[2]+un_lbm[3]+un_lbm[4]+un_lbm[5])  #un_lbm[2] # ? LBM
    vn_fdm[0] = vn_fdm[1]
    vn_fdm[nx_fdm+1] = vn_lbm[2]  #0.25*(vn_lbm[2]+vn_lbm[3]+vn_lbm[4]+vn_lbm[5]) #vn_lbm[2] # ? LBM
    
    if j%freq == 0:
#        print(j*dt)
        k = k+1
        u_fdm[:,k] = un_fdm[1:nx_fdm+1]
        v_fdm[:,k] = vn_fdm[1:nx_fdm+1]
        
        u_lbm[:,k] = np.sum(fu, axis = 1)
        v_lbm[:,k] = np.sum(fv, axis = 1)

end = time.time()
total_time = np.array([end - start])
np.savetxt('time.txt', total_time)

#%%
u_all = np.vstack((u_fdm,u_lbm))
v_all = np.vstack((v_fdm,v_lbm))

#%%
x = np.linspace(0.5*dx_fdm,lx-0.5*dx_lbm,nx)
t = np.linspace(0,tm,ns+1)

fig, ax = plt.subplots(1,2,figsize=(10,4),sharex=True)
axs = ax.flat
vmin = -1.0
vmax = 1.0

cs = axs[0].contourf(t,x,u_all,60,cmap='jet',vmin=vmin,vmax=vmax,zorder=-9)
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

cs = axs[1].contourf(t,x,v_all,30,cmap='jet')
axs[1].set_xlabel('$T$')
axs[1].set_ylabel('$X$')
axs[1].set_title('$V$')
cbaxes = fig.add_axes([0.6, -0.075, 0.35, 0.05]) 
cbar = fig.colorbar(cs, cax=cbaxes,orientation='horizontal')
cbar.set_ticks([-0.18,0.0,0.18])
cbar.set_ticklabels([-0.18,0.0,0.18])

fig.tight_layout()
plt.show()
fig.savefig(f'hybrid1_{nx}.pdf',dpi=300)
fig.savefig(f'hybrid1_{nx}.png',dpi=300)

np.savez(f'base_{nx}_{tm}' , x=x, t=t, u=u_all, v =v_all)
