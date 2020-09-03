
"""
Created on Mon May 11 10:34:23 2020

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.integrate import simps
#import pyfftw
import time

#from keras.utils import plot_model
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Input
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from scipy.stats import norm 
from keras import backend as kb
from sklearn.preprocessing import MinMaxScaler
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.model_selection import train_test_split
from keras.regularizers import l2

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker

font = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 14}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

#%%
def coeff_determination(y_true, y_pred):
    SS_res =  kb.sum(kb.square(y_true-y_pred ))
    SS_tot = kb.sum(kb.square( y_true - kb.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + kb.epsilon()) )

    
#%% main parameters:
nx = 200
nx_a = 100
tm = 450.0  	# max time
dt = 0.001	# time step
nt = int(tm/dt)

training = False

#%%
alpha = 0.7
data = np.load(f'../ftcs/solution_fdm_{nx}_{tm}_{alpha}.npz')
u_fdm_all = data['u']
v_fdm_all = data['v']
un_a = data['un_all'][1:nx+1,1:nt+1][nx_a-1,:].reshape(-1,1)
vn_a = data['vn_all'][1:nx+1,1:nt+1][nx_a-1,:].reshape(-1,1)

del data

data = np.load(f'../lbm/solution_lbm_{nx}_{tm}_{alpha}.npz')
u_lbm_all = data['u']
v_lbm_all = data['v']
fun_m1_A = data['fun_all'][2:nx+2,:,1:nt+1][nx_a,0,:].reshape(-1,1)
fvn_m1_A = data['fvn_all'][2:nx+2,:,1:nt+1][nx_a,0,:].reshape(-1,1)

fun_p1_A = data['fun_all'][2:nx+2,:,1:nt+1][nx_a,2,:].reshape(-1,1)
fvn_p1_A = data['fvn_all'][2:nx+2,:,1:nt+1][nx_a,2,:].reshape(-1,1)

del data

plt.plot(fun_p1_A,label='fu+1')
plt.plot(fun_m1_A,label='fu-1')
plt.legend()
plt.show()

plt.plot(fvn_p1_A,label='fv+1')
plt.plot(fvn_m1_A,label='fv-1')
plt.legend()
plt.show()

plt.plot(un_a,label='ua')
plt.plot(vn_a,label='va')
plt.legend()
plt.show()

alpha = alpha*np.ones(un_a[:-1,:].shape)
xtrain = np.hstack((un_a[:-1,:],vn_a[:-1,:],fun_m1_A[:-1,:],fvn_m1_A[:-1,:]))
#xtrain = np.hstack((un_a[:-1,:],vn_a[:-1,:],fun_m1_A[:-1,:],fvn_m1_A[:-1,:]))
ytrain = np.hstack((fun_p1_A[1:,:],fvn_p1_A[1:,:]))

del fun_m1_A, fvn_m1_A, fun_p1_A, fvn_p1_A

#%%
sc_input = MinMaxScaler(feature_range=(-1,1))
sc_input = sc_input.fit(xtrain)
xtrain_sc = sc_input.transform(xtrain)

sc_output = MinMaxScaler(feature_range=(-1,1))
sc_output = sc_output.fit(ytrain)
ytrain_sc = sc_output.transform(ytrain)

x_train, x_valid, y_train, y_valid = train_test_split(xtrain_sc, ytrain_sc, test_size=0.2 , shuffle= True)

#%%
nf = xtrain.shape[1]
nl = ytrain.shape[1]

n_layers = 1
n_neurons = [8]
lr = 0.001
epochs = 100
batch_size = 256

filepath = f"dnn_best_model_{tm}_{nf}.hd5"

if training:
    model = Sequential()
    input_layer = Input(shape=(nf,))
    
    x = Dense(n_neurons[0], activation='relu',  use_bias=True)(input_layer)
    for i in range(1,n_layers):
        x = Dense(n_neurons[i], activation='relu',  use_bias=True)(x)
    
    output_layer = Dense(nl, activation='linear', use_bias=True)(x)
    
    model = Model(input_layer, output_layer)
    
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[coeff_determination])
    
    model.summary()
    
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
            
    history_callback = model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size, 
                                 validation_data= (x_valid,y_valid),callbacks=callbacks_list)
    
    # plot history
    loss = history_callback.history["loss"]
    val_loss = history_callback.history["val_loss"]
    mse = history_callback.history['coeff_determination']
    val_mse = history_callback.history['val_coeff_determination']
    
    plt.figure()
    epochs = range(1, len(loss) + 1)
    plt.semilogy(epochs, loss, 'b', label='Training loss')
    plt.semilogy(epochs, val_loss, 'r', label='Validationloss')
    plt.title('Training and validation loss')
    plt.legend()
    filename = 'loss.png'
    plt.savefig(filename, dpi = 300)
    plt.show()

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
#filepath = "dnn_best_model.hd5"
saved_model = load_model(filepath, custom_objects={'coeff_determination': coeff_determination})
nx_fdm = 100		# number of intervals in x
lx_fdm = 10.0		# spatial domain lenght
nx_lbm = 100
lx_lbm = 10.0

lx = lx_fdm + lx_lbm

tm = 450.0  	# max time
dt = 0.001	# time step

alpha_test = 1.0

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

u0_fdm = alpha_test*np.tanh(xall_fdm - xc)
u0_lbm = alpha_test*np.tanh(xall_lbm - xc)

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
fequ, feqv = np.zeros((nx_lbm+1,3)), np.zeros((nx_lbm+1,3))
ru_lbm, rv_lbm = np.zeros((nx_lbm+1,3)), np.zeros((nx_lbm+1,3))
ou, ov = np.zeros((nx_lbm+1,3)), np.zeros((nx_lbm+1,3))
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
#    fun[2,2] = un_fdm[nx_fdm]/3.0  # fu[0,0] #
#    fvn[2,2] = vn_fdm[nx_fdm]/3.0 # fv[0,0] #
#    fun[2,2] = fu[0,0] #
#    fvn[2,2] = fv[0,0] #
    
    if j == 1:
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
    else:
        xtest = np.array([un_fdm[nx_fdm],vn_fdm[nx_fdm],fun[2,0] ,fvn[2,0]])
        xtest = xtest.reshape(1,-1)
        xtest_sc = sc_input.transform(xtest)
        ypred_sc = saved_model.predict(xtest_sc)
        ypred = sc_output.inverse_transform(ypred_sc)
        fun[2,2] = ypred[0,0]
        fvn[2,2] = ypred[0,1]
    
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
nx = nx_fdm+nx_lbm
data = np.load(f'../ftcs/solution_fdm_{nx}_{tm}_{alpha_test}.npz')
u_fdm_all = data['u']
v_fdm_all = data['v']

data = np.load(f'../lbm/solution_lbm_{nx}_{tm}_{alpha_test}.npz')
u_lbm_all = data['u']
v_lbm_all = data['v']

del data

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
#fig.savefig(f'dnn_{nx}_{nf}.pdf',dpi=300)
#fig.savefig(f'dnn_{nx}_{nf}.png',dpi=300)

#np.savez(f'dnn_{nx}_{tm}_{nf}' , x=x, t=t, u=u_all, v =v_all)










