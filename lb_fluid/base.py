import numpy as np
from numba import njit


class D2Q9FluidNVT(object):
    def __init__(self, shape, dt, dr, visc, u, rho, force, test_case=None):
        '''
        Object to describe and update the fluid grid for a D2Q9 lattice Boltzmann simulation.
        shape: shape of the fluid grid (3D)
        dt: time step
        dr: lattice spacing
        tau: kinematic viscosity
        u: initial velocity
        rho: initial density
        force: force field
        '''
        self.shape = shape
        self.dr = dr
        self.dt = dt
        self.c = np.array([[0,0],[0,-1],[0,1],[1,0],[1,-1],[1,1],[-1,0],[-1,-1],[-1,1]])
        self.cs = self.c * dr / dt
        self.w = np.array([4/9,1/9,1/9,1/9,1/36,1/36,1/9,1/36,1/36])
        self.s = np.sqrt(1/3) * dr / dt
        self.s2 = (dr / dt)**2 / 3
        self.tau = visc / (dt * self.s2) + 0.5 # dimesionless relaxation time
        self.c1 = 1
        self.c2 = 1/(self.s2)
        self.c3 = 1/(2*self.s2**2)
        self.c4 = 1/(2*self.s2)

        # initialize arrays
        self.f = np.zeros((self.shape[0],self.shape[1],self.shape[2]))
        self.feq = np.zeros((self.shape[0],self.shape[1],self.shape[2]))
        self.rho = np.ones((self.shape[1],self.shape[2]))
        self.u = np.ones((shape[1],shape[2], 2))
        self.u2 = np.zeros((shape[1],shape[2]))
        self.cu = np.zeros((shape[1],shape[2]))
        self.imp = np.zeros((shape[1],shape[2], 2))
        self.flow_force = np.zeros((shape[1],shape[2], 2))
        
        # set rho field
        self.rho *= rho #* self.dr**3

        # set u field
        if test_case == 'taylor-green':
            for i in range(self.shape[1]):
                for j in range(self.shape[2]):
                    self.u[i,j,0] = u[0] * np.sin(i / (2*np.pi)) * np.cos(j / (2*np.pi))
                    self.u[i,j,1] = -u[1] * np.cos(i / (2*np.pi)) * np.sin(j / (2*np.pi))
        else:
            self.u[:,:,0] = u[0] * np.sin()
            self.u[:,:,1] = u[1]

        # set flow force field
        self.flow_force[:,:,0] = force[0]
        self.flow_force[:,:,1] = force[1]

        # set feq field
        self.u2 = np.sum(self.u*self.u,axis=-1)
        self.update_feq()
        if np.any(self.feq<0):
            raise ValueError('Error: negative feq (minimum: {})'.format(np.min(self.feq)))

        # set f field
        self.f = np.copy(self.feq)

        # stream, update macroscopic and feq fields
        self.stream()
        self.update_macro()
        self.update_feq()


    def update_macro(self): # loop implementation: testing.ipynb, OOP Testing, first cell
        self.rho = np.sum(self.f,axis=0)
        self.u = np.sum(self.cs[:,None,None,:]*self.f[:,:,:,None],axis=0)
        self.u += self.dt * (self.imp + self.flow_force) / 2
        self.u = np.where(self.rho[:,:,None]!=0, self.u/self.rho[:,:,None], 0)
        self.u2 = np.sum(self.u**2,axis=-1)
        self.cu = np.sum(self.cs[:,None,None,:] * self.u[None,:,:,:], axis=(-1))
        return

    def update_feq(self): # loop implementation: testing.ipynb, OOP Testing, second cell
        self.feq = self.rho[None,:,:] * self.w[:,None,None] * \
                  (self.c1 + self.c2*self.cu + self.c3*self.cu**2 - self.c4*self.u2)
        if np.any(self.feq<0):
            raise ValueError('Error: negative feq (minimum: {})'.format(np.min(self.feq)))
        return
    
    def update_f_guo(self): # pretty well vectorized
        f_source = (1 - self.dt / (2*self.tau))
        f_source *= self.w
        f_source = f_source[:,None,None] * \
            np.sum(((self.cs[:,None,None,:]-self.u[None,:,:,:])/self.s2 + \
                    (self.cu/self.s2**2)[:,:,:,None]*self.cs[:,None,None,:]) * \
                    (self.imp[None,:,:,:]+self.flow_force[None,:,:,:]), axis=3)
        self.f += (self.feq - self.f) / self.tau + self.dt * f_source
        return

    def update_f_he(self): # pretty well vectorized
        f_source = (1 - self.dt / (2*self.tau))
        f_source *= np.sum(((self.cs[:,None,None,:]-self.u[None,:,:,:])/self.s * \
                            (self.imp[None,:,:,:] + self.flow_force[None,:,:,:])),axis=3)
        f_source = np.where(self.rho[None,:,:]!=0,
                            self.feq*f_source/self.rho[None,:,:],
                            np.zeros_like(self.feq))
        return

    def update_imp(self, lip_vel, idx, idx_n, dxy, gamma):
        u00 = self.u[idx[:,0],idx[:,1],:] * (1-dxy)[:,0,None] * (1-dxy)[:,1,None]
        u01 = self.u[idx_n[:,0], idx[:,1],:] * (dxy[:,0,None]) * (1-dxy)[:,1,None]
        u10 = self.u[idx[:,0], idx_n[:,1],:] * (1-dxy)[:,0,None] * (dxy[:,1,None])
        u11 = self.u[idx_n[:,0], idx_n[:,1],:] * (dxy[:,0,None]) * (dxy[:,1,None])
        if_vel = u00 + u01 + u10 + u11
        imp_on_lip = -gamma * (lip_vel[:,:2] - if_vel)
        imp_cont_00 = -imp_on_lip * (1-dxy)[:,0,None] * (1-dxy)[:,1,None]
        imp_cont_01 = -imp_on_lip * dxy[:,0,None] * (1-dxy)[:,1,None]
        imp_cont_10 = -imp_on_lip * (1-dxy)[:,0,None] * dxy[:,1,None]
        imp_cont_11 = -imp_on_lip * dxy[:,0,None] * dxy[:,1,None]
        for i in range(len(imp_on_lip)):
            self.imp[idx[i,0],idx[i,1],:] += imp_cont_00[i]
            self.imp[idx_n[i,0],idx[i,1],:] += imp_cont_01[i]
            self.imp[idx[i,0],idx_n[i,1],:] += imp_cont_10[i]
            self.imp[idx_n[i,0],idx_n[i,1],:] += imp_cont_11[i]
        return imp_on_lip
    
    def stream(self):
        for i in range(self.shape[0]):
            self.f[i] = np.roll(self.f[i], axis=(1,0), shift=self.c[i].astype(int))
        return

def get_lipid_indices(lipid_grid, dr, dim, num_dr):
    xy = np.zeros((lipid_grid.shape[0],2))
    xy[:,0] = np.where(lipid_grid[:,0] - dr/2 < 0, lipid_grid[:,0] - dr/2 + dim, lipid_grid[:,0] - dr/2)
    xy[:,1] = np.where(lipid_grid[:,1] - dr/2 < 0, lipid_grid[:,1] - dr/2 + dim, lipid_grid[:,1] - dr/2)
    dxy = (xy / dr) - (xy // dr)
    idx = (xy / dr).astype(int) % num_dr
    idx_n = (idx + 1) % num_dr
    return idx, idx_n, dxy
