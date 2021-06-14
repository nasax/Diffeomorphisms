import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, ndimage, integrate, sparse
from scipy.interpolate import griddata


def dataEvolution2( I0, I1, gradI1, psi ):

    dims = I1.shape
    #print(I0.shape)
    I0_w = interp( I0, psi )

    grad_psi1 = np.gradient( psi[:,:,0] )
    grad_psi2 = np.gradient( psi[:,:,1] )
    det_grad_psi = np.absolute( grad_psi1[0] * grad_psi2[1] - grad_psi1[1] * grad_psi2[0] )

    devol = np.zeros( [dims[0],dims[1],2] )
    devol[:,:,0] = -(I1 - I0_w) * gradI1[0] * det_grad_psi
    devol[:,:,1] = -(I1 - I0_w) * gradI1[1] * det_grad_psi

    return devol


def computeSobolevOperator( m, n, alpha ):

    N = m*n
    A = sparse.lil_matrix( (N, N) )

    p = 0
    for j in range(n):
        for i in range(m):
            num_neighbors = 0
            if i != m-1:
                A[p,p+1] = -alpha
                num_neighbors += 1
            if i != 0:
                A[p,p-1] = -alpha
                num_neighbors += 1
            if j != n-1:
                A[p,p+m] = -alpha
                num_neighbors += 1
            if j !=  0:
                A[p,p-m] = -alpha
                num_neighbors += 1
            A[p,p] = 1 + alpha * num_neighbors
            p += 1

    A = A.tocsc()

    return A


def updatePsi( psi, G, dt ):

    psi_px, psi_mx, psi_py, psi_my = shiftPhi(psi)
    psi_x = (G[:, :, [0,0]] < 0) * (psi_px - psi) + (G[:, :, [0,0]] >= 0) * (psi - psi_mx)
    psi_y = (G[:, :, [1,1]] < 0) * (psi_py - psi) + (G[:, :, [1,1]] >= 0) * (psi - psi_my)

    return psi - dt * ( G[:,:,[0,0]] * psi_x + G[:,:,[1,1]] * psi_y )


def sobolevGradDescent( I0, I1, gradI1, phi, psi, SobOp, Sobgrad, T, alpha, iters ):

    dims = I1.shape
    phi_k = np.copy(phi)
    psi_k = np.copy(psi)

    Sobgrad_x = Sobgrad[:,:,0].flatten('F')
    Sobgrad_y = Sobgrad[:,:,1].flatten('F')

    Sobgrad_w = np.zeros( Sobgrad.shape )

    for i in range(iters):
        L2grad = dataEvolution2( I0, I1, gradI1, psi_k )
        print(Sobgrad_x.shape)
        # current error on this line
        Sobgrad_x, info_x = sparse.linalg.cg( SobOp, L2grad[:,:,0].flatten('F'), Sobgrad_x, tol=1e-8 )
        Sobgrad_y, info_y = sparse.linalg.cg( SobOp, L2grad[:,:,1].flatten('F'), Sobgrad_y, tol=1e-8 )
        Sobgrad[:,:,0] = np.reshape( Sobgrad_x, (dims[0], dims[1]), order='F' )
        Sobgrad[:,:,1] = np.reshape( Sobgrad_y, (dims[0], dims[1]), order='F' )

        plotGrads( L2grad, Sobgrad )

        maxG = Sobgrad.max()
        dt = 0.2 / maxG if abs(maxG) > 1e-6 else 0

        Sobgrad_w[:,:,0] = interpolate.interp( Sobgrad[:,:,0], phi_k )
        Sobgrad_w[:,:,1] = interpolate.interp( Sobgrad[:,:,1], phi_k )

        phi_k += dt * Sobgrad_w
        psi_k = updatePsi( psi_k, Sobgrad, dt )
        T += dt

    return phi_k, psi_k, Sobgrad, T

def plotGrads( L2grad, Sobgrad ):

    fig = plt.figure(3)
    sp0 = plt.subplot(2, 2, 1)
    plt.imshow( L2grad[:,:,0], cmap='gray')
    sp0.set(title='L2Grad x')

    sp1 = plt.subplot(2, 2, 2)
    plt.imshow(L2grad[:, :, 1], cmap='gray')
    sp1.set(title='L2Grad y')

    sp2 = plt.subplot(2, 2, 3)
    plt.imshow( Sobgrad[:,:,0], cmap='gray')
    sp2.set(title='SobGrad x')

    sp3 = plt.subplot(2, 2, 4)
    plt.imshow(Sobgrad[:, :, 1], cmap='gray')
    sp3.set(title='SobGrad y')

    plt.show()
    
def interp( I, XY ):

    x = XY[:,:,0].ravel()
    y = XY[:,:,1].ravel()
    xy = np.array( [x,y] )

#    Iout = ndimage.map_coordinates( I, xy, mode='wrap', order=1 )
    Iout = ndimage.map_coordinates(I, xy, mode='nearest', order=1)
    Iout = Iout.reshape( I.shape )

    return Iout

def gradientFlowOperator():
    A = []
    return A