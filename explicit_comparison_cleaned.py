import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, ndimage, integrate, sparse
from scipy.ndimage import gaussian_filter
from scipy.ndimage import interpolation
from math import sqrt
import matplotlib.animation as anim
from PIL import Image
import os
import time
import csv
import cv2
import flowutils as flowut
import math
import sobolev as sob
from scipy.io import loadmat
import warnings

def interp( I, XY ):

    x = XY[:,:,0].ravel()
    y = XY[:,:,1].ravel()
    xy = np.array( [x,y] )

#    Iout = ndimage.map_coordinates( I, xy, mode='wrap', order=1 )
    Iout = ndimage.map_coordinates(I, xy, mode='nearest', order=1)
    Iout = Iout.reshape( I.shape )

    return Iout


def dataEvolution( I0, I1, gradI1, phi ):

    dims = I1.shape

    I1_w = interp( I1, phi )
    I1x_w = interp( gradI1[0], phi )
    I1y_w = interp( gradI1[1], phi )
    It = I1_w - I0

    if robust_func == "lorentzian":
        sigma = 1.5/255
        rho_deriv = It / ( 1 + It**2/(2*sigma**2) )
    elif robust_func == "charbonnier":
        eps = 0.001/255
        rho_deriv = It / np.sqrt( It**2 + eps**2 )
    else:
        rho_deriv = It

    devol = np.zeros( [dims[0],dims[1],2] )
    devol[:,:,0] = - rho_deriv * I1x_w
    devol[:,:,1] = - rho_deriv * I1y_w

    return devol


def shiftPhiPeriodic(phi):
    dims = phi.shape

    phi_px = np.vstack((phi[1:, :, :], np.expand_dims(phi[0, :, :], axis=0)))
    phi_px[-1, :, 0] += dims[0]
    phi_mx = np.vstack((np.expand_dims(phi[-1, :, :], axis=0), phi[0:-1, :, :]))
    phi_mx[0, :, 0] -= dims[0]

    phi_py = np.hstack((phi[:, 1:, :], np.expand_dims(phi[:, 0, :], axis=1)))
    phi_py[:, -1, 1] += dims[1]
    phi_my = np.hstack((np.expand_dims(phi[:, -1, :], axis=1), phi[:, 0:-1, :]))
    phi_my[:, 0, 1] -= dims[1]

    return phi_px, phi_mx, phi_py, phi_my


def shiftPhi(phi):
    dims = phi.shape

    phi_px = phi[ np.r_[1:dims[0], dims[0]-1], :, :]
    phi_px[-1, :, 0] += 1
    phi_mx = phi[ np.r_[0,0:(dims[0]-1)], :, :]
    phi_mx[0, :, 0] -= 1

    phi_py = phi[:, np.r_[1:dims[1], dims[1]-1], :]
    phi_py[:, -1, 1] += 1
    phi_my = phi[:, np.r_[0,0:(dims[1]-1)], :]
    phi_my[:, 0, 1] -= 1

    return phi_px, phi_mx, phi_py, phi_my


def Laplacian( phi ):

    phi_px, phi_mx, phi_py, phi_my = shiftPhi(phi)
    Lap_phi = phi_px + phi_mx + phi_py + phi_my - 4 * phi

    return Lap_phi


def energy_grad( I0, I1, gradI1, phi, alpha ):

    return dataEvolution( I0, I1, gradI1, phi ) + alpha * Laplacian( phi )


def energy( I0, I1, phi, alpha ):

    I1_w = interp( I1, phi )

    phi_px, phi_mx, phi_py, phi_my = shiftPhi(phi)
    phi_m_id_x = ( phi_px - phi_mx ) / 2
    phi_m_id_x[:,:,0] -= 1
    phi_m_id_y = ( phi_py - phi_my ) / 2
    phi_m_id_y[:,:,1] -= 1

    It = I1_w - I0
    if robust_func=="lorentzian":
        sigma = 1.5 / 255
        data_term = np.sum( np.log( 1 + It**2/(2*sigma**2) ) )
    elif robust_func=="charbonnier":
        eps = 0.001/255
        data_term = np.sum( np.sqrt( It**2 + eps**2 ) )
    else:
        data_term = np.sum( It**2 )

    return  data_term + alpha * np.sum( phi_m_id_x**2 + phi_m_id_y**2 )


def gradDescent( I0, I1, gradI1, phi, T, alpha, iters ):

    dt = 0.22 / alpha
#    dt = dt if dt < 1 else 1

    phi_k = np.copy(phi)

    for i in range(iters):
        T += dt
        e_grad = energy_grad(I0, I1, gradI1, phi_k, alpha)
        phi_k += dt*e_grad

    return phi_k, T


def computeDataLinearizedFlowOperator(gradI1w, m, n, dt):

    Ix2 = gradI1w[0].flatten()**2 + 1/dt
    Iy2 = gradI1w[1].flatten()**2 + 1/dt
    IxIy = gradI1w[0].flatten() * gradI1w[1].flatten()

    A1 = sparse.spdiags( Ix2, 0, m*n, m*n )
    A2 = sparse.spdiags( IxIy, 0, m*n, m*n )
    A3 = sparse.spdiags( Iy2, 0, m*n, m*n )

    M = sparse.bmat( [ [A1,A2], [A2, A3] ] )

    return M


#row-wise format
def computeLaplaceFlowOperator(m,n):

    data = np.ones((5,m*n))
    data[1,(n-1):(m*n+1):n] = 0
    data[2, 0:(m*n-n+1):n] = 0
    data[3, n*(m-1):(m*n+1)] = 0
    data[4, 0:n] = 0
    data[0,:] = -data[1,:] - data[2,:] - data[3,:] - data[4,:]

    diags = np.array( [0,-1,1,-n,n] )

    a = sparse.spdiags( data, diags, m*n, m*n )
    A = sparse.bmat( [ [a,None], [None,a] ] )

    return A


def computeLinearizedFlowOperator(gradI1w,m,n,alpha,dt):

    return computeDataLinearizedFlowOperator(gradI1w, m, n, dt) - alpha*computeLaplaceFlowOperator(m,n)


def linearizedOpticalFlow(I0, I1, gradI1, phi, v, alpha, iters):

    phi_k = phi.copy()
    v_k = v.copy()

    dims = v.shape
    m = dims[0]
    n = dims[1]
    # code is using 1/dt 
    dt = 100

    for i in range(iters):

        gradI1w = gradI1.copy()
        gradI1w[0] = interp( gradI1[0], phi_k )
        gradI1w[1] = interp( gradI1[1], phi_k )
        I1w = interp( I1, phi_k )
        It = I1w - I0

        A = computeLinearizedFlowOperator( gradI1w, m, n, alpha, dt )

        Lap_phi = Laplacian( phi_k )
        bx = -It * gradI1w[0] + alpha*Lap_phi[:,:,0]
        by = -It * gradI1w[1] + alpha*Lap_phi[:,:,1]
        b = np.hstack( (bx.flatten(), by.flatten()) )

        vk_flat = np.hstack( (v_k[:,:,0].flatten(), v_k[:,:,1].flatten()) )

        uv_incr, info = sparse.linalg.cg(A, b, vk_flat, tol=1e-8)
#        uv_incr = sparse.linalg.spsolve( A, b )

        v_k[:,:,0] = np.reshape( uv_incr[0:m*n], (m,n) )
        v_k[:,:,1] = np.reshape( uv_incr[m*n:], (m,n) )

        phi_k += v_k

    return phi_k, v_k


def computeStepSize( I0, I1, phi ):

    if robust_func == "h&s":
        dt = 0.99 / sqrt(0.75 + 4 * alpha)
    else:
        I1w = interp( I1, phi )
        It = I1w - I0
        if robust_func == "lorentzian":
            sigma = 1.5 / 255
            rho_p = It / ( sigma**2 + 0.5*It**2)
            rho_pp = (sigma**2-It**2/2) / ( (sigma**2 + It**2/2)**2 )
        else:
            eps = 0.001 / 255
            rho_p = It / np.sqrt(It**2 + eps**2)
            rho_pp = eps**2 / ( (It**2 + eps**2)**(3/2) )
        max_rho_p = np.fabs( rho_p ).max()
        max_rho_pp = np.fabs( rho_pp ).max()
        dt = 0.99 / sqrt( 0.25*max_rho_pp + 8*alpha )

    return dt


def computeStepSize2(alpha):
    
    if robust_func == "h&s":
        dt = 2 / sqrt( 0.25*2 + 8*alpha )
    elif robust_func == "lorentzian":
        sigma = 1.5 / 255
        dt = 2 / sqrt( 0.25*(0.5/(sigma**2)-1) + 8*alpha )    
    else:
        eps = 0.001 / 255
        dt = 2 / sqrt( 0.25*( 1/eps ) + 8*alpha )

    return dt*0.8

def computeStepSize3(alpha):  
    if robust_func == "h&s":
        dt = 2 / sqrt(1 + 8*alpha )
    return dt*0.8

def computeStepSize4(alpha):  
    if robust_func == "h&s":
        dt = 2 / sqrt(1/alpha )
    return dt*0.8


def explicitEvol2nd(I0, I1, gradI1, phi, v, T, alpha, iters,lamb):
    dt = computeStepSize3(alpha)

    phi_k = np.copy(phi)
    v_k = np.copy(v)
    for i in range(iters):
#        dt = computeStepSize( I0, I1, phi )

        T += dt
        phi_kp1 = phi_k + dt * v_k
        e_grad = energy_grad(I0, I1, gradI1, phi_kp1, alpha)
#        v_kp1 = v_k + dt * (-3/T * v_k + e_grad)
#        v_kp1 = (v_k + dt * e_grad)/(1+3*dt/T)
        #v_kp1 = (v_k + dt * (-3 / (2*T) * v_k + e_grad)) /(1+dt*3/(2*T))
        v_kp1 = (v_k + dt * (-lamb * v_k + e_grad)) /(1+dt*lamb)
        phi_k = phi_kp1
        v_k = v_kp1

    return phi_k, v_k, T

def explicitEvol( I0, I1, gradI1, phi, v, T, alpha, iters,lamb):

    dt = computeStepSize4(alpha)

    phi_k = np.copy(phi)
    v_k = np.copy(v)
    for i in range(iters):
#        dt = computeStepSize( I0, I1, phi )

        T += dt
        phi_kp1 = phi_k + dt * v_k
        e_grad = energy_grad(I0, I1, gradI1, phi_kp1, alpha)
#        v_kp1 = v_k + dt * (-3/T * v_k + e_grad)
#        v_kp1 = (v_k + dt * e_grad)/(1+3*dt/T)
        #v_kp1 = (v_k + dt * (-3 / (2*T) * v_k + e_grad)) /(1+dt*3/(2*T))
        v_kp1 = (v_k + dt * (-lamb * v_k + e_grad)) /(1+dt*lamb)
        phi_k = phi_kp1
        v_k = v_kp1

    return phi_k, v_k, T


def RK4( I0, I1, gradI1, phi, v, T, alpha, iters ):

    dt = 1.4/sqrt(2*alpha)

    v_k = np.copy(v)
    phi_k = np.copy(phi)

    s = [1,2,2,1]

    for i in range(iters):
        v_kp1 = np.copy(v_k)
        phi_kp1 = np.copy( phi_k )

        a_v = np.zeros ( v.shape )
        a_phi = np.zeros( phi.shape )

        T += dt

        for j in range(len(s)):
            v_step = v_k + a_v / s[j]
            phi_step = phi_k + a_phi / s[j]
            t_step = T + ( dt / s[j] if j==0 else 0 )

            e_grad = energy_grad( I0, I1, gradI1, phi_step, alpha )
            a_v = dt * ( ( -3 / t_step ) * v_step + e_grad )
            a_phi = dt * v_step

            v_kp1 += a_v * s[j] / 6.0
            phi_kp1 += a_phi * s[j] / 6.0

        phi_k = phi_kp1
        v_k = v_kp1

    return phi_k, v_k, T


def RK45( I0, I1, gradI1, phi, v, T, alpha, iters ):

    dt = 1.4 / sqrt(2 * alpha)
    dt = 0.99 / sqrt(0.75 + 4 * alpha)
    dims = phi.shape
    n = phi.size
    y0 = np.hstack( ( phi.flatten(), v.flatten() ) )

    def f(t, y):

        y_phi = y[0:n].reshape( dims )
        y_v = y[n:].reshape( dims )

        y_v_der = -3 / (t + dt) * y_v + energy_grad(I0, I1, gradI1, y_phi, alpha)

        return np.hstack( (y_v.flatten(), y_v_der.flatten()) )

    ode_out = integrate.solve_ivp( f, [T,T+dt*iters], y0, t_eval=[T+dt*iters], method='RK45', vectorized=True, rtol=0.001, atol=1e-6 )
    y = ode_out.y
    T += dt*iters

    return y[0:n].reshape( dims ), y[n:].reshape( dims ), T


def nesterov( I0, I1, gradI1, y, x, lambda_k, alpha, iters ):

    dt = 0.2 / sqrt(2 * alpha)
    dt = 0.15 / alpha

    x_k = np.copy(x)
    y_k = np.copy(y)

    for i in range(iters):
        e_grad = energy_grad( I0, I1, gradI1, x_k, alpha)
        y_kp1 = x_k + dt * e_grad
        lambda_kp1 = ( 1 + sqrt( 1 + 4*lambda_k*lambda_k ) )/2
        gamma_k = (1-lambda_k)/lambda_kp1
        x_kp1 = (1-gamma_k) * y_kp1 + gamma_k * y_k

        y_k = y_kp1
        x_k = x_kp1
        lambda_k = lambda_kp1

    return y_k, x_k, lambda_k 

'''
Start of File Loading Methods 
'''
# Loads, normalizes and scales image files 
def image_load(I0_name,I1_name,scale):
    
    # load images
    I0 = cv2.imread(I0_name)
    I1 = cv2.imread(I1_name)
    
    # Scale Images
    I0 = cv2.resize( I0, None, fx=scale, fy=scale,interpolation=cv2.INTER_AREA )
    I1 = cv2.resize( I1, None, fx=scale, fy=scale,interpolation=cv2.INTER_AREA )
    
    # Convert to Grayscale 
    I0 = cv2.cvtColor( I0, cv2.COLOR_RGB2GRAY )
    I1 = cv2.cvtColor( I1, cv2.COLOR_RGB2GRAY )
    
    # Cast as Type float
    I0 = I0.astype( float )
    I1 = I1.astype( float )
    
    # Normalize 
    Imax = max( I0.max(), I1.max() )
    Imin = max( I0.min(), I1.min() )
    I0 = (I0-Imin)/(Imax-Imin)
    I1 = (I1-Imin)/(Imax-Imin)
    
    return (I0,I1)

# Load flow file 
def load_flow(ground_truth,scale):
    load_flo = loadmat(ground_truth)['reduced']
    load_flo = cv2.resize(load_flo,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    load_flo = load_flo*scale
    return load_flo

# Master file list of files used for performance comparison
def file_list(image_selection):
    
    ground_truth_files = ['ground_truth/whale_flow_10.mat',
                                 'ground_truth/urban_flow_10.mat',
                                 'ground_truth/venus_flow_10.mat',
                                 'ground_truth/grove3_flow_10.mat',
                                 'ground_truth/hydrangea_flow_10.mat',
                                 'ground_truth/dimetrodon_flow_10.mat',
                                 'ground_truth/grove2_flow_10.mat']
    
    I0_files = ['data/other-data/RubberWhale/frame11.png',
                'data/other-data/Urban3/frame11.png',
                'data/other-data/Venus/frame11.png',
                'data/other-data/Grove3/frame11.png',
                'data/other-data/Hydrangea/frame11.png',
                'data/other-data/Dimetrodon/frame11.png',
                'data/other-data/Grove2/frame11.png']
    
    I1_files =  ['data/other-data/RubberWhale/frame10.png',
                'data/other-data/Urban3/frame10.png',
                'data/other-data/Venus/frame10.png',
                'data/other-data/Grove3/frame10.png',
                'data/other-data/Hydrangea/frame10.png',
                'data/other-data/Dimetrodon/frame10.png',
                'data/other-data/Grove2/frame10.png']
    
    ground_truth = []
    I0 = []
    I1 = []
    
    for ii,value in enumerate(image_selection):
        if value == 1:
            ground_truth.append(ground_truth_files[ii])
            I0.append(I0_files[ii])
            I1.append(I1_files[ii])
    
    return(ground_truth,I0,I1)

'''
Errors and Statistics Calculations
'''
def compute_lambda(my_gradient,blur_radius):
    grad_x = my_gradient[0]
    grad_y = my_gradient[1]
    grad_xx = grad_x*grad_x
    grad_yy = grad_y*grad_y
    grad_xy = grad_x*grad_y
    
    blur_xx = gaussian_filter(grad_xx,sigma=blur_radius)
    blur_yy = gaussian_filter(grad_yy,sigma=blur_radius)
    blur_xy = gaussian_filter(grad_xy,sigma=blur_radius)
    eigen_values = []
    img = np.zeros((grad_x.shape[0],grad_x.shape[1]))
    for ii in range(0,grad_x.shape[0]):
        for jj in range(0,grad_x.shape[1]):
            w,v = np.linalg.eig([[ blur_xx[ii][jj],blur_xy[ii][jj] ],[blur_xy[ii][jj],blur_yy[ii][jj]]])
            min_eigen = np.min(w)
            eigen_values.append(min_eigen)
            img[ii][jj] = min_eigen
    
    return np.max(eigen_values)

def build_blank_arrays(list_size):
    array_1 = [[] for _ in range(list_size)]
    array_2 = [[] for _ in range(list_size)]
    array_3 = [[] for _ in range(list_size)]
    array_4 = [[] for _ in range(list_size)]
    return (array_1,array_2,array_3,array_4)

def mean_and_std(vector):
    print('Mean is ' + str(np.mean(vector)))
    print('Std is  ' + str(np.std(vector)))
    return [np.mean(vector),np.std(vector)]

# returns average end point error 
def compute_aee_error(comp_flow,ground_truth_flow):
    occulsion_bound = 1e8
    indices = np.where(np.logical_and(np.abs(ground_truth_flow[:,:,0]) < occulsion_bound, np.abs(ground_truth_flow[:,:,1]) < occulsion_bound))
    comp_flow_x = comp_flow[:,:,0]
    comp_flow_y = comp_flow[:,:,1]
    
    ground_truth_flow_x = ground_truth_flow[:,:,0]
    ground_truth_flow_y = ground_truth_flow[:,:,1]
    
    flow_diff_x = comp_flow_x[indices] - ground_truth_flow_x[indices]
    flow_diff_y = comp_flow_y[indices] - ground_truth_flow_y[indices]
    end_point_error = np.mean(np.sqrt(np.square(flow_diff_x)+np.square(flow_diff_y)))
    
    return end_point_error
    
def compute_aae_error(comp_flow,ground_truth_flow):
    # Attempt at vectorized computation
    limit = 1e8
    indices = np.where(ground_truth_flow[:,:,0] < limit)
    flow_array = np.ones((comp_flow.shape[0],comp_flow.shape[1],3))
    flow_array[:,:,0:2] = comp_flow
        
    truth_array = np.ones((ground_truth_flow.shape[0],ground_truth_flow.shape[1],3))
    truth_array[:,:,0:2]= ground_truth_flow
        
    product = np.einsum('ijk,ijk->ij',flow_array,truth_array)
    norm_1 = np.sqrt(np.einsum('ijk,ijk->ij',flow_array,flow_array))
    norm_2 = np.sqrt(np.einsum('ijk,ijk->ij',truth_array,truth_array))
        
    product = np.multiply(product,1/norm_1,1/norm_2)
    
    product = product[indices]
    indices_fix = np.where((product)>1) 
    product[indices_fix] = 1
    indices_fix = np.where(-1*(product)>1) 
    product[indices_fix] = -1
    
    return np.mean(np.arccos(product))  
    
'''
Main Program Work Flow 
'''
def main_program(I0_name,I1_name,ground_truth,alpha,method,iters,lamda,old_scale,scale,old_flow,use_pyramid):
    
    int_iters = 2
    full_ground_flow = load_flow(ground_truth,1)
    
    (I0,I1) = image_load(I0_name,I1_name,scale)
    gradI1 = np.gradient(I1)
    
    T = 0
    methods = ['Explicit','RK4','Nesterov','Gradient Descent','RK45','Sobolev','Linear Optical Flow']
    
    (m,n) = I0.shape
    phi_x, phi_y = np.mgrid[ range(m), range(n) ].astype(float)
    phi = np.stack((phi_x,phi_y), axis=-1)
    
    if(use_pyramid == 1):
        if(len(old_flow) is not 0):
            old_flow = cv2.resize(old_flow,(phi.shape[1],phi.shape[0]),interpolation=cv2.INTER_AREA)*(scale/old_scale)
            # Debug routine
            phi[:,:,0] = old_flow[:,:,0] + phi_x
            phi[:,:,1] = old_flow[:,:,1] + phi_y
    
    v = np.zeros(phi.shape)
    X, Y = np.meshgrid( np.arange(m), np.arange(n) )

    (e,compute_time,EE_error,AE_error,data_vector) = ([],[],[],[],[])
    (energy_window,flow_norm_window) = (np.zeros(10),np.zeros(10))   
    (counting_index,total_time) = (0,0)
    flow = np.zeros( (m,n,2) )

    for i in range(0,iters):
        start = time.time()
        if method==0:
            phi_new, v, T = explicitEvol(I0, I1, gradI1, phi, v, T, alpha, int_iters,lamda)
        elif method ==6:
            phi_new, v = linearizedOpticalFlow(I0,I1,gradI1, phi, v, alpha, int_iters)
        end = time.time()
        
        total_time = total_time + end-start
        compute_time.append(total_time)
            
        #I1w = interp(I1, phi_new)
        calc_energy = energy(I0, I1, phi, alpha) 
        e.append( calc_energy )
            
        energy_window[counting_index] = calc_energy 
        epsilon = (scale)*5e-3
            
        phi = phi_new
            
        # flow computation
        flow[:,:,0] = phi[:,:,0] - phi_x
        flow[:,:,1] = phi[:,:,1] - phi_y
        comp_flow = flow
           
        # Change flow convention to ground truth 
        temp = comp_flow.copy()
        temp[:,:,0] = -1*comp_flow[:,:,1]
        temp[:,:,1] = -1*comp_flow[:,:,0]
        
        
        temp =  cv2.resize(temp,(full_ground_flow.shape[1],full_ground_flow.shape[0]))/scale
        EE_error.append(compute_aee_error(temp,full_ground_flow))
        AE_error.append(compute_aae_error(temp,full_ground_flow))
            
        #img_flow = flowut.flowToColor(flow)
        #im.set_array( img_flow )
            
        flow_norm_window[counting_index] = np.linalg.norm(comp_flow)
        counting_index += 1 
        counting_index = counting_index % 10
        
        
        
        data_vector = [EE_error[-1],AE_error[-1],total_time,i,phi,comp_flow,EE_error,AE_error]
        
        if(np.std(flow_norm_window) < epsilon):
            return data_vector
    return data_vector 

if __name__ == '__main__': 
    # Ignore Depreciated Plotting Warning
    warnings.filterwarnings("ignore", message="")
    # hacky jumps in pyramid scheme
    scales = [0.03125,0.0625,0.125,0.25,0.5,1.0]
    image_selection = (1,1,1,1,1,1,1)
    lambda_schemes = [1]
    use_pyramid = 1
    # Do not run multiple alphas with more than one scale term and pyramid
    alpha_range = [0.04]
    lambda_image_percentage = .04
    alpha_labels = ['single']
    iters = 2000
    (ground_truth_files,I0_paths,I1_paths) = file_list(image_selection)
    (old_linear_phi,old_explicit_phi,blank_1,blank_2) = build_blank_arrays(len(ground_truth_files))
    (old_linear_flow,old_explicit_flow,blank_1,blank_2) = build_blank_arrays(len(ground_truth_files))
    robust_func = "h&s"
    
    # explicit = 0, RK4 = 1, Nesterov = 2, gradient descent = 3, RK45 = 4, Sobolev = 5, LinearOptFlow = 6
    for nn in lambda_schemes:
        if(use_pyramid == 1):
            csv_name = 'Performance_comparison_pyramid_scheme_' + str(nn) + '.csv'
        if(use_pyramid == 0):
            csv_name = 'Performance_comparison_no_pyramid_scheme_' + str(nn) + '.csv'
        with open(csv_name, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=' ',quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            for jj in range(0,len(scales)):
                
                cur_scale = scales[jj]
                old_scale = scales[jj] if jj==0 else scales[jj-1]
                # build list of empty arrays for processing current instance 
                (end_point_linear,angular_error_linear,comp_time_linear,iterations_linear) = build_blank_arrays(len(alpha_range))
                (end_point_explicit,angular_error_explicit,comp_time_explicit,iterations_explicit) = build_blank_arrays(len(alpha_range))
                
                print('-----------------------------------------')
                print('Running ' + str(cur_scale) +  ' Resolution Scale')
                print('-----------------------------------------')

                csvwriter.writerow(['Resolution:_' + str(cur_scale)])
                for kk in range(0,len(ground_truth_files)):
                    
                    ground_truth = ground_truth_files[kk]
                    I0_name = I0_paths[kk]
                    I1_name = I1_paths[kk]
                    (I0,I1) = image_load(I0_name,I1_name,cur_scale)
    
                    gradI1 = np.gradient(I1)
                    sigma = min(4,np.sqrt(I1.shape[0]*I1.shape[1])*lambda_image_percentage)
                    #sigma = 5
                    my_lambda = compute_lambda(gradI1,sigma)
                    print('Test Lambda Value ' + str(my_lambda))
                    print('Processing Image Pair ' + str(kk))
                    
                    for iii in range(0,len(alpha_range)):
                        alpha_coeff = alpha_range[iii]
                        #alpha_coeff = alpha_range[iii]*(cur_scale*(1/scales[0]))*(cur_scale*(1/scales[0]))
                        if(nn==1): # conditionally compute lambda
                            my_lambda = 8*np.sqrt(np.pi*(alpha_coeff/(I1.shape[0]*I1.shape[1])))
                            
                            print('Current lambda value ' + str(my_lambda))
                        
                        method = 6 # change to 6 for actual comparison
                        data_vector = main_program(I0_name,I1_name,ground_truth,alpha_coeff,method,iters,my_lambda,old_scale,cur_scale,old_linear_flow[kk],use_pyramid)
                        end_point_linear[iii].append(data_vector[0])
                        angular_error_linear[iii].append(data_vector[1])
                        comp_time_linear[iii].append(data_vector[2])
                        iterations_linear[iii].append(data_vector[3])
                        old_linear_phi[kk] = data_vector[4]
                        old_linear_flow[kk] = data_vector[5]
                        
                        # save results linear 
                        temp = old_linear_flow[kk].copy()
                        temp[:,:,0] = -1*old_linear_flow[kk][:,:,1]
                        temp[:,:,1] = -1*old_linear_flow[kk][:,:,0]
                        #plt.figure(111)
                        #plt.imshow(flowut.flowToColor(temp))
                        #plt.show()
                        my_path = 'FlowsLinear/' + I0_name.split('/')[-2]
                        if not os.path.exists(my_path):
                            os.makedirs(my_path)
                        my_str = 'FlowsLinear/' + I0_name.split('/')[-2] + '/' + 'img_flow_'  + str(cur_scale) + '.png'
                        #plt.savefig(my_str)
                        im = Image.fromarray(flowut.flowToColor(temp))
                        im.save(my_str)
                        
                        plt.figure(222)
                        plt.clf()
                        plt.plot(data_vector[6])
                        plt.title('EE Error')
                        new_str = 'FlowsLinear/' + I0_name.split('/')[-2] + '/' + 'EE_error_'  + str(cur_scale) + '.png'
                        plt.savefig(new_str)
                        
                        plt.figure(333)
                        plt.clf()
                        plt.plot(data_vector[7])
                        plt.title('AE_Error')
                        new_str_2 = 'FlowsLinear/' + I0_name.split('/')[-2] + '/' + 'AE_error_'  + str(cur_scale) + '.png'
                        
                        method = 0
                        data_vector = main_program(I0_name,I1_name,ground_truth,alpha_coeff,method,iters,my_lambda,old_scale,cur_scale,old_explicit_flow[kk],use_pyramid)
                        end_point_explicit[iii].append(data_vector[0])
                        angular_error_explicit[iii].append(data_vector[1])
                        comp_time_explicit[iii].append(data_vector[2])
                        iterations_explicit[iii].append(data_vector[3])
                        old_explicit_phi[kk] = data_vector[4]
                        old_explicit_flow[kk]= data_vector[5]
                        
                        # save results explicit 
                        temp = old_explicit_flow[kk].copy()
                        temp[:,:,0] = -1*old_explicit_flow[kk][:,:,1]
                        temp[:,:,1] = -1*old_explicit_flow[kk][:,:,0]
                        #plt.figure(111)
                        #plt.imshow(flowut.flowToColor(temp))
                        #plt.show()
                        my_path = 'FlowsExplicit/' + I0_name.split('/')[-2]
                        if not os.path.exists(my_path):
                            os.makedirs(my_path)
                        my_str = 'FlowsExplicit/' + I0_name.split('/')[-2] + '/' + 'img_flow_'  + str(cur_scale) + '.png'
                        #plt.savefig(my_str)
                        im = Image.fromarray(flowut.flowToColor(temp))
                        im.save(my_str)
                        
                        plt.figure(222)
                        plt.clf()
                        plt.plot(data_vector[6])
                        plt.title('EE Error')
                        new_str = 'FlowsExplicit/' + I0_name.split('/')[-2] + '/' + 'EE_error_'  + str(cur_scale) + '.png'
                        plt.savefig(new_str)
                        
                        plt.figure(333)
                        plt.clf()
                        plt.plot(data_vector[7])
                        plt.title('AE_Error')
                        new_str_2 = 'FlowsExplicit/' + I0_name.split('/')[-2] + '/' + 'AE_error_'  + str(cur_scale) + '.png'
                        plt.savefig(new_str_2)
                            
                        
                '''
                    Generate and Plot Final Results from Dataset
                '''
                print('------------------------------------')
                print('Linear Results')
                print('------------------------------------')
                csvwriter.writerow(['Linearized_Optical_Flow','Alpha_Values','Iterations_(avg):', 
                             'Time_to_Converge_(avg):', 'Time_to_Converge_(std):',
                             'Angular_Error_(avg):','Angular_Error_(std):', 
                             'End_Point_Error_(avg):','End_Point_Error_(std):'])
                
                for labels in range(0,len(alpha_labels)):
                    print(alpha_labels[labels] + ' alpha linear End Point Error')
                    end_point_error = mean_and_std(end_point_linear[labels])
                    print(alpha_labels[labels] + ' alpha linear Angular Error')
                    angular_error = mean_and_std(angular_error_linear[labels])
                    print(alpha_labels[labels] + '  alpha linear total time')
                    comp_time = mean_and_std(comp_time_linear[labels])
                    print(alpha_labels[labels] + '  alpha linear iterations')
                    current_iterations = mean_and_std(iterations_linear[labels])
                    
                    alpha_coeff = alpha_range[labels]
                    #alpha_coeff = alpha_range[labels]*(cur_scale*(1/scales[0]))*(cur_scale*(1/scales[0]))
                    csvwriter.writerow([alpha_labels[labels],alpha_coeff,str(current_iterations[0])
                    ,str(comp_time[0]),str(comp_time[1]),str(angular_error[0])
                    ,str(angular_error[1]),str(end_point_error[0]),str(end_point_error[1])])
                
                print('------------------------------------')
                print('Explicit Results')
                print('------------------------------------')
                csvwriter.writerow(['Explicit','Alpha_Values','Iterations_(avg):', 
                             'Time_to_Converge_(avg):', 'Time_to_Converge_(std):',
                             'Angular_Error_(avg):','Angular_Error_(std):', 
                             'End_Point_Error_(avg):','End_Point_Error_(std):','Speed_Up'])
        
                for labels in range(0,len(alpha_range)):
                    print(alpha_labels[labels] + ' alpha explicit End Point Error')
                    end_point_error = mean_and_std(end_point_explicit[labels])
                    print(alpha_labels[labels] + ' alpha explicit Angular Error')
                    angular_error = mean_and_std(angular_error_explicit[labels])
                    print(alpha_labels[labels] + '  alpha explicit total time')
                    comp_time = mean_and_std(comp_time_explicit[labels])
                    print(alpha_labels[labels] + '  alpha explicit iterations')
                    current_iterations = mean_and_std(iterations_explicit[labels])
                    performance_ratio = (np.mean(comp_time_linear[labels]))/(np.mean(comp_time_explicit[labels]))
                    print('Performance Ratio: ' + str(performance_ratio))
                    alpha_coeff = alpha_range[labels]
                    #alpha_coeff = alpha_range[labels]*(cur_scale*(1/scales[0]))*(cur_scale*(1/scales[0]))
                    csvwriter.writerow([alpha_labels[labels],alpha_coeff,str(current_iterations[0])
                    ,str(comp_time[0]),str(comp_time[1]),str(angular_error[0])
                    ,str(angular_error[1]),str(end_point_error[0]),str(end_point_error[1]),str(performance_ratio)])