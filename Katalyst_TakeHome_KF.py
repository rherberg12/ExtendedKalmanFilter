import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read in measurements
df = pd.read_csv('measurements.csv', index_col = None, header = 'infer')

# Organize measurements to each respective source
gps_measurements = df[df['sensor_id'] == 'gps_measurement'].drop(['sensor_id','ra_deg','dec_deg','ra_rate_deg/s','dec_rate_deg/s'], axis = 1)
ground_observer_1 = df[df['sensor_id'] == 'ground_observer_1'].drop(['sensor_id','r_x_km','r_y_km','r_z_km','v_x_km/s','v_y_km/s','v_z_km/s'], axis = 1)
ground_observer_2 = df[df['sensor_id'] == 'ground_observer_2'].drop(['sensor_id','r_x_km','r_y_km','r_z_km','v_x_km/s','v_y_km/s','v_z_km/s'], axis = 1)
gps_measurements.reset_index(drop=True, inplace=True)
ground_observer_1.reset_index(drop=True, inplace=True)
ground_observer_2.reset_index(drop=True, inplace=True)

# Define geodetic position of ground stations
G1 = np.array([-111.536,35.097,2.206]) # [degrees,degrees,km] (lat,long,height)
G2 = np.array([-70.692,-29.016,2.380]) # [degrees,degrees,km] (lat,long,height)

# Define noise covariances matrices for sensors
R_gps = np.eye(6) * np.array([[(5000**2)*10**-3],[(5000**2)*10**-3],[(5000**2)*10**-3],[(0.5**2)*10**-3],[(0.5**2)*10**-3],[(0.5**2)*10**-3]]) # [km, km/s] Satellite local GPS
R_gs1 = np.eye(4) * np.array([[1],[1],[0.01],[0.01]]) # [arcSec^2, arcSec^2/sec^2] Ground station with electro-optical sensor, 1
R_gs2 = np.eye(4) * np.array([[0.01],[0.01],[0.0001],[0.0001]]) # [arcSec^2, arcSec^2/sec^2] Ground station with electro-optical sensor, 2

# Kalman Filter Input Matrices:
sig_qr = 1E-3
sig_qv = 1E-6
Q_gps = np.block([[sig_qr**2*np.eye(3),np.zeros((3,3))],[np.zeros((3,3)),sig_qv**2*np.eye(3)]]) # State model noise covariance (6x6)
Q_gps = Q_gps.reshape(6,6)
M_gps = np.eye(6) # Process noise (state model noise) mapping matrix
# When Q is large, the Kalman Filter tracks large changes in
# the sensor measurements more closely than for smaller Q
u_gps = 0 # Assumption 1: no control
# eom_fun_gps = eom_orbit_controlled_ECI_w_covariance
eom_options_gps = [Q_gps,M_gps,u_gps]
H_gps = np.eye(6) # mapping matrix for state estimates to measurement
L_gps = np.eye(6) # Sensor noise mapping matrix to measurement

# Local Functions:

def EKF(dk, z_k, x_k_minus_1, u_k_minus_1, P_k_minus_1, H_k, L_k, R_k, eom_fun, eom_options):
    # dk: time interval in seconds, z_k: observation at k, x_k_minus_1: state estimate at k-1, 
    # u_k_minus_1: control vector at k-1, P_k_minus_1: state covariance estimate at k-1
    # eom_fun: function to be integrated for state estimate, eom_options: required variables for eom_fun

    # Options:
    integration_dt = 10 # integration time step [s]

    # Calculate state estimate based on x_k_minus_1 and assume no control input. Use rk4 integration to step through to next estimate
    integ_points = int(dk/integration_dt)
    if integ_points < 1.0: # Not significant enough of a time difference to initiate new estimate, adjust integration_dt to resolve
        x_k = x_k_minus_1
        P_k = P_k_minus_1
        P_k = P_k.reshape(len(x_k_minus_1),len(x_k_minus_1))
    else:
        integ_time = np.linspace(0,dk,integ_points) # Isolated time vector for integration 0->dk
        # Propogate state and covariance matrices
        integ_x = np.array(x_k_minus_1[:])
        integ_x = np.append(integ_x,P_k_minus_1.reshape(len(x_k_minus_1),len(x_k_minus_1))) # Isolated state for integration, steps from x_k_minus_1 -> x_k
        for i in range(integ_points-1):
            integ_x_res = rk4(eom_fun, integ_time[i+1]-integ_time[i], integ_time[i], integ_x, eom_options)
            integ_x = integ_x_res
        x_k = integ_x[:len(x_k_minus_1)]
        P_k = integ_x[len(x_k_minus_1):]
        P_k = P_k.reshape(len(x_k_minus_1),len(x_k_minus_1))

    residual = z_k - (H_k @ x_k)
    S_k = H_k @ P_k @ np.transpose(H_k) + L_k @ R_k @ np.transpose(L_k)
    K_k = P_k @ np.transpose(H_k) @ np.linalg.pinv(S_k)

    x_k = x_k + (K_k @ residual)
    P_k = P_k - (K_k @ H_k @ P_k)

    corr = (K_k @ residual)

    return [x_k,P_k]


def eom_orbit_controlled_ECI_w_covariance(t, X, vars):
    # X[0:7] is state position and velocity, X[7:end] is covariance matrix (6x6)
    mu = 3.986*10**5 # km^3/s^2
    Qs = vars[0] # State model noise covariance matrix
    M = vars[1] # Noise covariance matrix mapping matrix
    u = vars[2] # Control vector

    r_vec = X[:3]
    r_norm = np.linalg.norm(r_vec)
    v_vec = X[3:6]
    ## Use completed set if need to perform coordinate transform to apply control vector
    #r_hat = r_vec/r_norm
    #n_hat = np.cross(r_vec,v_vec)/np.linalg.norm(np.cross(r_vec,v_vec))
    #t_hat = np.cross(n_hat,r_hat)/np.linalg.norm(np.cross(n_hat,r_hat))

    P = X[6:].reshape(6,6)
    Gr = mu/np.linalg.norm(r_vec)**5*(3*(r_vec @ np.transpose(r_vec))-r_norm**2 * np.eye(3))
    F = np.block([[np.zeros((3,3)), np.eye(3)],[Gr, np.zeros((3,3))]])
    dP = F @ P @ np.transpose(F) + M @ Qs @ np.transpose(M)
    # F*P+P*F.T simplifies computation load from true linear propogation equation, F*P*F.T
    # Linearization of the nonlinear dynamics was computed via the jacobian in Gr.

    dstate = np.array([v_vec,-mu*r_vec/r_norm**3 + u])

    dP = dP.reshape(36)
    dX = np.append(dstate,dP)
    return dX

def rk4(fun,dt,tk,xk,fun_vars): # fun_vars includes the control input and variables required for the eoms

    f1 = fun(tk,xk,fun_vars)
    f2 = fun(tk+dt/2,xk+(dt/2)*f1,fun_vars)
    f3 = fun(tk+dt/2,xk+(dt/2)*f2,fun_vars)
    f4 = fun(tk+dt,xk+(dt)*f3,fun_vars)
    
    xout = xk + (dt/6)*(f1+2*f2+2*f3+f4)
    return xout

def gps_EKF_calculation():
    # Operate kalman filter through measurements to define estimates state vector
    x_k_minus_1 = gps_measurements.iloc[0,1:]
    z_k_minus_1 = x_k_minus_1
    t0 = gps_measurements.iloc[0,0] # Initial time
    u_k_minus_1 = eom_options_gps[2] # Assumption 1: no control
    P_k_minus_1 = np.eye(6)*np.array([[1.0],[1.0],[1.0],[0.01],[0.01],[0.01]])
    P_k_minus_1 = P_k_minus_1.reshape((len(x_k_minus_1)*len(x_k_minus_1)),1)
    H_k_minus_1 = H_gps
    L_k_minus_1 = L_gps
    R_k_minus_1 = R_gps
    eom_options_k_minus_1 = eom_options_gps
    x_array = np.array(x_k_minus_1[:])
    time_array = np.array([t0])

    for k in range(len(gps_measurements.index)-1):
        dk = gps_measurements.iloc[k+1,0] - gps_measurements.iloc[k,0] # Time step
        x_estimate_k, P_estimate_k = EKF(dk,z_k_minus_1,x_k_minus_1,u_k_minus_1,P_k_minus_1,H_k_minus_1,L_k_minus_1,R_k_minus_1,eom_orbit_controlled_ECI_w_covariance,eom_options_k_minus_1)
        x_array = np.append(x_array,x_estimate_k[:])
        time_array = np.append(time_array,time_array[-1]+dk)
        x_k_minus_1 = x_estimate_k
        P_k_minus_1 = P_estimate_k
    
    # Post-Process
    ax = plt.figure().add_subplot(projection='3d')
    x_array = x_array.reshape(len(time_array),6)
    x_array = pd.DataFrame(x_array, index = time_array, columns=['r_x_km','r_y_km','r_z_km','v_x_km/s','v_y_km/s','v_z_km/s'])
    gps_measurements.set_index('time',inplace=True)
    correction = x_array.subtract(gps_measurements)
    a_gps = np.zeros((len(gps_measurements),1))
    a_EKF = np.zeros((len(gps_measurements),1))
    for i in range(len(a_gps)):
        a_gps[i] = np.sqrt(gps_measurements.iloc[i,0]**2 + gps_measurements.iloc[i,1]**2 + gps_measurements.iloc[i,2]**2)
        a_EKF[i] = np.sqrt(x_array.iloc[i,0]**2 + x_array.iloc[i,1]**2 + x_array.iloc[i,2]**2)
    ax.plot(gps_measurements['r_x_km'],gps_measurements['r_y_km'],gps_measurements['r_z_km'], label = 'gps')
    ax.plot(x_array['r_x_km'],x_array['r_y_km'],x_array['r_z_km'],label = 'EKF')
    plt.legend()
    plt.style.use('ggplot')
    plt.show()

gps_EKF_calculation()

    

