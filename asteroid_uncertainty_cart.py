import spiceypy as spice
import numpy as np
import numpy as n
import matplotlib.pyplot as plt
import scipy.constants as sc

# 1. Load necessary kernels
spice.furnsh("naif0012.tls")
spice.furnsh("/home/j/src/maarsy_meteors/daniel/de430.bsp")       # Or another planetary ephemeris file
spice.furnsh("pck00010.tpc")     # Planetary constants kernel
spice.furnsh("earth_latest_high_prec.bpc")

mu_sun = 1.32712440018e11  # Gravitational parameter of Sun in km^3/s^2

epoch_et0 = spice.str2et("2025-05-05T00:00:00")

def observe_asteroid(astate,
                     epoch_et0,
                     epoch_et):

    # kep (q AU, log10(e), i, Omega, omega, nu)
    #                       300000000,     # Semi-major axis in km (e.g., 2 AU ≈ 300 million km)
    #   0.1,           # Eccentricity
    #    np.radians(10),     # Inclination
    #np.radians(80),     # RAAN
    # np.radians(45),     # Argument of periapsis
    #  np.radians(30),      # Mean anomaly
    #   epoch_et,
    #    mu_sun
    
    # 2. Observer info (Geodetic coordinates)
    # Example: lat/lon in degrees, height in km
    lat = 69.6496      # Tromsø
    lon = 18.9560
    alt = 0.01         # 10 meters above sea level
    
    # Convert to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    # Convert to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # Time of observation
#    utc = '2025-05-05T00:00:00'
 #   et = spice.str2et(utc)

    earth_re=spice.bodvrd("EARTH", "RADII", 3)[1][0]
#    print(earth_re)
    earth_flattening = 1 / 298.257223563
 #   print(earth_flattening)

    # Compute position of the observer relative to Earth center
    # Output in ITRF93 frame, can convert later
    pos = spice.georec(lon_rad, lat_rad, alt, earth_re, earth_flattening)
    #  print(pos)
    xform = spice.sxform("ITRF93", "J2000", epoch_et)
    xform=np.ascontiguousarray(xform[:3,:3])
    obs_j2000 = spice.mxvg(xform, pos)

    # Convert to IAU_EARTH frame
    obs_pos = spice.recgeo(pos, earth_re, earth_flattening)
    # print(obs_pos)
    earth_pos, _ = spice.spkpos("EARTH", epoch_et, "J2000", "NONE", "SUN")
    obs_helio_j2000 = [earth_pos[i] + obs_j2000[i] for i in range(3)]
    
    # Transform observer position to J2000 frame
    state = spice.sxform("ITRF93", "J2000", epoch_et)

    # Convert to heliocentric ecliptic (IAU76/J2000)
    rotation = spice.pxform("J2000", "ECLIPJ2000", epoch_et)
    rotation=np.ascontiguousarray(rotation)
    obs_helio_j2000=np.ascontiguousarray(obs_helio_j2000)
    obs_helio_eclip = spice.mxv(rotation, obs_helio_j2000)
    # print(state)
    #matrix = np.ascontiguousarray(state[:3, :3])
    #vector = np.ascontiguousarray(pos)
    #observer_j2000 = spice.mxv(matrix,vector)

#    print("Observer position in J2000 frame (km):", observer_j2000)

    # 3. Define Keplerian elements of the asteroid
    # [semi-major axis (km), eccentricity, inclination (rad), 
    #  longitude of ascending node (rad), argument of periapsis (rad), mean anomaly (rad)]
    # Convert to Cartesian state vector

    # Epoch of these elements
    # get kep el
    kep_elements = spice.oscelt(astate, epoch_et0, mu_sun)
    # propagate state
    state_asteroid = spice.conics(kep_elements, epoch_et)
    
    return(obs_helio_eclip,state_asteroid)

#    print("Asteroid state in J2000 (km, km/s):", state_asteroid)

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)  # inclination
    phi = np.arctan2(y, x)    # azimuth
    return(r, theta, phi)

def simulate_measurement(state,#=n.array([2*sc.au/1e3,0.3,20,80,45,30]),
                         epoch_et0,
                         times):#=epoch_et0+n.arange(100)*7*24*3600):
    ap=[]
    decs=[]
    azs=[]
    rs=[]
    for i in range(len(times)):
        obs_pos,ast_pos=observe_asteroid(state,epoch_et0,epoch_et=times[i])
        ap.append(ast_pos)
        rel_pos=ast_pos[:3]-obs_pos
        rr=n.linalg.norm(rel_pos)
        r,theta,phi=cartesian_to_spherical(rel_pos[0],rel_pos[1],rel_pos[2])
        rs.append(rr)
        decs.append(theta)
        azs.append(phi)
    rs=n.array(rs)
    decs=n.array(decs)
    azs=n.array(azs)
    ap=n.array(ap)
    return(rs,decs,azs,ap)



def estimate_keplerian(obs_range,
                       obs_dec,
                       obs_az,
                       state_guess=n.array([2.1,0.25,23,82,48,27]),
                       use_range=True,
                       angle_std=5e-6,
                       range_std=1,                       
                       times=epoch_et0+n.arange(100)*7*24*3600):
    def ss(x):
        model_range,model_dec,model_az,model_ap=simulate_measurement(x,epoch_et0,times=times)
        s=0.0
        s+=n.sum((n.angle(n.exp(1j*model_dec)*n.exp(-1j*obs_dec))**2))/angle_std**2
        s+=n.sum((n.angle(n.exp(1j*model_az)*n.exp(-1j*obs_az))**2))/angle_std**2
        if use_range:
            s+=n.sum((obs_range-model_range)**2.0)/range_std**2
        print(s)
        return(s)
    import scipy.optimize as sio
#    xhat=sio.minimize(ss,state_guess,method="Nelder-Mead",tol=1e-6,options={'maxfev':10000,'maxiter':10000,'disp': True}).x#,bounds=[(-3,2),(-15,0),(0,180),(
    xhat=sio.minimize(ss,state_guess,method="Powell",tol=1e-6,options={'maxfev':10000,'maxiter':10000,'disp': True}).x#,bounds=[(-3,2),(-15,0),(0,180),(    
#    xhat=sio.minimize(ss,state_guess,method="Nelder-Mead",tol=1e-6,options={'maxfev':10000,'maxiter':10000,'disp': True}).x#,bounds=[(-3,2),(-15,0),(0,180),(    
#    xhat=sio.minimize(ss,xhat,method="BFGS",tol=1e-6,options={'maxiter':10000,'disp': True}).x#,bounds=[(-3,2),(-15,0),(0,180),(    
#    xhat=sio.minimize(ss,xhat,method="Nelder-Mead").x#,bounds=[(-3,2),(-15,0),(0,180),(    
 #   print(xhat)
#    print(ss(xhat))
    return(xhat)


def get_initial_state(kep, epoch_et0): # a (AU),e,i (deg),O (deg),o (deg),nu (deg)
    kep_elements = [
        (sc.au/1e3)*kep[0]*(1.0-kep[1]),     # PERIFOCAL DISTANCE = (a*(1-e)))#Semi-major axis in km (e.g., 2 AU ≈ 300 million km)
        kep[1],           # Eccentricity
        np.radians(kep[2]),     # Inclination (rad)
        np.radians(kep[3]),     # RAAN (rad)
        np.radians(kep[4]),     # Argument of periapsis (rad)
        np.radians(kep[5]),      # Mean anomaly (rad)
        epoch_et0,   # epoch of elements
        mu_sun
    ]
    state_asteroid = spice.conics(kep_elements, epoch_et0)
#    print(state_asteroid)
    return(state_asteroid)

# (2025 LB)
#times=epoch_et0+29*24*3600 - n.concatenate((n.arange(30)*6*3600,[365*24*3600]))
times=epoch_et0+30*24*3600 - n.arange(100)*24*3600

kep_true=n.array([2.158504572325312, # a (AU)
                  0.5573199074979915, # e
                  7.977400997581499,  # inc
                  73.8364986266037,   # node 
                  212.6277929093889,  # aop
                  342.498256445095])  # mean anom at epoch

state_true=get_initial_state(kep_true, epoch_et0)
state_guess=state_true+0.001*n.array([1e4,1e4,1e4,0.1,0.1,0.1])
#print(state_true)
obs_range,obs_dec,obs_az,ap=simulate_measurement(state_true,epoch_et0,times=times)
lunar_dist=384400e3
plt.plot(times,obs_range*1e3/lunar_dist,".")
plt.xlabel("Time (J2000 sec)")
plt.ylabel("Range (LD)")
plt.show()

angle_std=5e-6
range_std=1               
range_err=n.random.randn(len(obs_range))*range_std
dec_err=n.random.randn(len(obs_range))*angle_std
az_err=n.random.randn(len(obs_range))*angle_std

xhat=estimate_keplerian(obs_range+range_err,obs_dec+dec_err,obs_az+az_err,state_guess=state_guess,times=times,use_range=False)
#xhat[1]=10**xhat[1]
print("Estimated Keplerian parameters using angles only x,y,z,vx,vy,vz (km and km/s)")
print(xhat)
print("Error")
print(xhat-state_true)
#print("Error (percent)")
#print( 100*(kep_true-xhat)/kep_true)

xhat=estimate_keplerian(obs_range+range_err,obs_dec+dec_err,obs_az+az_err,state_guess=state_guess,times=times,use_range=True)
#xhat[1]=10**xhat[1]

print("Estimated Keplerian parameters using angles and range x,y,z,vx,vy,vz (km and km/s)")
print(xhat)
print("Error")
print(xhat-state_true)
#print("Error (percent)")
#print( 100*(kep_true-xhat)/kep_true)


#print(xhat)
#print(10**xhat[1])
print("True Keplerian parameters x,y,z,vx,vy,vz (km and km/s)")
print(state_true)

# estimate jacobian
state_dx=n.array([1,1,1,1,1,1])*1e-3
nm=len(obs_range)
# without range
J0=n.zeros([2*nm,6])
# with range
J1=n.zeros([3*nm,6])

for i in range(6):
    state_pert=n.copy(state_true)
    state_pert[i]=state_true[i]+state_dx[i]
    obs_range_dx,obs_dec_dx,obs_az_dx,ap=simulate_measurement(state_pert,epoch_et0=epoch_et0,times=times)
    # without range
    J0[0:nm,i]=(obs_dec_dx-obs_dec)/state_dx[i]/angle_std
    J0[nm:(nm+nm) ,i]=(obs_az_dx-obs_az)/state_dx[i]/angle_std
    # with range
    J1[0:nm,i]=(obs_dec_dx-obs_dec)/state_dx[i]/angle_std
    J1[nm:(nm+nm) ,i]=(obs_az_dx-obs_az)/state_dx[i]/angle_std
    J1[(nm+nm):(nm+nm+nm) ,i]=(obs_range_dx-obs_range)/state_dx[i]/range_std

# with angles only
print("standard deviation with angles only")
print("x,y,z,vx,vy,vz (km and km/s)")
print(n.sqrt(n.diag(n.linalg.inv(n.dot(n.transpose(J0),J0)))))
# with angles and range
print("x,y,z,vx,vy,vz (km and km/s)")
print(n.sqrt(n.diag(n.linalg.inv(n.dot(n.transpose(J1),J1)))))



    
    
    
