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


epoch_et0 = spice.str2et("2025-05-05T00:00:00")

def observe_asteroid(kep=n.array([2*sc.au/1e3,0.3,10,80,45,30]),
                     epoch_et=epoch_et0+3600):    
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
    mu_sun = 1.32712440018e11  # Gravitational parameter of Sun in km^3/s^2
    # Epoch of these elements
    

 #   print(epoch_et)
    kep_elements = [
        kep[0]*(1-kep[1]),     # PERIFOCAL DISTANCE = (a*(1-e)))#Semi-major axis in km (e.g., 2 AU ≈ 300 million km)
        kep[1],           # Eccentricity
        np.radians(kep[2]),     # Inclination (rad)
        np.radians(kep[3]),     # RAAN (rad)
        np.radians(kep[4]),     # Argument of periapsis (rad)
        np.radians(kep[5]),      # Mean anomaly (rad)
        epoch_et0,   # epoch of elements
        mu_sun
    ]


    # IAU76/J2000 heliocentric ecliptic osculating elements
    #elements = [a_km, ecc, inc, raan, argp, m0, epoch_et, mu_sun]
    
    # epoch of observations (J2000 seconds)
    
    state_asteroid = spice.conics(kep_elements, epoch_et)
    return(obs_helio_eclip,state_asteroid)

#    print("Asteroid state in J2000 (km, km/s):", state_asteroid)

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)  # inclination
    phi = np.arctan2(y, x)    # azimuth
    return(r, theta, phi)

def simulate_measurement(kep=n.array([2*sc.au/1e3,0.3,20,80,45,30]),
                         times=epoch_et0+n.arange(100)*7*24*3600):
#    op=[]
    ap=[]
    decs=[]
    azs=[]
    rs=[]
    for i in range(len(times)):
        obs_pos,ast_pos=observe_asteroid(kep,epoch_et=times[i])
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
                       k_guess=n.array([2.1*sc.au/1e3,0.25,23,82,48,27]),
                       use_range=True,
                       angle_std=5e-6,
                       range_std=2,                       
                       times=epoch_et0+n.arange(100)*7*24*3600):
    kepin=n.zeros(6)
    def ss(x):
 #       print(x)
        kepin[:]=x
        kepin[1]=10**x[1]
        model_range,model_dec,model_az,model_ap=simulate_measurement(kep=kepin,times=times)
        s=0.0
        s+=n.sum((n.angle(n.exp(1j*model_dec)*n.exp(-1j*obs_dec))**2))/angle_std
        s+=n.sum((n.angle(n.exp(1j*model_az)*n.exp(-1j*obs_az))**2))/angle_std
        if use_range:
            s+=n.sum((obs_range-model_range)**2.0)/range_std
#        print(s)
        return(s)
    import scipy.optimize as sio
    k_guess0=n.copy(k_guess)
    k_guess0[1]=n.log10(k_guess0[1])
    xhat=sio.minimize(ss,k_guess0,method="Nelder-Mead",bounds=[(1e6,1e9),(-15,1),(0,180),(0,360),(0,360),(0,360)]).x
    xhat=sio.minimize(ss,xhat,method="Nelder-Mead",bounds=[(1e6,1e9),(-15,1),(0,180),(0,360),(0,360),(0,360)]).x    
    #xhat=sio.fmin(ss,xhat)
    print(ss(xhat))
    return(xhat)

# (2025 LB)
times=epoch_et0+n.arange(30)*24*3600
# 
kep_true=n.array([2.158504572325312*sc.au/1e3,
                  0.5573199074979915, # e
                  7.977400997581499,  # inc
                  73.8364986266037,   # node 
                  212.6277929093889,  # aop
                  342.498256445095])  # mean anom at epoch
kep_guess=kep_true+n.array([0.01*sc.au/1e3,0.01,0.1,0.1,0.1,0.1])
#kep_guess=n.array([2.01*sc.au/1e3,0.3005,20.2,80.1,45.1,30.1])

obs_range,obs_dec,obs_az,ap=simulate_measurement(kep_true,times=times)
plt.plot(times,obs_range*1e3/sc.au,".")
plt.xlabel("Time (J2000 sec)")
plt.ylabel("Range (AU)")
plt.show()

angle_std=5e-6
range_std=1                  
range_err=n.random.randn(len(obs_range))*range_std
dec_err=n.random.randn(len(obs_range))*angle_std
az_err=n.random.randn(len(obs_range))*angle_std

xhat=estimate_keplerian(obs_range+range_err,obs_dec+dec_err,obs_az+az_err,k_guess=kep_guess,times=times,use_range=False)
xhat[1]=10**xhat[1]
print("Estimated Keplerian parameters using angles only (RP,e,i,Omega,omega,nu)")
print(xhat)
#print("Error (percent)")
#print( 100*(kep_true-xhat)/kep_true)

xhat=estimate_keplerian(obs_range+range_err,obs_dec+dec_err,obs_az+az_err,k_guess=kep_guess,times=times,use_range=True)
xhat[1]=10**xhat[1]

print("Estimated Keplerian parameters using angles and range (RP,e,i,Omega,omega,nu)")
print(xhat)
#print("Error (percent)")
#print( 100*(kep_true-xhat)/kep_true)


#print(xhat)
#print(10**xhat[1])
print("True Keplerian parameters (RP,e,i,Omega,omega,nu)")
print(kep_true)

# estimate jacobian
kep_dx=n.array([1e8,1,1,1,1,1])*0.0001
nm=len(obs_range)
# without range
J0=n.zeros([2*nm,6])
# with range
J1=n.zeros([3*nm,6])

for i in range(6):
    kep_pert=n.copy(kep_true)
    kep_pert[i]=kep_true[i]+kep_dx[i]
    obs_range_dx,obs_dec_dx,obs_az_dx,ap=simulate_measurement(kep_pert,times=times)
    # without range
    J0[0:nm,i]=(obs_dec_dx-obs_dec)/kep_dx[i]/angle_std
    J0[nm:(nm+nm) ,i]=(obs_az_dx-obs_az)/kep_dx[i]/angle_std
    # with range
    J1[0:nm,i]=(obs_dec_dx-obs_dec)/kep_dx[i]/angle_std
    J1[nm:(nm+nm) ,i]=(obs_az_dx-obs_az)/kep_dx[i]/angle_std
    J1[(nm+nm):(nm+nm+nm) ,i]=(obs_range_dx-obs_range)/kep_dx[i]/range_std

# with angles only
print("standard deviation with angles only")
print("(RP km,e,i deg,omega deg,Omega deg,nu deg)")
print(n.sqrt(n.diag(n.linalg.inv(n.dot(n.transpose(J0),J0)))))
# with angles and range
print("(RP km,e,i deg,omega deg,Omega deg,nu deg)")
print(n.sqrt(n.diag(n.linalg.inv(n.dot(n.transpose(J1),J1)))))



    
    
    
