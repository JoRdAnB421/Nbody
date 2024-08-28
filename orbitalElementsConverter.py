import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

class Binary:
    '''
    Initialises a binary based on some initial conditions
    '''
    G = 39.42 # AU^3 / Msol / Year^2
    AU_yr_2_km_s = 4.744 # Convert from AU/year to km/s

    def __init__(self, m1, m2, *args, input_type='Orbital_Elements'):
        # Define the gravitation parameter mu
        self.m1 = m1
        self.m2 = m2
        self.mu = self.G*(m1+m2)

        # Entering as orbital elements or state vector
        if input_type=='Orbital_Elements':
            self.ecc, self.semi, self.inc, self.Omega, self.omega, self.theta, self.t0 = args
        
        elif input_type=='State_Vector':
            self.rorb1, self.vorb1, self.rorb2, self.vorb2 = args

            # Convert velocities to AU/year
            self.vorb1/=self.AU_yr_2_km_s
            self.vorb2/=self.AU_yr_2_km_s

        else:
            raise ValueError("Unknown input type")

        return

    def ref_plane_2_orb_plane(self,vec):
        '''
        Rotate from the equatorital/reference plane to the orbital plane

        Inertial coordinate frame (Equatorial plane defined by I,J)
        ----------------------------
        I = direction of vernal equinox
        J = perp to I
        K = perp to reference plane

        Orbtial Frame (Orbital plane defined by x, y) 
        ----------------------------
        x = direction to pericenter
        y = perp to x and z
        z = perp to orbital plane

        Equations
        ----------------------------
        X = x1*I + x2*J + x3*K
        Y = y1*I + y2*J + y3*K
        Z = z1*I + z2*J + z3*K


        Performs Rotation

        |x1 x2 x3|      |cos(omega)  sin(omega) 0||1     0         0   ||cos(Omega)  sin(Omega) 0|
        |y1 y2 y3|   =  |-sin(omega) cos(omega) 0||0 cos(inc)  sin(inc)||-sin(Omega) cos(Omega) 0|
        |z1 z2 z3|      |   0          0        1||0 -sin(inc) cos(inc)||   0          0        1|
        '''

        # Define Rotation matrices
        rotomg = np.array([[np.cos(self.omega),  np.sin(self.omega), 0],
                            [-np.sin(self.omega), np.cos(self.omega), 0],
                            [0, 0, 1]
                            ]) # Argument of periapsis rotation
        
        rotInc = np.array([[1, 0, 0],
                           [0, np.cos(self.inc), np.sin(self.inc)],
                           [0, -np.sin(self.inc), np.cos(self.inc)]
                           ]) # Inclination rotation

        rotOmg = np.array([[np.cos(self.Omega), np.sin(self.Omega),  0],
                           [-np.sin(self.Omega), np.cos(self.Omega), 0],
                           [0, 0, 1]
                           ]) # Longitude of Ascending Node
        
        # Apply numerical threshold to handle floating point inaccuracies
        rotomg = np.where(np.isclose(rotomg, 0, atol=1e-10), 0, rotomg)
        rotInc = np.where(np.isclose(rotInc, 0, atol=1e-10), 0, rotInc)
        rotOmg = np.where(np.isclose(rotOmg, 0, atol=1e-10), 0, rotOmg)

        # Perform the rotations 
        totRot = np.matmul(rotOmg, np.matmul(rotInc, rotomg))
        
        # Ensure shape of vector for cross produt
        if vec.shape!=(3,1): vec=vec.reshape(3,1)

        # Rotate vector
        newVec = np.cross(totRot, vec)
        
        return newVec
    
    def orb_plane_2_ref_plane(self,vec):
        '''
        Rotate from the orbital plane plane to the equatorital/reference

        Orbtial Frame (Orbital plane defined by x, y) 
        ----------------------------
        x = direction to pericenter
        y = perp to x and z
        z = perp to orbital plane

        
        Inertial coordinate frame (Equatorial plane defined by I,J)
        ----------------------------
        I = direction of vernal equinox
        J = perp to I
        K = perp to reference plane

        Equations
        ----------------------------
        I = i1*X + i2*Y + i3*Z
        J = j1*X + j2*Y + j3*Z
        K = k1*X + k2*Y + k3*Z

        Performs Rotation

        |i1 i2 i3|      |cos(Omega) -sin(Omega) 0||1     0         0   ||cos(omega) -sin(omega) 0|
        |j1 j2 j3|   =  |sin(Omega)  cos(Omega) 0||0 cos(inc) -sin(inc)||sin(omega)  cos(omega) 0|
        |k1 k2 k3|      |   0          0        1||0 sin(inc)  cos(inc)||   0          0        1|
        '''

        # Define Rotation matrices
        rotOmg = np.array([[np.cos(self.Omega),  -np.sin(self.Omega), 0],
                           [np.sin(self.Omega), np.cos(self.Omega), 0],
                           [0, 0, 1]
                           ]) # Argument of periapsis rotation
        
        rotInc = np.array([[1, 0, 0],
                           [0, np.cos(self.inc), -np.sin(self.inc)],
                           [0, np.sin(self.inc), np.cos(self.inc)]
                           ]) # Inclination rotation

        rotomg = np.array([[np.cos(self.omega), -np.sin(self.omega),  0],
                           [np.sin(self.omega), np.cos(self.omega), 0],
                           [0, 0, 1]
                           ]) # Longitude of Ascending Node
        
        # Apply numerical threshold to handle floating point inaccuracies
        rotomg = np.where(np.isclose(rotomg, 0, atol=1e-10), 0, rotomg)
        rotInc = np.where(np.isclose(rotInc, 0, atol=1e-10), 0, rotInc)
        rotOmg = np.where(np.isclose(rotOmg, 0, atol=1e-10), 0, rotOmg)

        # Perform the rotations 
        totRot = np.matmul(rotomg, np.matmul(rotInc, rotOmg))

        # Rotate vector
        newVec = np.dot(totRot, vec)
        
        return newVec
    
    def transform_2_COM_frame(self, rad, vel):
        '''
        Transforms from the co-moving frame of one of the bodies to the COM frame 
        
        Co-moving frame (centered on M1)
        -----------------------------------------------
        vorb = mu/h * [-sin(theta)X + (e+cos(theta))Y]
        rorb = h^2/mu * 1/(1+e*cos(theta)) * [cos(theta)X + sin(theta)Y]
        
        rcom = rorb * m2/(m1+m2)
        vcom = vorb * m2/(m1+m2)

        COM Frame
        ------------------------------------------------
        rorb1 = -m2/(m1+m2) * rorb
        rorb2 = m1/(m1+m2) * rorb

        vorb1 = -m2/(m1+m2) * vorb
        vorb2 = m1/(m1+m2) * vorb
        '''
        # Define the mass factors
        factM2 = self.m2/(self.m1+self.m2)
        factM1 = self.m1/(self.m1+self.m2)
        
        # transform the position and velocities
        rorb1 = -factM2 * rad
        rorb2 = factM1 * rad
        vorb1 = -factM2 * vel
        vorb2 = factM1 * vel

        return rorb1, rorb2, vorb1, vorb2

    def transform_2_comoving_frame(self, rad, vel):
        '''
        Transforms from the COM frame to the co-moving frame of one of the bodies

        COM Frame
        ------------------------------------------------
        rorb1 = -m2/(m1+m2) * rorb
        rorb2 = m1/(m1+m2) * rorb

        vorb1 = -m2/(m1+m2) * vorb
        vorb2 = m1/(m1+m2) * vorb
        
        Co-moving frame (centered on M1)
        -----------------------------------------------
        rorb = (m1+m2)/m1 * rorb2
        vorb = (m1+m2)/m1 * vorb2
        
        '''
        # Define the mass factor
        fact = (m1+m2)/m1

        # Shift into comoving frame centered on m1
        rorb = fact * rad
        vorb = fact * vel

        return rorb, vorb

    def calc_state_vector(self):
        '''
        Computes the state vector from the orbital elements
        
        Position and Velocity in orbital plane
        ------------------------------------------------
        vorb = mu/h * [-sin(theta)X + (e+cos(theta))Y]
        rorb = h^2/mu * 1/(1+e*cos(theta)) * [cos(theta)X + sin(theta)Y]
        '''
        # Calculate specific angular momentum h from semi and ecc
        hsq = self.semi * self.mu * (1+self.ecc)

        # Define rorb and vorb
        rorb = np.array([np.cos(self.theta), np.sin(self.theta),0]) * hsq/self.mu * 1/(1+self.ecc*np.cos(self.theta))
        vorb = np.array([-np.sin(self.theta), (self.ecc + np.cos(self.theta)),0]) * self.mu/hsq**0.5

        # Rotate into reference frame
        rinert = self.orb_plane_2_ref_plane(rorb)
        vinert = self.orb_plane_2_ref_plane(vorb)
    
        # transform into COM frame 
        # self.rorb1, self.rorb2, self.vorb1, self.vorb2 = self.transform_2_COM_frame(rinert, vinert)
        self.rorb1, self.vorb1 = rinert, vinert
        self.rorb2, self.vorb2 = -self.rorb1, -self.vorb1
        # Convert velocities to km/s
        self.vorb1*=self.AU_yr_2_km_s
        self.vorb2*=self.AU_yr_2_km_s

    def printState(self):
        '''
        Prints the complete state vector if it exists
        '''
        try:
            print(f'Body 1   :   M1 = {self.m1} Msol    :    R1 = {self.rorb1} AU    :    V1 = {self.vorb1} km/s\nBody 2   :   M2 = {self.m2} Msol    :    R2 = {self.rorb2} AU    :    V2 = {self.vorb2} km/s\n')
        except:
            return 'State vector not yet calculated or input'
    
    def returnState(self):
        '''
        Returns the state vector with form :
        m1, m2, rorb1, vorb1, rorb2, vorb2
        '''
        return [self.m1, self.m2, self.rorb1, self.vorb1, self.rorb2, self.vorb2]

    def printOrbElems(self):
        '''
        Prints the orbital elements
        '''
        try:
            print (f'Ecc = {self.ecc}   Semi = {self.semi}   incl = {self.inc}    Omega = {self.Omega}    omega = {self.omega}    theta = {self.theta}\n')
        except:
            return ' Orbital elements not yet calculated or input'
        
    def calc_orbital_elems(self):
        '''
        Computes the orbital elements from the state vector
        
        State Vector
        ----------------------
        (x1, y1, z1, x2, y2, z2, vx1, vy1, vz1, vx2, vy2, vz2)
        
        Orbital elements
        -----------------------
        a = semi-major axis
        e = eccentricity
        i = inclination
        Omega = Longitude of asecending node
        omega = argumentof periapsis
        theta = true anomaly

        Returns all elements
        '''

        # First convert into comoving frame of M1
        # rorb, vorb = self.transform_2_comoving_frame(self.rorb2, self.vorb2)

        rorb, vorb = self.rorb1, self.vorb1

        # For use later
        r_norm = np.linalg.norm(rorb)
        v_norm = np.linalg.norm(vorb)

        ######################################################
        # Now calculate angular momentum and its magnitude
        h_vec = np.cross(rorb, vorb)
        h_mag = np.linalg.norm(h_vec)

        ######################################################
        # Now we can find the inclination
        inc = np.arccos(h_vec[2]/h_mag)

        #####################################################
        # For the longitude of ascending node we need a vector norm to the reference plane and orbital plane
        K = np.array([0,0,1]) # Make a vector normal to reference plane

        N_vec = np.cross(K, h_vec)
        if all(N_vec==0): N_vec=np.array([1,0,0])
        N = np.linalg.norm(N_vec)
        if N!=0 : Omega = np.arccos(N_vec[0]/N)
        else: Omega=0

        # Now check which quadrant it lies in (since it varies from 0 --> 360)
        if N_vec[1]<0: 
            Omega = 2*np.pi - Omega

        ########################################################
        # Now Eccentricity from a standard formula
        e_vec = (np.cross(vorb, h_vec))/self.mu - rorb/r_norm

        ecc = np.linalg.norm(e_vec)

        #########################################################
        # Now Argument of periapsis found from definition of dot product
        print(np.dot(e_vec,N_vec)/(ecc*N))
        omega = np.arccos(np.dot(e_vec,N_vec)/(ecc*N))

        # Again it varies from 0 --> 360 so we check which quad it is in
        if e_vec[2]<0:
            omega = 2*np.pi - omega

        ###########################################################
        # True anomaly 
        theta = np.arccos(np.dot(e_vec, rorb)/(ecc*r_norm))

        ############################################################
        # Finally semi-major axis
        semi = h_mag**2/self.mu * 1/(1+ecc)

        # Save to class
        self.ecc=ecc
        self.semi=semi
        self.inc=inc
        self.Omega=Omega
        self.omega=omega
        self.theta=theta

        return ecc, semi, inc, Omega, omega, theta


def Rot_orb_2_ref(vec, Omega, inc, omega):
    # Define Rotation matrices
    rotOmg = np.array([[np.cos(Omega),  -np.sin(Omega), 0],
                    [np.sin(Omega), np.cos(Omega), 0],
                    [0, 0, 1]
                    ]) # Argument of periapsis rotation
            
    rotInc = np.array([[1, 0, 0],
                    [0, np.cos(inc), -np.sin(inc)],
                    [0, np.sin(inc), np.cos(inc)]
                    ]) # Inclination rotation
    rotomg = np.array([[np.cos(omega), -np.sin(omega),  0],
                    [np.sin(omega), np.cos(omega), 0],
                    [0, 0, 1]
                    ]) # Longitude of Ascending Node
            
    # Apply numerical threshold to handle floating point inaccuracies
    rotomg = np.where(np.isclose(rotomg, 0, atol=1e-10), 0, rotomg)
    rotInc = np.where(np.isclose(rotInc, 0, atol=1e-10), 0, rotInc)
    rotOmg = np.where(np.isclose(rotOmg, 0, atol=1e-10), 0, rotOmg)
    # Perform the rotations 
    totRot = np.matmul(rotomg, np.matmul(rotInc, rotOmg))
    # Rotate vector
    newVec = np.dot(totRot, vec)

    return newVec

if __name__ =="__main__":
    # Ecc, semi, incl, Omega, omega, theta, t0
    argsOrb = [0.5, 5, np.deg2rad(0), 0, np.deg2rad(120), 0., 0]
    m1=0.3
    m2=1.4
    binary = Binary(m1, m2, *argsOrb)

    binary.calc_state_vector()
    binary.printState()

    ##### Checking
    statevect = binary.returnState()

    # Defines inner binary
    mtot = statevect[0]+statevect[1]
    m1 = statevect[0]
    m2 = statevect[1]
    r1 = statevect[2]
    v1 = statevect[3]
    r2 = statevect[4]
    v2 = statevect[5]
    
    ##### Test
    args = [r1, v1, r2, v2]
    print(args)
    bin_test = Binary(m1,m2,*args, input_type='State_Vector')
    bin_test.calc_orbital_elems()

    bin_test.printOrbElems()

    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotting orbital ellipse in 3D
    phi = np.linspace(0,2*np.pi, 100)
    a = argsOrb[1]
    b = np.sqrt(a**2*(1-argsOrb[0]**2))

    x = a*np.cos(phi); y = b*np.sin(phi); z = np.zeros_like(phi)
    vec = np.vstack((x, y, z))

    # Rotate ellipse
    newvec = Rot_orb_2_ref(vec, argsOrb[3], argsOrb[2], argsOrb[4])

    # ax.plot(vec[0,:], vec[1,:], vec[2,:], color='tab:red')
    ax.plot(newvec[0,:], newvec[1,:], newvec[2,:], color='black')

    ax.plot(r1[0],r1[1],r1[2], 'o')
    ax.plot(r2[0],r2[1],r2[2], 'o')
    ax.quiver(r1[0],r1[1],r1[2], 2*v1[0]/np.linalg.norm(v1),2* v1[1]/np.linalg.norm(v1),2*v1[2]/np.linalg.norm(v1), color='tab:green',)
    ax.quiver(r2[0],r2[1],r2[2], 2*v2[0]/np.linalg.norm(v2),2* v2[1]/np.linalg.norm(v2),2*v2[2]/np.linalg.norm(v2), color='tab:green',)



    ### Print values for binary
    G = 886.46 # AU /Msol km^2/s^2
    v_convert = lambda M, R : np.sqrt(G*M/(2*R))
    v_convert = np.sqrt(G/(1))


    print(m1, *r1, *v1/v_convert)
    print(m2, *r2, *v2/v_convert)
    print()

    #### Add triple
    # Ecc, semi, incl, Omega, omega, theta, t0
    tripargs = [0, 10, np.deg2rad(70), 0, 0 , 0, 0]
    m3 = 0.01
    binaryOut = Binary(mtot, m3, *tripargs)

    binaryOut.calc_state_vector()
    binaryOut.printState()

    # Get state
    outstate = binaryOut.returnState()
    mtot=outstate[0]
    m3 = outstate[1]
    rinCOM = outstate[2]
    vinCOM = outstate[3]
    r3 = outstate[4]
    v3 = outstate[5]

    ### Correct inner binary motion to new COM frame
    # r1 += rinCOM
    # r2 += rinCOM
    # v1 += vinCOM
    # v2 += vinCOM


    #### Algorthimic Regularisation input
    G = 886.46 # AU /Msol km^2/s^2
    v_convert = np.sqrt(G/(1))

    print('Input for Algorithmic Regularisation\n')
    print(m3, *r3, *v3/v_convert)
    print(m1, *r1, *v1/v_convert)
    print(m2, *r2, *v2/v_convert)


    # Plotting orbital ellipse in 3D
    phi = np.linspace(0,2*np.pi, 100)
    a = tripargs[1]
    b = np.sqrt(a**2*(1-tripargs[0]**2))

    x = a*np.cos(phi); y = b*np.sin(phi); z = np.zeros_like(phi)
    vec = np.vstack((x, y, z))

    # Rotate ellipse
    newvec = Rot_orb_2_ref(vec, tripargs[3], tripargs[2], tripargs[4])

    ax.plot(newvec[0,:], newvec[1,:], newvec[2,:], color='tab:purple')

    # ax.plot(r1[0],r1[1],r1[2], 'o')
    ax.plot(r3[0],r3[1],r3[2], 'o')
    ax.quiver(r3[0],r3[1],r3[2], 10*v3[0]/np.linalg.norm(v3),10* v3[1]/np.linalg.norm(v3),10*v3[2]/np.linalg.norm(v3), color='tab:green',)

    plt.show()

