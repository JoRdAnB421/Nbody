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
        self.plotReady=False

        # Define the gravitation parameter mu
        self.m1 = m1
        self.m2 = m2
        self.mu = self.G*(m1+m2)

        # Entering as orbital elements or state vector
        if input_type=='Orbital_Elements':
            self.ecc, self.semi, self.inc, self.Omega, self.omega, self.theta, self.t0 = args
        
        elif input_type=='State_Vector':
            self.r1, self.v1, self.r2, self.v2 = args

            # Convert velocities to AU/year
            self.v1/=self.AU_yr_2_km_s
            self.v2/=self.AU_yr_2_km_s

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
        rotomg = np.array([[np.cos(-self.omega),  np.sin(-self.omega), 0],
                            [-np.sin(-self.omega), np.cos(-self.omega), 0],
                            [0, 0, 1]
                            ]) # Argument of periapsis rotation
        
        rotInc = np.array([[1, 0, 0],
                           [0, np.cos(-self.inc), np.sin(-self.inc)],
                           [0, -np.sin(-self.inc), np.cos(-self.inc)]
                           ]) # Inclination rotation

        rotOmg = np.array([[np.cos(-self.Omega), np.sin(-self.Omega),  0],
                           [-np.sin(-self.Omega), np.cos(-self.Omega), 0],
                           [0, 0, 1]
                           ]) # Longitude of Ascending Node
        
        # Apply numerical threshold to handle floating point inaccuracies
        rotomg = np.where(np.isclose(rotomg, 0, atol=1e-10), 0, rotomg)
        rotInc = np.where(np.isclose(rotInc, 0, atol=1e-10), 0, rotInc)
        rotOmg = np.where(np.isclose(rotOmg, 0, atol=1e-10), 0, rotOmg)

        # Perform the rotations 
        totRot = rotomg@rotInc@rotOmg
        
        newVec = vec@totRot.T
        
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
        rotOmg = np.array([[np.cos(-self.Omega),  -np.sin(-self.Omega), 0],
                           [np.sin(-self.Omega), np.cos(-self.Omega), 0],
                           [0, 0, 1]
                           ]) # Argument of periapsis rotation
        
        rotInc = np.array([[1, 0, 0],
                           [0, np.cos(self.inc), -np.sin(self.inc)],
                           [0, np.sin(self.inc), np.cos(self.inc)]
                           ]) # Inclination rotation

        rotomg = np.array([[np.cos(-self.omega), -np.sin(-self.omega),  0],
                           [np.sin(-self.omega), np.cos(-self.omega), 0],
                           [0, 0, 1]
                           ]) # Longitude of Ascending Node
        
        # Apply numerical threshold to handle floating point inaccuracies
        rotomg = np.where(np.isclose(rotomg, 0, atol=1e-10), 0, rotomg)
        rotInc = np.where(np.isclose(rotInc, 0, atol=1e-10), 0, rotInc)
        rotOmg = np.where(np.isclose(rotOmg, 0, atol=1e-10), 0, rotOmg)

        # Perform the rotations 
        totRot = rotOmg@rotInc@rotomg
        
        newVec = vec@totRot.T
        
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

    def __repr__(self):
        '''
        Prints the complete state vector if it exists
        '''
        try:
            return (f'Body 1   :   M1 = {self.m1} Msol    :    R1 = {self.r1} AU    :    V1 = {self.v1} km/s\nBody 2   :   M2 = {self.m2} Msol    :    R2 = {self.r2} AU    :    V2 = {self.v2} km/s\n')
        except:
            return 'State vector not yet calculated or input'
    
    def returnState(self):
        '''
        Returns the state vector with form :
        m1, m2, rorb1, vorb1, rorb2, vorb2
        '''
        return [self.m1, self.m2, self.r1, self.v1, self.r2, self.v2]

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

        rorb, vorb = self.r1, self.v1

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
        self.plotReady=True

        return ecc, semi, inc, Omega, omega, theta

    def calc_state_vector(self):
        '''
        Computes the state vector from the orbital elements
        
        Position and Velocity in orbital plane
        ------------------------------------------------
        vorb = mu/h * [-sin(theta)X + (e+cos(theta))Y]
        rorb = h^2/mu * 1/(1+e*cos(theta)) * [cos(theta)X + sin(theta)Y]
        '''
        # Calculate specific angular momentum h from semi and ecc
        hsq = self.mu*self.semi*(1-self.ecc**2)

        # Define rorb and vorb
        rorb = np.array([np.cos(self.theta), np.sin(self.theta),0]) * hsq/self.mu * 1/(1+self.ecc*np.cos(self.theta))
        vorb = np.array([-np.sin(self.theta), (self.ecc + np.cos(self.theta)),0]) * self.mu/hsq**0.5

        # Rotate into reference frame
        rinert = self.orb_plane_2_ref_plane(rorb)
        vinert = self.orb_plane_2_ref_plane(vorb)

        self.m1frac = self.m2/(self.m1+self.m2)
        self.m2frac = self.m1/(self.m1+self.m2)

        # transform into COM frame 
        self.r1, self.v1 = -rinert*self.m1frac, -vinert*self.m1frac
        self.r2, self.v2 = rinert*self.m2frac, vinert*self.m2frac

        # Convert velocities to km/s
        self.v1*=self.AU_yr_2_km_s
        self.v2*=self.AU_yr_2_km_s

        self.plotReady=True

    def plotPositions(self, figNum=1, quiverSize=1):
        '''
        Plot the position of the particles as well as the orbtial ellipse
        '''
        if not self.plotReady: 
            print('Need to calculate State Vector / Orbital Elements')
            return
        
        fig=plt.figure(figNum)
        ax = fig.add_subplot(111, projection='3d')

        # Plotting orbital ellipses in 3D
        phi = np.linspace(0,2*np.pi, 1000)
        
        # Adjust semi major axis for both orbits
        a1 = self.semi * self.m1frac
        a2 = self.semi * self.m2frac

        b1 = np.sqrt(a1**2*(1-self.ecc**2))
        b2 = np.sqrt(a2**2*(1-self.ecc**2))

        c1 = a1*self.ecc
        c2 = a2*self.ecc

        x1 = a1*np.cos(phi)+c1; y1 = b1*np.sin(phi); z1 = np.zeros_like(phi)
        vec1 = np.vstack((x1, y1, z1))
        
        x2 = a2*np.cos(phi)-c2; y2 = b2*np.sin(phi); z2 = np.zeros_like(phi)
        vec2 = np.vstack((x2, y2, z2))
        
        # Rotate ellipse
        newvec1 = np.array([self.orb_plane_2_ref_plane(i) for i in vec1.T]).T
        newvec2 = np.array([self.orb_plane_2_ref_plane(i) for i in vec2.T]).T

        # ax.plot(vec[0,:], vec[1,:], vec[2,:], color='tab:red')
        ax.plot(newvec1[0,:], newvec1[1,:], newvec1[2,:], color='black')
        ax.plot(newvec2[0,:], newvec2[1,:], newvec2[2,:], color='grey')

        ax.plot(self.r1[0],self.r1[1],self.r1[2], 'o')
        ax.plot(self.r2[0],self.r2[1],self.r2[2], 'o')

        # Plot velocities
        ax.quiver(self.r1[0],self.r1[1],self.r1[2], quiverSize*self.v1[0]/np.linalg.norm(self.v1),
                    quiverSize*self.v1[1]/np.linalg.norm(self.v1),quiverSize*self.v1[2]/np.linalg.norm(self.v1), color='tab:green',)
        ax.quiver(self.r2[0],self.r2[1],self.r2[2], quiverSize*self.v2[0]/np.linalg.norm(self.v2),
                    quiverSize*self.v2[1]/np.linalg.norm(self.v2),quiverSize*self.v2[2]/np.linalg.norm(self.v2), color='tab:green',)
        
        return fig, ax
    
    def returnNBodyInput(self, GNBody=886.46):
        '''
        Converts the velocity into NBody Units and then prints a string
        string which is in the form required for the NBody input
        
        GNBody = Gravitational constant conversion into km^2/s^2 AU/Msol
        '''
        
        v_convert = np.sqrt(GNBody)
        print(f'{self.m1} {self.r1[0]} {self.r1[1]} {self.r1[2]} {self.v1[0]/v_convert} {self.v1[1]/v_convert} {self.v1[2]/v_convert}')
        print(f'{self.m2} {self.r2[0]} {self.r2[1]} {self.r2[2]} {self.v2[0]/v_convert} {self.v2[1]/v_convert} {self.v2[2]/v_convert}')

class Triple():
    '''
    initialises a triple system

    If givinig orbital parameters give in args in following form 

    inner binary = Ecc, semi, incl, Omega, omega, theta, t0
    tertirary-InnerCOM Binary = Ecc, semi, incl, Omega, omega, theta, t0

    args [List or array] = inner binary + tertirary-InnerCOM Binary 
    '''
    plotReadyTrip=False

    def __init__(self, m1,m2,m3, *args, input_type='Orbital_Elements'):
        self.plotReady=False

        # Define the 
        self.m1 = m1
        self.m2 = m2
        self.mtot = self.m1+self.m2
        self.m3 = m3

        # Entering as orbital elements or state vector
        if input_type=='Orbital_Elements':
            self.binary_inner = Binary(m1, m2, *args[:7])
            self.binary_out = Binary(self.mtot, m3, *args[7:])

            ### Euler angles for the outer orbit
            self.Omega = args[10]
            self.omega = args[11]
            self.inc = args[9]


        elif input_type=='State_Vector':
            print('not working yet SORRY!!')
            raise
            self.r1, self.v1, self.r2, self.v2 , self.r3, self.v3 = args
            self.binary_inner = Binary(m1, m2, *args[:4], input_type='State_Vector')

            outerargs = []
            self.binary_outer = Binary(self.mtot, m3, args[:4], input_type='State_Vector')

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
        rotomg = np.array([[np.cos(-self.omega),  np.sin(-self.omega), 0],
                            [-np.sin(-self.omega), np.cos(-self.omega), 0],
                            [0, 0, 1]
                            ]) # Argument of periapsis rotation
        
        rotInc = np.array([[1, 0, 0],
                           [0, np.cos(self.inc), np.sin(self.inc)],
                           [0, -np.sin(self.inc), np.cos(self.inc)]
                           ]) # Inclination rotation

        rotOmg = np.array([[np.cos(-self.Omega), np.sin(-self.Omega),  0],
                           [-np.sin(-self.Omega), np.cos(-self.Omega), 0],
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
        rotOmg = np.array([[np.cos(-self.Omega),  -np.sin(-self.Omega), 0],
                           [np.sin(-self.Omega), np.cos(-self.Omega), 0],
                           [0, 0, 1]
                           ]) # Argument of periapsis rotation
        
        rotInc = np.array([[1, 0, 0],
                           [0, np.cos(self.inc), -np.sin(self.inc)],
                           [0, np.sin(self.inc), np.cos(self.inc)]
                           ]) # Inclination rotation

        rotomg = np.array([[np.cos(-self.omega), -np.sin(-self.omega),  0],
                           [np.sin(-self.omega), np.cos(-self.omega), 0],
                           [0, 0, 1]
                           ]) # Longitude of Ascending Node
        
        # Apply numerical threshold to handle floating point inaccuracies
        rotomg = np.where(np.isclose(rotomg, 0, atol=1e-10), 0, rotomg)
        rotInc = np.where(np.isclose(rotInc, 0, atol=1e-10), 0, rotInc)
        rotOmg = np.where(np.isclose(rotOmg, 0, atol=1e-10), 0, rotOmg)

        # Perform the rotations 
        totRot = rotOmg@rotInc@rotomg
        
        newVec = vec@totRot.T
        
        
        return newVec
    
    def calc_state_vector(self):
        '''
        Computes the state vector for the triple system
        '''

        # First create the inner binary to get r1, r2, v1, v2
        self.binary_inner.calc_state_vector()
        _,_,self.r1,self.v1,self.r2,self.v2 = self.binary_inner.returnState()

        # Then create the outer binary to get r3, v3
        self.binary_out.calc_state_vector()
        _,_,self.rinCOM,self.vinCOM,self.r3,self.v3 = self.binary_out.returnState()

        self.plotReadyTrip=True

    def plotTriple(self):
        '''
        Plots the initial positions of particles and orbital ellipse
        '''
        if not self.plotReadyTrip: 
            print('Generate state vectors first')
            return
        
        fig, ax = self.binary_inner.plotPositions(1, quiverSize=5)
        
        # Plotting orbital ellipse in 3D
        phi = np.linspace(0,2*np.pi, 100)
        a = args[8] * self.mtot/(self.m3+self.mtot)
        b = np.sqrt(a**2*(1-args[7]**2))
        c = a*args[7]
        x = a*np.cos(phi)-c; y = b*np.sin(phi); z = np.zeros_like(phi)
        vec = np.vstack((x, y, z))

        # Rotate ellipse
        newvec = np.array([self.orb_plane_2_ref_plane(i) for i in vec.T]).T


        ax.plot(newvec[0,:], newvec[1,:], newvec[2,:], color='tab:purple')

        # ax.plot(r1[0],r1[1],r1[2], 'o')
        ax.plot(self.r3[0],self.r3[1],self.r3[2], 'o')
        quiverSize=5
        ax.quiver(self.r3[0],self.r3[1],self.r3[2], quiverSize*self.v3[0]/np.linalg.norm(self.v3),quiverSize* 
                  self.v3[1]/np.linalg.norm(self.v3),quiverSize*self.v3[2]/np.linalg.norm(self.v3), color='tab:green',)
    
    def returnNBodyInput(self, GNBody=886.46):
        '''
        prints state vector ready for NBody code
        '''
        self.binary_inner.returnNBodyInput()

        v_convert = np.sqrt(GNBody)
        print(f'{self.m3} {self.r3[0]} {self.r3[1]} {self.r3[2]} {self.v3[0]/v_convert} {self.v3[1]/v_convert} {self.v3[2]/v_convert}')


if __name__ =="__main__":
    # Ecc, semi, incl, Omega, omega, theta, t0
    argsOrb = [0.5, 5, np.deg2rad(0), 0, np.deg2rad(120), 1., 0]
    m1=1
    m2=0.5
    binary = Binary(m1, m2, *argsOrb)

    binary.calc_state_vector()
    print(binary)
    
    binary.plotPositions()
    plt.show()

    ### Print values for binary NBody
    # binary.returnNBodyInput()


    ############## Create a triple
    # Ecc, semi, incl, Omega, omega, theta, t0
    argsinner = [0.8, 5, np.deg2rad(24), np.pi/3, np.deg2rad(120), 4.3, 0]
    argsouter = [0.1, 10, np.deg2rad(70), 3/2 * np.pi, 4.7 , 2, 0]
    args = np.append(argsinner, argsouter)
    m1=1
    m2=1
    m3=1

    triple = Triple(m1,m2,m3,*args)
    triple.calc_state_vector()

    triple.plotTriple()

    triple.returnNBodyInput()
    plt.show()
    