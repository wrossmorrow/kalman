
import pandas as pd
import numpy as np

from abc import abstractmethod
from haversine import haversine
from scipy.linalg import ldl, solve_triangular

class KalmanFilter:

    def __init__(self, *args, **kwargs): 

        # column names to use
        self._time = kwargs['time_col'] if 'time_col' in kwargs else 'time'
        self._lat = kwargs['lat_col'] if 'lat_col' in kwargs else 'lat'
        self._lon = kwargs['long_col'] if 'long_col' in kwargs else 'lon'
        self._err = kwargs['err_col'] if 'err_col' in kwargs else 'err'
        self._hav = kwargs['hav_col'] if 'hav_col' in kwargs else 'hav'

    @abstractmethod
    def N(self): pass

    @abstractmethod
    def M(self): pass

    @abstractmethod
    def W(self): pass

    @abstractmethod
    def Q(self): pass

    @abstractmethod
    def R(self): pass

    def f(self, window, x, F): 
        """ override for filters with nonlinear models """
        return F.dot(x)

    def h(self, window, x, H): 
        """ override for filters with nonlinear measurements """
        return H.dot(x)

    @abstractmethod
    def F(self, window, x): pass

    @abstractmethod
    def H(self, window, x): pass

    @abstractmethod
    def z(self, window): pass

    @abstractmethod
    def x0(self, window): pass

    @abstractmethod
    def lat(self, x): pass

    @abstractmethod
    def lon(self, x): pass

    @abstractmethod
    def err(self, x, z): pass

    @abstractmethod
    def hav(self, x, z): pass

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def run(self, data: pd.DataFrame, useinv=True):

        N, M, W = self.N(), self.M(), self.W()
        Q, R = self.Q(), self.R()
        T, P, U = np.eye(N), np.eye(N), np.zeros((M,N+1))
        y, x = np.zeros((N,1)), self.x0(data.iloc[:W])

        kf = pd.DataFrame(index=data.index, columns=["time","lat","lon","err","hav"])
        for i in range(W):
            kf.time[i] = data[self._time].iloc[i]
            kf.lat.iloc[i] = data[self._lat].iloc[i]
            kf.lon.iloc[i] = data[self._lon].iloc[i]
            kf.err.iloc[i] = 0.0
            kf.hav.iloc[i] = 0.0

        # NOTE: it would be nice to be able to window over the GPS data
        # frame but pandas doesn't do multi-column rolling windows. We'll
        # get only one column at a time. Hence, explicit iteration over 
        # the windowed rows is required. 

        i = W
        while i < data.shape[0]:

            # window of data this prediction depends on
            window = data.iloc[i-W:i+1]
            
            # partial derivatives of f; just a matrix for linear models
            # and does not actually depend on x
            F = self.F(window, x)

            # y <- f(x|F) (just Fx for linear models)
            y = self.f(window, x, F)

            # z <- measurement
            z = self.z(window)

            # partial derivatives of h; just a matrix for linear measurements
            # and does not actually depend on y
            H = self.H(window, y)

            # r <- z - h(y) (just z - Hy for linear measurements)
            r = z - self.h(window, y, H)

            # T <- F P F' + Q
            T = F.dot(P.dot(F.T)) + Q
            
            # S <- H T H' + R
            S = H.dot(T.dot(H.T)) + R
            
            # iS <- inv(S) (yes, inv is fastest. Go figure.)
            iS = np.linalg.inv(S)

            # K <- T H' inv(S)
            K = T.dot( H.T.dot(iS) )

            # x <- y + K(z - Hy)
            x = y + K.dot(r)

            # P <- ( I - K H ) T = T - K H T
            P = T - K.dot(H.dot(T))
            
            # store results
            kf.time[i] = data.time.iloc[i]
            kf.lat.iloc[i] = self.lat(x)
            kf.lon.iloc[i] = self.lon(x)
            kf.err.iloc[i] = self.err(x, z)
            kf.hav.iloc[i] = self.hav(x, z)
            
            # increment
            i += 1

        # remap column names in case these were customized
        cmap = {c: getattr(self, f"_{c}") for c in kf.columns}
        kf.rename(mapper=cmap, axis='columns', inplace=True)
        return kf


class KalmanFilter_CV_LL(KalmanFilter):

    """
    Kalman Filter implementation using lat/lon only. 
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._F = np.eye(4)
        self._H = np.eye(2,4)
        self._Q = kwargs['Q'] if 'Q' in kwargs else np.eye(4)
        self._R = kwargs['R'] if 'R' in kwargs else np.eye(2)

    def N(self): return 4
    def M(self): return 2
    def W(self): return 2
    def Q(self): return self._Q
    def R(self): return self._R
    def H(self, window, x): return self._H
    def lat(self, x): return x[0,0]
    def lon(self, x): return x[1,0]
    def err(self, x, z): return np.linalg.norm( x[:2,0] - z[:,0] )
    def hav(self, x, z): return haversine(x[0,0], z[0,0], x[1,0], z[1,0])

    def F(self, window, x): 
        dt = window[self._time].iloc[1] - window[self._time].iloc[0]
        self._F[0,2] = self._F[1,3] = dt
        return self._F

    def z(self, window):
        return np.array([
            window[self._lat].iloc[1],
            window[self._lon].iloc[1]
        ]).reshape((2,1))

    def x0(self, window):
        dt = window[self._time].iloc[1] - window[self._time].iloc[0]
        return np.array([
            window[self._lat].iloc[1], 
            window[self._lon].iloc[1],
            (window[self._lat].iloc[1]-window[self._lat].iloc[0])/dt, 
            (window[self._lon].iloc[1]-window[self._lon].iloc[0])/dt,
        ]).reshape((4,1))


class KalmanFilter_CV_LLD(KalmanFilter):

    """
    Kalman Filter implementation using finite differences 
    in GPS coords (pretending these are euclidean)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._F = np.eye(4)
        self._H = np.eye(4)
        self._Q = kwargs['Q'] if 'Q' in kwargs else np.eye(4)
        self._R = kwargs['R'] if 'R' in kwargs else np.eye(4)

    def N(self): return 4
    def M(self): return 4
    def W(self): return 2
    def Q(self): return self._Q
    def R(self): return self._R
    def H(self, window, x): return self._H
    def lat(self, x): return x[0,0]
    def lon(self, x): return x[1,0]
    def err(self, x, z): return np.linalg.norm( x[:2,0] - z[:2,0] )
    def hav(self, x, z): return haversine(x[0,0], z[0,0], x[1,0], z[1,0])

    def F(self, window, x): 
        dt = window[self._time].iloc[1] - window[self._time].iloc[0]
        self._F[0,2] = self._F[1,3] = dt
        return self._F

    def z(self, window):
        dt = window[self._time].iloc[1] - window[self._time].iloc[0]
        return np.array([
            window[self._lat].iloc[1], 
            window[self._lon].iloc[1],
            (window[self._lat].iloc[1]-window[self._lat].iloc[0])/dt, 
            (window[self._lon].iloc[1]-window[self._lon].iloc[0])/dt,
        ]).reshape((4,1))

    def x0(self, window):
        dt = window[self._time].iloc[1] - window[self._time].iloc[0]
        return np.array([
            window[self._lat].iloc[1], 
            window[self._lon].iloc[1],
            (window[self._lat].iloc[1]-window[self._lat].iloc[0])/dt, 
            (window[self._lon].iloc[1]-window[self._lon].iloc[0])/dt,
        ]).reshape((4,1))


class KalmanFilter_CV_LLV(KalmanFilter):

    """
    Kalman Filter implementation with a velocity measurement. 
    Velocity in euclidean coords, not pseudo-spherical GPS coords. 
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._F = np.eye(4)
        self._H = np.eye(3,4)
        self._Q = kwargs['Q'] if 'Q' in kwargs else np.eye(4)
        self._R = kwargs['R'] if 'R' in kwargs else np.eye(3)
        self._r = 6371055.0
        self._vel = kwargs['speed_col'] if 'speed_col' in kwargs else 'speed'

    def N(self): return 4
    def M(self): return 3
    def W(self): return 2
    def Q(self): return self._Q
    def R(self): return self._R
    def lat(self, x): return x[0,0]
    def lon(self, x): return x[1,0]
    def err(self, x, z): return np.linalg.norm( x[:2,0] - z[:2,0] )
    def hav(self, x, z): return haversine(x[0,0], z[0,0], x[1,0], z[1,0])

    def h(self, window, x, H):
        y = np.zeros((3,1))
        y[0], y[1] = x[0], x[1]
        y[2] = self._r * np.sqrt( x[2]**2 + 2 * np.cos(x[0])**2 * x[3]**2 )
        return y

    def F(self, window, x): 
        dt = window[self._time].iloc[1] - window[self._time].iloc[0]
        self._F[0,2] = self._F[1,3] = dt
        return self._F

    def H(self, window, x): 
        c   = np.cos(x[0])
        c2  = c**2
        x32 = x[3]**2
        rdv = self._r / np.sqrt( x[2]**2 + 2 * c2 * x32 )
        self._H[2,0] = -2 * rdv * np.sin(x[0]) * c * x32
        self._H[2,2] = rdv * x[2]
        self._H[2,3] = 2 * rdv * c2 * x[3]
        return self._H

    def z(self, window):
        dt = window[self._time].iloc[1] - window[self._time].iloc[0]
        return np.array([
            window[self._lat].iloc[1], 
            window[self._lon].iloc[1],
            window[self._vel].iloc[1],
        ]).reshape((3,1))

    def x0(self, window):
        dt = window[self._time].iloc[1] - window[self._time].iloc[0]
        return np.array([
            window[self._lat].iloc[1], 
            window[self._lon].iloc[1],
            (window[self._lat].iloc[1]-window[self._lat].iloc[0])/dt, 
            (window[self._lon].iloc[1]-window[self._lon].iloc[0])/dt,
        ]).reshape((4,1))
