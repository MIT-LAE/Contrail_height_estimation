cimport cython
cimport numpy as np
import numpy as np
np.import_array()

cdef extern from "math.h":
    double floor(double)
    double ceil(double)
    double round(double)

cdef extern from "numpy/npy_math.h":
    bint npy_isnan(double x)


def interp2d_12(np.ndarray[float, ndim=2, mode="c"] data not None,
                np.ndarray[float, ndim=1, mode="c"] X not None,
                np.ndarray[float, ndim=2, mode="c"] Z not None,
                float x1, float x2, int nx,
                float z1, float z2, int nz):
    """Interpolate 2D data with coordinates given by 1D and 2D arrays.
    data is a two-dimensional array of data to be interpolated.
    X and Z are one- and two-dimensional arrays, giving coordinates
    of data points along the first and second axis, resp.
    data, X and Z are expected to be C-contiguous float32 numpy arrays
    with no mask and no transformation (such as transposition) applied.

    Source: 
        https://github.com/peterkuma/ccplot/blob/master/ccplot/algorithms.pyx
    """
    cdef int i, j, n, m
    cdef float n1, n2
    cdef float m1, m2
    cdef float xs, zs
    cdef int w, h

    xs = (x2 - x1)/nx
    zs = (z2 - z1)/nz
    w = data.shape[0]
    h = data.shape[1]

    output = np.zeros((nx, nz), dtype=np.float32)
    cdef np.ndarray[float, ndim=2] out = output
    cdef np.ndarray[int, ndim=2] q = np.zeros((nx, nz), dtype=np.int32)

    for i in range(w):
        n1 = ((X[i-1] + X[i])/2 - x1)/xs if i-1 >= 0 else -1
        n2 = ((X[i+1] + X[i])/2 - x1)/xs if i+1 < w else nx
        if n2 - n1 < 1: n1 = n2 = (X[i] - x1)/xs

        for j in range(h):
            m1 = ((Z[i,j-1] + Z[i,j])/2 - z1)/zs if j-1 >= 0 else -1
            m2 = ((Z[i,j+1] + Z[i,j])/2 - z1)/zs if j+1 < h else nz
            if m2 - m1 < 1: m1 = m2 = (Z[i,j] - z1)/zs

            for n in range(<int>(n1+0.5), <int>(n2+0.5+1)):
                for m in range(<int>(m1+0.5), <int>(m2+0.5+1)):
                    if n < 0 or n >= nx: continue
                    if m < 0 or m >= nz: continue
                    if npy_isnan(data[i,j]): continue
                    out[n,m] += data[i,j]
                    q[n,m] += 1

    for n in range(nx):
        for m in range(nz):
            out[n,m] /= q[n,m]

    return output

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True) 
cpdef double bicubic_interpolation(double[:] y, double[:] y1, double[:] y2, double[:] y12, double x1l,
                         double x1u, double x2l, double x2u, double x1, double x2):
    
    """
    Uses bicubic interpolation based on function values 'y', derivatives w.r.t. the first coordinate 'y1'
    , w.r.t. the second coordinate 'y2', and the cross-derivative 'y12'. The function values and 
    derivatives are given on the square bounded by 'x1l', 'x1u', 'x2l', 'x2u'. 
    
    Parameters
    ----------
    y: np.array (size 4)
        Function values (numbered counter clockwise from the lower left)
    y1: np.array (size 4)
        Derivatives w.r.t. first coordinate
    y2: np.array (size 4)
        Derivatives w.r.t. second coordinate
    y12: np.array (size 4)
        Cross derivatives
    x1l: float
        lower x1 coordinate
    x1u: float
        upper x1 coordinate
    x2l: float
        lower x2 coordinate
    x2u: float
        upper x2 coordinate
    x1: float
        x1 location to interpolate at
    x2: float
        x2 location to interpolate at
    """
    cdef int i
    cdef double t, u, d1, d2, result
    cdef double c[4][4]
    
    d1 = x1u - x1l
    d2 = x2u - x2l
    
    # Get coefficients
    bicubic_coefficients(y, y1, y2, y12, d1, d2, c)
    
    # Normalized coordinates in square
    t = (x1-x1l)/d1
    u = (x2-x2l)/d2
    
    result = 0.0
    for i in range(3,-1,-1):
        result = t*result + ((c[i][3]*u + c[i][2])*u + c[i][1])*u + c[i][0]
    
    return result
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True) 
cpdef void bicubic_coefficients(double[:] y, double[:] y1, double[:] y2, double[:] y12,
                         double d1, double d2, double[:,:] c):
    """
    Computes the coefficients ``a_{ij}`` of the bicubic interpolant

    ``p(x_1, x_2) = \sum_{i=0}^{3}\sum_{j=0}^{3} a_{ij} x_1^i x_2^j``

    Parameters
    ----------
    y: np.array (size 4)
        Function values (numbered counter clockwise from the lower left)
    y1: np.array (size 4)
        Derivatives w.r.t. first coordinate
    y2: np.array (size 4)
        Derivatives w.r.t. second coordinate
    y12: np.array (size 4)
        Cross derivatives
    d1: float
        Length of grid cell in x1 direction
    d2: float
        Length of grid cell in x2 direction
    c: np.array (size 4 by 4)
        Stores resulting coefficients
    """
    
    # This is the inverse of the matrix A in Ac=x
    # where c are the coefficients of the polynomial 
    cdef double[16][16] weights = [[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                   [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                                   [-3,0,0,3,0,0,0,0,-2,0,0,-1,0,0,0,0],
                                   [2,0,0,-2,0,0,0,0,1,0,0,1,0,0,0,0],
                                   [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                                   [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                                   [0,0,0,0,-3,0,0,3,0,0,0,0,-2,0,0,-1],
                                   [0,0,0,0,2,0,0,-2,0,0,0,0,1,0,0,1],
                                   [-3,3,0,0,-2,-1,0,0,0,0,0,0,0,0,0,0],
                                   [0,0,0,0,0,0,0,0,-3,3,0,0,-2,-1,0,0],
                                   [9,-9,9,-9,6,3,-3,-6,6,-6,-3,3,4,2,1,2],
                                   [-6,6,-6,6,-4,-2,2,4,-3,3,3,-3,-2,-1,-1,-2],
                                   [2,-2,0,0,1,1,0,0,0,0,0,0,0,0,0,0],
                                   [0,0,0,0,0,0,0,0,2,-2,0,0,1,1,0,0],
                                   [-6,6,-6,6,-3,-3,3,3,-4,4,2,-2,-2,-2,-1,-1],
                                   [4,-4,4,-4,2,2,-2,-2,2,-2,-2,2,1,1,1,1]]

    cdef int l, k, j, i
    cdef double xx, d1d2, 
    cdef double cl[16]
    cdef double x[16]
    

    # Set up vector 'x' to multiply with A^-1
    d1d2 = d1*d2
    for i in range(4):
        x[i] = y[i]
        x[i+4] = y1[i]*d1
        x[i+8] = y2[i]*d2
        x[i+12]= y12[i]*d1d2

    # Matrix multiplication of A^-1 with 'x' to yield 'c'
    for i in range(16):
        xx=0.0
        for k in range(16):
            xx += weights[i][k]*x[k]
        cl[i] = xx
    
    # Convert 'c' to array form
    l = 0
    for i in range(4):
        for j in range(4):
            c[i][j] = cl[l]
            l += 1
            
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True) 
cpdef double interpolate_2d(double lon, double lat, double[:] lons, double[:] lats, double[:,:] variable):
    
    cdef double res
    cdef double y1[4] 
    cdef double y2[4]
    cdef double y12[4]
    cdef double y[4]
    cdef int[4] map_i = [2, 2, 1, 1]
    cdef int[4] map_j = [1, 2, 2, 1]
    cdef int i
    
    for k in range(4):
        y[k] = variable[map_i[k]][map_j[k]]

    calculate_derivatives(lons, lats, variable, y1, y2, y12)
    res = bicubic_interpolation(y, y1, y2, y12, lons[1], lons[2], lats[1], lats[2],
                                lon, lat)
    
    return res
            

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True) 
cpdef void polynomial_interpolation(double[:] xa, double[:] ya, double[:, :] Q, int n, double x):
    """
    Computes the 1D polynomial interpolation of degree n-1 at location 'x', given
    function values 'ya' at locations 'xa'. Uses Neville's method to recursively 
    construct the interpolant.
     
    
    Parameters
    ----------
    xa: np.array (of size n)
        Independent variable locations
    ya: np.array (of size n)
        Function values
    Q: np.array (of size n x n)
        Storage for results
    n: int 
        Number of function values
    x: float
        Location to evaluate interpolant
    """
    cdef int i, j
    
    for i in range(n):
        Q[i,0] = ya[i]

    for i in range(1,n):
        for j in range(1,i+1):
            Q[i,j] = ((x - xa[i - j])*(Q[i,j - 1]) - (x - xa[i])*(Q[i- 1,j - 1]))/(xa[i] - xa[i - j])
    
@cython.boundscheck(False)
@cython.cdivision(True) 
cpdef void calculate_derivatives(double[:] x1, double[:] x2, double[:, :] y, double[:] y1, double[:] y2,
                          double[:] y12):
    
    cdef int[4] map_i = [2, 2, 1, 1]
    cdef int[4] map_j = [1, 2, 2, 1]
    cdef int i, j, l
    
    for l in range(4):
        i = map_i[l]
        j = map_j[l]
        
        y1[l] = (y[i][j+1] - y[i][j-1])/(x1[j+1] - x1[j-1])
        y2[l] = (y[i+1][j] - y[i-1][j])/(x2[i+1] - x2[i-1])
        y12[l] = (y[i+1][j+1] - y[i+1][j-1] - y[i-1][j+1] + y[i-1][j-1])/((x1[j+1]-x1[j-1])*(x2[i+1] - x2[i-1]))
        
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True) 
cpdef double interpolate_4d(double lon, double lat, double pres, double t,double[:,:,:,:] variable,
                   double[:] lons, double[:] lats, double[:] pressures, double[:] times):
    
    cdef int i, j, k, l
    cdef double t_vals[2]
    cdef double p_vals[3]
    cdef double Q[3][3]
    cdef double R[2][2]
    cdef double res, dt
    
    for l in range(2):
        for k in range(3):
            p_vals[k] = interpolate_2d(lon, lat, lons, lats, variable[l,k,:,:])
        
        polynomial_interpolation(pressures, p_vals, Q, 3, pres)
        t_vals[l] = Q[2][2]
        
    dt = times[1]-times[0]
    res = t_vals[0]*((times[1]-t)/dt) + t_vals[1]*(t-times[0])/dt
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True) 
cpdef double interpolate_3d(double lon, double lat, double t, double[:,:,:] variable,
                   double[:] lons, double[:] lats, double[:] times):
    
    cdef int i, j, k, l
    cdef double t_vals[2]
    cdef double res, dt
    
    for l in range(2):
        t_vals[l] = interpolate_2d(lon, lat, lons, lats, variable[l,:,:])
        
    dt = times[1]-times[0]
    res = t_vals[0]*((times[1]-t)/dt) + t_vals[1]*(t-times[0])/dt
    return res