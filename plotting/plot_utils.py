import numpy as np

def sphere(n):

  theta = np.linspace(0, 2*np.pi, n+1)
  phi = np.linspace(-np.pi/2, np.pi/2, n+1)
  [theta,phi] = np.meshgrid (theta, phi)

  x = np.cos (phi) * np.cos (theta)
  y = np.cos (phi) * np.sin (theta)
  z = np.sin (phi)

  return x, y, z


# def cylinder_octave(r, n):

    # phi = np.linspace (0, 2*np.pi, n+1);
    # idx = 1:length (r);
    # [phi, idx] =np.meshgrid (phi, idx);
    # z = (idx - 1) / (length (r) - 1);
    # r = r(idx);
    # [x, y] = pol2cart (phi, r);




# from http://python4econ.blogspot.com/2013/03/matlabs-cylinder-command-in-python.html
def cylinder(r,n):
    '''
    Returns the unit cylinder that corresponds to the curve r.
    INPUTS:  r - a vector of radii
             n - number of coordinates to return for each element in r

    OUTPUTS: x,y,z - coordinates of points
    '''

    # ensure that r is a column vector
    r = np.atleast_2d(r)
    r_rows,r_cols = r.shape
    
    if r_cols > r_rows:
        r = r.T

    # find points along x and y axes
    points  = np.linspace(0,2*np.pi,n+1)
    x = np.cos(points)*r
    y = np.sin(points)*r

    # find points along z axis
    rpoints = np.atleast_2d(np.linspace(0,1,len(r)))
    z = np.ones((1,n+1))*rpoints.T
    
    return x,y,z
