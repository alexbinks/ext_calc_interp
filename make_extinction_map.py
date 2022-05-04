# author: Alex Binks
# date: March 2022

# Program which interpolates the "stilism" dust maps.
# https://arxiv.org/pdf/1706.07711.pdf
# to find E(B-V) as a function of Galactic coordinates
# (l, b) and distance (here 1000./parallax).
# Stilism grids only go out to 300pc in the galactic
# plane, and it seems that in regions where |z|>300pc
# the numbers returned from Stilism are clipped at the
# value of the furthest available distance along the
# line of sight. We replace these values using linear
# extrapolation to fits of E(B-V) versus distance
# along these lines in l and b. Not perfect, but
# possibly better than leaving them at a fixed value.


# The grid comprises of ~10^5 node points in l, b and
# distance which were calculated using stilism.py,
# basically a tool to generate a url, and extract the
# results to a text file. They have dimensions of 
# l=longditude: 0 < l (every 5 degrees) < 360
# b=latitutde:-90 < b (every 5 degrees) < +90
# d=distance:   0 < d (every 25 parsec) < 500
# The distance step is chosen to cover the structural
# length scale typical of a star-forming region.
# Too small and our extinction maps become jagged (we
# want a smooth function), too large and we fail to
# resolve extincted regions. Also 25 pc saves on
# download time.

# Minimal required input data is position and parallax.
# Interpolation uses the distance to the eight nearest
# node points within the sphere, using the solution
# provided by Jack D'Auruzio
# https://math.stackexchange.com/questions/1078231/distance-between-2-points-in-3d-space-in-spherical-polar-coordinates
# An inverse distance weighted average is found for the
# 8 E(B-V) values at each of the node points.

import numpy as np
from astropy.io import ascii, fits
from astropy.table import Table
import astropy.units as u
import time
from astropy.coordinates import SkyCoord
import math
from scipy.interpolate import RegularGridInterpolator

# Function to find the values at each of the 8 node
# points closest to the data. If longditude (l) is
# >355, then set the upper limit to zero (to wrap
# everything back around).
def lbd_limits(i, v1m, v2m, v3m, v1, v2, v3):
    if v1 > 355.0:
        x1u = 0
    else:
        x1u = v1m[v1m >= v1].min()
    x1l = v1m[v1m <= v1].max()
    x2u, x2l = v2m[v2m >= v2].min(), v2m[v2m <= v2].max()
    x3u, x3l = v3m[v3m >= v3].min(), v3m[v3m <= v3].max()
    return x1l, x1u, x2l, x2u, x3l, x3u

# Function to return the indices for each of the 8 node
# points. 
def get_vertices(a1, a2, a3, p):
    v1l, v1u, v2l, v2u, v3l, v3u = \
    p[0], p[1], p[2], p[3], p[4], p[5]
    g000 = [(a1==v1l) & (a2==v2l) & (a3==v3l)]
    g001 = [(a1==v1l) & (a2==v2l) & (a3==v3u)]
    g010 = [(a1==v1l) & (a2==v2u) & (a3==v3l)]
    g100 = [(a1==v1u) & (a2==v2l) & (a3==v3l)]
    g011 = [(a1==v1l) & (a2==v2u) & (a3==v3u)]
    g101 = [(a1==v1u) & (a2==v2l) & (a3==v3u)]
    g110 = [(a1==v1u) & (a2==v2u) & (a3==v3l)]
    g111 = [(a1==v1u) & (a2==v2u) & (a3==v3u)]
    return np.where(g000)[1], np.where(g001)[1], \
           np.where(g010)[1], np.where(g100)[1], \
           np.where(g011)[1], np.where(g101)[1], \
           np.where(g110)[1], np.where(g111)[1]




# read in the extinction map
em = ascii.read("cpoints_orig_red2.csv", format='csv')


# read in the data and convert to galactic coordinates
with fits.open('Gaia_eDR3_master_light.fits', mode='update') as hdu:
    head = hdu[0].header
    data = hdu[1].data
    ra_x, dec_x, plx_x = data["ra"][4354600:4355000], data["dec"][4354600:4355000], data["parallax"][4354600:4355000]

c_icrs = SkyCoord(ra=ra_x*u.degree, dec=dec_x*u.degree, frame='icrs')
gal_x = c_icrs.galactic
#ltest, btest, dtest = gal_x.l.degree.ravel(), gal_x.b.degree.ravel(), 1000.0/plx_x
ltest, btest, dtest = np.linspace(1,359, 1000), np.linspace(-89, 89, 1000), np.linspace(1,500, 1000)
print(ltest)
print(btest)
print(dtest)
# set up the extinction map.
l, b, dis, EBV, dis_s = em["l"], em["b"], em["dis"], em["EBV_fix"], em["dis_s"]
l_map, b_map = np.unique(l), np.unique(b)

# find the values of the node points for every data point, store as array...
c = [np.asarray(lbd_limits(i, l, b, dis, ltest[i], btest[i], dtest[i])) for i in range(len(ltest))]
c = np.asarray(c)

# convert the data points l and b to radians (for the calculation at the end).
ltest, btest = ltest*(math.pi/180.), btest*(math.pi/180.)

# store the indices.
d = [np.asarray(get_vertices(l, b, dis, c[i])).ravel() for i in range(len(c))]
d = np.asarray(d)

# convert the node points l and b to radians (for the calculation at the end).
l, b = l*(math.pi/180.), b*(math.pi/180.)


# The maths part. 
# Set up a loop over all the data points   
for j in range(len(ltest)):
    # initialize 2 arrays for storing EBV values and distances to each node point. 
    # Refreshes on each iteration.
    M = []
    r = []
    # Loop over each of the node points.
    for i in d[j]:
        # Fancy spherical distance calculations.
        a1 = dis[i]**2 + dtest[j]**2
        a2 = 2.0*dis[i]*dtest[j]
        a3 = np.cos(l[i])*np.cos(ltest[j])*np.cos(b[i]-btest[j])
        a4 = np.sin(l[i])*np.sin(ltest[j])
        # Fill the EBV and distance arrays.
        M.append(np.sqrt(a1 - a2*(a3 + a4)))
        r.append(EBV[i])
#        print(dis[i], b[i]*180./math.pi, l[i]*180./math.pi, EBV[i])
    # print out the weighted average EBV value (note the weights are 1.0/M since closer distance = better
    # weighting. Consider using inverse distance squared maybe.)
    print(ltest[j]*180./math.pi, btest[j]*180./math.pi, dtest[j], np.ma.average(r, weights=1.0/np.array(M)))


# Old code to extrapolate the extinction maps beyond |z|>300pc.
#r_arr = []
#for lp in l_map:
#    for bp in b_map:
#        gx = em[(l==lp) & (b==bp)]
#        gd = em[(dis_s==dis) & (l==lp) & (b==bp)]
#        rf = np.polyfit(gd["dis"], gd["EBV_s"], 1)
#        for k in gx:
#           print(k)
#           if k["dis"] == k["dis_s"]:
#               r_arr.append(k["EBV_s"])
#           else:
#               r_arr.append(rf[1] + rf[0]*k["dis"])
#ascii.write([r_arr], 'values.dat', overwrite=True)  

