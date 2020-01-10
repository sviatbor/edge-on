import itertools
import timeit
from numba import jit
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from astropy.io import fits
import math
import warnings
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import *
from surf_fit import surf_fit
import scipy as sci
import copy
import os.path
import os

warnings.filterwarnings("ignore")
start = timeit.default_timer()

def polyfit2d(x, y, z, order):
  ncols = (order + 1)**2
  G = np.zeros((x.size, ncols))
  ij = itertools.product(range(order+1), range(order+1))
  for k, (i, j) in enumerate(ij):
    G[:, k] = x**i * y**j
  m, _, _, _ = np.linalg.lstsq(G, z)
  return m

def polyval2d(x, y, m):
  order = int(np.sqrt(len(m))) - 1
  ij = itertools.product(range(order+1), range(order+1))
  z = np.zeros_like(x)
  for a, (i, j) in zip(m, ij):
    z += a * x**i * y**j
  return z

def mask_image(image, square_nan_size, n_iter, n_sigma):
    for i in range(n_iter):
        if i == 1:
            image_temp = image
        if i >= 1:
            image = image_temp
            image_temp = image
            med_temp = np.nanmedian(image_temp)
            sig_temp = np.nanstd(image_temp)
            ind_nan = np.where(image_temp >= med_temp+n_sigma*sig_temp)
            num_nan = (ind_nan[1]).shape[0]
            image_temp[ind_nan] = np.NaN
            for j in range(num_nan):
                x = (ind_nan[0])[j]
                y = (ind_nan[1])[j]
                image[x-square_nan_size:x+square_nan_size, y-square_nan_size:y+square_nan_size] = np.NaN
    return image

survey = 'sdss'
galname = '1440'
band = 'g'
extension = 0
order = 3
image_file = fits.open('./'+galname+'/'+galname+'_'+survey+'_'+band+'.fits')
header = image_file[extension].header
hdu = image_file[extension]
image = hdu.data
size = image.shape
image_orig = copy.copy(image)

n_iter = 6
n_sigma = 3
square_nan_size = int(np.ceil(np.sqrt(size[0]*size[1])/150.))

image = mask_image(image, square_nan_size, n_iter, n_sigma)

llim = np.median(image_orig)
ulim = np.mean(image_orig[(image_orig>np.mean(image_orig)+1*np.std(image_orig)) & 
  (image_orig<np.mean(image_orig)+3*np.std(image_orig))])
fig, ax = plt.subplots(1, 2)
ax[0].imshow(image_orig, vmin=llim, vmax=ulim)
ax[1].imshow(image, vmin=llim, vmax=ulim)
plt.show(block=False)

square_size = int(square_nan_size*1.5)
height = int(np.floor(size[0]/square_size))
width = int(np.floor(size[1]/square_size))
nan_thr = 0.3
image_trunc = image[0:height*square_size, 0:width*square_size]
image_rebin = image_trunc.reshape((height, image_trunc.shape[0]//height, width, -1))

image_rebin_nanmean = np.zeros(shape=(height*width, 1))
x_center = np.zeros(shape=(height*width, 1))
y_center = np.zeros(shape=(height*width, 1))

num = 0
for i in range(image_rebin.shape[0]):
  for j in range(image_rebin.shape[2]):
    x_center[num, 0] = j*square_size+np.floor(square_size/2)+1
    y_center[num, 0] = i*square_size+np.floor(square_size/2)+1
    square = image_rebin[i, 0:square_size, j, 0:square_size]
    n_nan = (np.argwhere(np.isnan(square))).shape[0]
    if (float(n_nan)/square_size**2 <= nan_thr):
      image_rebin_nanmean[num, 0] = np.nanmean(square)
    else:
      image_rebin_nanmean[num, 0] = np.NaN 
    num += 1

ind_good = np.where(~np.isnan(image_rebin_nanmean))
num_good = ind_good[0].shape[0]
x = x_center[ind_good]
y = y_center[ind_good]
z = image_rebin_nanmean[ind_good]


data = np.zeros(shape=(num_good, 3))
for i in range(num_good):
  data[i, :] = [x[i], y[i], z[i]]

poly_coeff = polyfit2d(data[:, 0], data[:, 1], data[:, 2], order)
X, Y = np.meshgrid(np.linspace(0, size[1], 50), 
             np.linspace(0, size[0], 50))
Z = polyval2d(X, Y, poly_coeff)



llim = np.nanmedian(data[:, 2])-5*np.nanstd(data[:, 2])
ulim = np.nanmedian(data[:, 2])+5*np.nanstd(data[:, 2])
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.9)
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='r', s=20)
ax.set_zlim3d(llim, ulim)
plt.xlabel('X')
plt.ylabel('Y')
ax.set_zlabel('Z')
plt.show(block=False)


X, Y = np.meshgrid(np.linspace(0, size[1], size[1]), 
             np.linspace(0, size[0], size[0]))
Z = polyval2d(X, Y, poly_coeff)

image_orig -= Z

if os.path.exists(galname+'_'+band+'_bg.fits'):
  os.remove(galname+'_'+band+'_bg.fits')
fits.writeto(galname+'_'+band+'_bg.fits', image_orig, header)





