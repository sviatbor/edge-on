from matplotlib.patches import Circle, Ellipse, Rectangle
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import optimize,misc
import numpy as np
import warnings
import timeit
import sewpy
import copy
import sys
import os

warnings.filterwarnings("ignore")
start = timeit.default_timer()

print
print "FINDING GALAXY AND ELIMINATION OF SURROUNDING OBJECTS"
print "-----------------------------------------------------"

def rot_image(image, xy, pos_angle):
    print "IMAGE ROTATION"
    im_rot = rotate(image, pos_angle, reshape=False)
    org_center = (np.array(image.shape[:2][::-1])-1)/2.
    rot_center = (np.array(im_rot.shape[:2][::-1])-1)/2.
    org = xy.T-org_center.T
    a = np.deg2rad(pos_angle)
    new = np.array([org[:,0]*np.cos(a)+org[:,1]*np.sin(a),
            -org[:,0]*np.sin(a)+org[:,1]*np.cos(a)])
    return im_rot, (new.T+rot_center.T).T

def in_ellipse(g_ell_center, ellipse_a, ellipse_b, img_tmp):
    size_img = img_tmp.shape
    for i_ell in range(size_img[0]):
        for j_ell in range(size_img[1]):
            xc = i_ell-g_ell_center[0]
            yc = j_ell-g_ell_center[1]
            rad_ell = (xc**2/(ellipse_b)**2) + (yc**2/(ellipse_a)**2)
            if rad_ell > 1.:
                img_tmp[i_ell,j_ell] = np.NaN
    return img_tmp

def fix_rot(image, xyc, a, b, angle_var):
    ang_rot = np.linspace(-angle_var, angle_var, 20*angle_var+1)    
    xlim = np.array([int(np.round(xyc[0]-(a+5))),int(np.round(xyc[0]+(a+5)))])
    if ((xlim[1]-xlim[0]+1)%2 == 1):
        xlim[0] -=1
    ylim = np.array([int(np.round(xyc[1]-(b+5))),int(np.round(xyc[1]+(b+5)))])
    if ((ylim[1]-ylim[0]+1)%2 == 1):
        ylim[0] -= 1
    data_trunc = image[ylim[0]:ylim[1]+1, xlim[0]:xlim[1]+1]
    size_trunc = data_trunc.shape
    xyc[0] = (size_trunc[0]+1)/2
    xyc[1] = (size_trunc[1]+1)/2
    for i in range(len(ang_rot)):
        data_tmp, (xtmp, ytmp) = rot_image(data_trunc, xyc, ang_rot[i])
        size_tmp = data_tmp.shape
        xtmp = np.round(size_tmp[0]/2)
        ytmp = np.round(size_tmp[1]/2)
        data_tmp = in_ellipse(np.array([xtmp,ytmp]),a,b,data_tmp)
        if (size_tmp[1]%2 == 0):
            xlim1 = int(np.round((size_tmp[1])/2))
            xlim2 = int(np.floor((size_tmp[1]-1)/2))+1
        else:
            xlim1 = int(np.round((size_tmp[1])/2))
            xlim2 = int(np.round((size_tmp[1])/2))+1
        data1 = data_tmp[0:size_tmp[0]+1,xlim1:size_tmp[1]+1]
        data2 = data_tmp[0:size_tmp[0]+1,0:xlim2]
    return data_tmp

def surf_brightness(image, xyc, r):
    img_tmp = image[int(xyc[1]-r[1]):int(xyc[1]+r[1]+1),int(xyc[0]-r[0]):int(xyc[0]+r[0]+1)]
    sb = np.mean(img_tmp)
    return sb

def replace_region(img, xc, yc, r, x1, y1, br_gal):
    size_img = img.shape
    xcc = xc
    ycc = int(2*y1)-yc
    if (ycc-r<=y1) & (ycc+r>y1):
        ycc = yc
        xcc = int(2*x1)-xc
        if (np.mean(data[int(y1-2):int(y1+2),int(xc-2):int(xc+2)])<=surf_br_gal/10):
            if (xc>=x1) & (xc+3*r+1<=size_img[1]):
                img_tmp = img[xc+r:xc+3*r+1,ycc-r:ycc+r+1]
                img_tmp = in_ellipse(np.array([r,r]),r,r,img_tmp)
                img[xc+r:xc+3*r+1,ycc-r:ycc+r+1] = img_tmp
#                img = in_ellipse(np.array([xc+2*r,ycc]),r,r,img)            
            if (xc<x1) & (xc-3*r>=0):
                img_tmp = img[xc-3*r:xc-1*r+1,ycc-r:ycc+r+1]
                img_tmp = in_ellipse(np.array([r,r]),r,r,img_tmp)
                img[xc-3*r:xc-1*r+1,ycc-r:ycc+r+1] = img_tmp
#                img = in_ellipse(np.array([xc-2*r,ycc]),r,r,img)
        else:
                img_tmp = img[2*x1-xc-r:2*x1-xc+r+1,ycc-r:ycc+r+1]
                img_tmp = in_ellipse(np.array([r,r]),r,r,img_tmp)
                img[2*x1-xc-r:2*x1-xc+r+1,ycc-r:ycc+r+1] = img_tmp
#                img = in_ellipse(np.array([2*x1-xc,ycc]),r,r,img)            
    else:
        img_tmp = img[xc-r:xc+r+1,yc-r:yc+r+1]
        img_tmp = in_ellipse(np.array([r,r]),r,r,img_tmp)
        img = in_ellipse(np.array([xc,yc]),r,r,img)
    return img  

def sextractor(image_name):
    sew = sewpy.SEW(params=['X_IMAGE', 'Y_IMAGE', 'FLUX_RADIUS(3)', 'THETA_IMAGE','ELLIPTICITY','A_IMAGE',
        'B_IMAGE','DISK_THETA_IMAGE'], config={'DETECT_MINAREA':5, 'PHOT_FLUXFRAC':'0.5, 1.0, 1.0'})
    str_out = sew(image_name)
    obj_tab = str_out['table']
    return obj_tab

def read_image(catalog, survey, galname, band, extension):
    print "READING THE IMAGE"
    image_file = fits.open('./'+catalog+galname+'/'+catalog+galname+'_'+survey+'_'+band+'.fits')
    header = image_file[extension].header
    hdu = image_file[extension]
    image = hdu.data
    size = image.shape
    image_orig = copy.copy(image)
    return image, image_orig, size, header         

def rotation_plot(image, data_rot, catalog, galname):
    llim = np.median(image)
    ulim = np.mean(image[(image>np.mean(image)+1*np.std(image)) &
        (image<np.mean(image)+3*np.std(image))])
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(image,vmin=llim,vmax=ulim)
    ax[1].imshow(data_rot,vmin=llim,vmax=ulim)
    ax[0].set_title(catalog.upper()+galname)
    ax[1].set_title(catalog.upper()+galname+' rotated')
    plt.show(block=False)

catalog = 'fgc'
survey = 'sdss'
galname = '1440'
band = 'r'
extension = 0
image_name = './'+catalog+galname+'/'+catalog+galname+'_'+survey+'_'+band+'_bg.fits'

image, image_orig, size, header = read_image(catalog, survey, galname, band, extension)

obj_tab = sextractor(image_name)
x_image = np.float64(obj_tab['X_IMAGE'])
y_image = np.float64(obj_tab['Y_IMAGE'])
flux_rad_0 = np.float64(obj_tab['FLUX_RADIUS'])
flux_rad_1 = np.float64(obj_tab['FLUX_RADIUS_1'])
flux_rad_2 = np.float64(obj_tab['FLUX_RADIUS_2'])
pos_angle = np.float64(obj_tab['DISK_THETA_IMAGE'])
ellipticity = np.float64(obj_tab['ELLIPTICITY'])
a_axis = np.float64(obj_tab['A_IMAGE'])
b_axis = np.float64(obj_tab['B_IMAGE'])
a_axis_list = list(a_axis)

a_axis_sort = list(sorted(a_axis_list, reverse=True))
ind_gal = [a_axis_list.index(a_axis_sort[i]) for i in range(len(a_axis_list))][0]
a_axis = a_axis[ind_gal]
b_axis = b_axis[ind_gal]

xx, yy = x_image, y_image
x0, y0 = x_image[ind_gal], y_image[ind_gal]
pos_angle = pos_angle[ind_gal]

data_rot, (xx1,yy1) = rot_image(image, np.array([xx,yy]), pos_angle)
x1 = xx1[ind_gal]
y1 = yy1[ind_gal]

rotation_plot(image, data_rot, catalog, galname)
nb = 12
na = 5
data = data_rot[int(y1-b_axis*nb):int(y1+b_axis*nb)+1,int(x1-a_axis*na):int(x1+a_axis*na)+1]
size_data = data.shape
yy1 -= int(y1-b_axis*nb)
xx1 -= int(x1-a_axis*na)
y1 -= int(y1-b_axis*nb)
x1 -= int(x1-a_axis*na)

####################SELECT OBJECTS CLOSE TO THE GALAXY##################
#################EXCLUDE THE GALAXY ITSELF FROM THIS LIST###############
ind_inner = np.where((xx1>=0) & (xx1<=size_data[1]) &\
(yy1>=0) & (yy1<=size_data[0]))
x_obj = list(xx1[ind_inner]-1)
y_obj = list(yy1[ind_inner]-1)
rad_obj_30 = list(flux_rad_0[ind_inner])
rad_obj_90 = list(flux_rad_1[ind_inner])
rad_obj_100 = list(flux_rad_2[ind_inner])
ind = np.where(rad_obj_100 == np.max(rad_obj_100))[0][0]
rad_obj_30.pop(ind)
rad_obj_90.pop(ind)
rad_obj_100.pop(ind)
x_obj.pop(ind)
y_obj.pop(ind)
#######################################################################

################EXCLUDE OBJECTS IN THE GALAXY PLANE####################
#############THEY ARE PROBABLY THE STARFORMATION REGIONS###############
ind_plane = np.where((abs(x_obj-x1)<=2*a_axis) & (abs(y_obj-y1)<=b_axis/2))
if ind_plane[0].size != 0:
    for i in range(len(ind_plane[0])):
        x_obj.pop(ind_plane[0][i])
        y_obj.pop(ind_plane[0][i])
        rad_obj_30.pop(ind_plane[0][i])
        rad_obj_90.pop(ind_plane[0][i])
        rad_obj_100.pop(ind_plane[0][i])
######################################################################
data_orig = copy.copy(data)
#################CALCULATE MEAN SURFACE BRIGHNESS#####################
#######################OF EACH CLOSE OBJECT###########################
surf_br_gal = surf_brightness(data,np.array([x1,y1]),np.array([a_axis, b_axis]))
surf_br = np.zeros((len(x_obj),1))
rad_obj = np.zeros((len(x_obj),1))
for i in range(len(x_obj)):
    surf_br[i] = surf_brightness(data,np.array([x_obj[i],y_obj[i]]),np.array([rad_obj_30[i],rad_obj_30[i]]))
    if surf_br[i]>=3*surf_br_gal:
        rad_obj[i] = rad_obj_100[i]*2.5
    else:
        rad_obj[i] = rad_obj_90[i]
    ylim11 = max([int(y_obj[i]-rad_obj[i]+1),0])
    ylim12 = min([int(y_obj[i]+rad_obj[i]+1),size_data[0]]) 
    ylim21 = np.max([int(2*y1)-ylim12,0])
    ylim22 = ylim21+(ylim12-ylim11)
    xlim11 = int(x_obj[i]-rad_obj[i]+1)
    xlim12 = int(x_obj[i]+rad_obj[i]+1)
    xlim21 = xlim11
    xlim22 = xlim12
#    if i == 9:
#        print ylim11
#        print ylim12 
#        print xlim11
#        print xlim12
#        print '-----------'
#        print ylim21
#        print ylim22
#        print xlim21
#        print xlim22
    if (ylim11<=y1) & (ylim12>y1):
        ylim11 = max([int(y_obj[i]-rad_obj[i]+1),0])
        ylim12 = min([int(y_obj[i]+rad_obj[i]+1),size_data[0]])
        ylim21 = ylim11
        ylim22 = ylim12
        xlim11 = int(x_obj[i]-rad_obj[i]+1)
        xlim12 = int(x_obj[i]+rad_obj[i]+1)
        if (np.mean(data[int(y1-2):int(y1+2),int(x_obj[i]-2):int(x_obj[i]+2)])<=surf_br_gal/10):
            if (x_obj[i]>=x1) & (int(x_obj[i]+3*rad_obj[i]+2) <= size_data[1]):
                xlim21 = int(x_obj[i]+rad_obj[i]+1)
                xlim22 = xlim21+(xlim12-xlim11)
            if (x_obj[i]<x1) & (int(x_obj[i]-3*rad_obj[i]) >= 0):
                xlim21 = int(x_obj[i]-3*rad_obj[i])
                xlim22 = int(x_obj[i]-rad_obj[i])
        else:            
            xlim21 = int(2*x1)-xlim12
            xlim22 = int(2*x1)-xlim11
        data[ylim11:ylim12,xlim11:xlim12] = np.fliplr(data[ylim21:ylim22,xlim21:xlim22])
    else:
        data[ylim11:ylim12,xlim11:xlim12] = np.flipud(data[ylim21:ylim22,xlim21:xlim22])
    

fig, ax = plt.subplots(2,1)
ax[0].imshow(data_orig,vmin=0,vmax=0.3)
ax[1].imshow(data,vmin=0,vmax=0.3)
ax[0].set_title(catalog.upper()+galname)
for i in range(len(x_obj)):
    #    sq = plt.Rectangle((x_obj[i]-rad_obj_30[i],y_obj[i]-rad_obj_30[i]),2*rad_obj_30[i],2*rad_obj_30[i],color='r',fill=False)
    circ = plt.Circle((x_obj[i],y_obj[i]),rad_obj_100[i],color='r',fill=False)
    ax[0].add_patch(circ)
#    ax.add_patch(sq)
#sq = plt.Rectangle((x1-a,y1-b),2*a,2*b,color='r',fill=False)
#ax.add_patch(sq)
#ax[0].add_patch(Ellipse((x1,y1),width=2*a,height=2*b,edgecolor='r',facecolor='none',linewidth=2))
plt.show(block=False)


if os.path.exists(galname+'_'+band+'_2.fits'):
    os.remove(galname+'_'+band+'_2.fits')
data2 = copy.copy(data)
data2[int(y1-2*b_axis):int(y1+2*b_axis),int(x1-4*a_axis):int(x1+4*a_axis)] = 0
fits.writeto(galname+'_'+band+'_2.fits',data2)

obj_tab = sextractor(galname+'_'+band+'_2.fits')
x_image = np.float64(obj_tab['X_IMAGE'])
y_image = np.float64(obj_tab['Y_IMAGE'])
a_axis = np.float64(obj_tab['A_IMAGE'])
b_axis = np.float64(obj_tab['B_IMAGE'])
surf_br2 = np.zeros((len(x_image),1))
for i in range(len(x_image)):
    surf_br2[i] = surf_brightness(data2,np.array([x_image[i],y_image[i]]),[1,1])
ind_max = np.where(surf_br2 == np.max(surf_br2))[0][0]
rad_max = (a_axis[ind_max]+b_axis[ind_max])/2

ylim11 = max([int(y_image[ind_max]-rad_max+1),0])
ylim12 = min([int(y_image[ind_max]+rad_max+1),size_data[0]])
ysize = ylim12-ylim11+1
ylim21 = int(2*y1)-ylim12
ylim22 = int(2*y1)-ylim11
data[ylim11:ylim12,int(x_image[ind_max]-rad_max+1):int(x_image[ind_max]+rad_max+1)] = \
np.flipud(data[ylim21:ylim22,int(x_image[ind_max]-rad_max+1):int(x_image[ind_max]+rad_max+1)])


#width=245
#height = 150
#data = data[:,int(x1-width):int(x1+width)]
'''
fig, ax = plt.subplots(2,1)
ax[0].imshow(data_orig,vmin=0,vmax=0.3)
ax[1].imshow(data,vmin=0,vmax=0.3)
ax[0].set_title(catalog.upper()+galname)
for i in range(len(x_obj)):
    #    sq = plt.Rectangle((x_obj[i]-rad_obj_30[i],y_obj[i]-rad_obj_30[i]),2*rad_obj_30[i],2*rad_obj_30[i],color='r',fill=False)
    circ = plt.Circle((x_obj[i],y_obj[i]),rad_obj_100[i],color='r',fill=False)
    ax[0].add_patch(circ)   
plt.show(block=False)
'''

size_data = data.shape
data+=0.0008
def fit_func(z,i1,z1,xc,i2,zz):
    return i1/(np.cosh((z-xc)/z1))+i2/(np.cosh((z-xc)/zz))

def func1(z,i,z0,xc):
    return i/(np.cosh((z-xc)/z0))
if band == 'i':
    bg = 1.900000880970275e-03
if band == 'g':
    bg = 4.470077578045589e-04
scale = 0.396127
distance = 10.9e6
'''
fig = plt.figure()
ax = plt.axes()
vert_prof = np.array([sum(l) for l in data])
w = np.sqrt(np.abs(vert_prof+bg*size_data[1]))
z = np.linspace(0,(size_data[0]-1),size_data[0])-(size_data[0]-1)/2
params, params_covariance = optimize.curve_fit(fit_func, z,vert_prof, p0=[120,8.8,0,13,31],sigma=w)
print params
plt.errorbar(z,vert_prof,yerr=w,fmt='o')
plt.plot(z,fit_func(z,params[0],params[1],params[2],params[3],params[4]),'r')
plt.plot(z,func1(z,params[0],params[1],params[2]),'blue')
plt.plot(z,func1(z,params[3],params[4],params[2]),'green')

plt.show(block=False)
print '=============='
print 'z1='+repr(params[1]*scale/206265*distance)
print 'z1='+repr(params[4]*scale/206265*distance)
'''
