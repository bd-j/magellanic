import numpy as np
import matplotlib.pyplot as pl
pl.rcParams['image.origin'] = 'lower'
pl.rcParams['image.interpolation'] = 'nearest'
pl.rcParams['image.aspect'] = 'equal'
import fsps
from magellanic import mcutils, datautils
import astropy.io.fits as pyfits

imdir = '/Users/bjohnson/Projects/magellanic/images/'
imname_map = {'lmc':{'irac_1':imdir+'SAGE_LMC_IRAC3.6_2_mosaic.fits',
                     'irac_2':imdir+'SAGE_LMC_IRAC4.5_2_mosaic.fits',
                     'irac_3':imdir+'SAGE_LMC_IRAC5.8_2_mosaic.fits',
                     'irac_4':imdir+'SAGE_LMC_IRAC8.0_2_mosaic.fits',
                     '2mass_ks': None},
              'smc':{'irac_1':imdir+'SAGE_SMC_IRAC3.6_2_mosaic.fits',
                     'irac_2':imdir+'SAGE_SMC_IRAC4.5_2_mosaic.fits',
                     'irac_3':imdir+'SAGE_SMC_IRAC5.8_2_mosaic.fits',
                     'irac_4':imdir+'SAGE_SMC_IRAC8.0_2_mosaic.fits'
                     #'2mass_ks':imdir+'IRSF_kSMC.fits',
                     #'2mass_ks':imdir+'IRSF_kSMC.fits',
                     #'2mass_ks':imdir+'IRSF_kSMC.fits'
                     }
              }


def build_observed_image(cloud, rnames, imname):
    """
    :param cloud:
        'lmc' or 'smc'
        
    :param rnames:
        A list of H & Z region names (e.g. GH_01)
        
    :param imname:
        String giving image name (including path ) of the original
        observed image, assumed to be in MJy/sr
        
    :returns im:
        Array of observed fluxes, in units of AB maggies. (assuming
        input image is in MJy/sr).  The shape depends on wich cloud.
    """
    crpix, crval, cdelt, [nx, ny] = mcutils.mc_ast(cloud, badenes=True)
    im = np.zeros([nx, ny])
    mags = datautils.region_photometer(cloud, rnames, imname)
    for name, mag in zip(rnames, mags):
        x, y = mcutils.regname_to_xy(name, cloud=cloud, badenes=True)
        im[x, y] = 10**(-0.4*mag) / np.size(x)
    return im
          
def build_observed_images_complicated(cloud, rnames, imnames):
    """Like above but works for more complicated regions.  Hoever,
    since the H&Z regions are simple, not necessary
    """
    import sedpy.ds9region as ds9
    crpix, crval, cdelt, [nx, ny] = mcutils.mc_ast(cloud, badenes=True)
    im = np.zeros([len(imnames), nx, ny])
    for j, imname in enumerate(imnames):
        if imname is None:
            continue
        regions, xs, ys = [], [], []
        for k, name in enumerate(rnames):
            x, y = mcutils.regname_to_xy(name, cloud=cloud)
            shape, defstring = mcutils.corners_of_region(name, cloud.lower(),
                                                         string=True)
            regions.append(ds9.Polygon(defstring))
            xs.append(x)
            ys.append(y)
            
        mags = datautils.photometer(imname, regions)
        for x, y, mag in zip(xs, ys, mags):
            im[j, x, y] = 10**(-0.4*mag) / np.size(x)

    return im

if __name__ == "__main__":
    istring = 'hz_images/HZ_{0}_{1}.fits'
    for cloud in ['lmc', 'smc']:
        regions = mcutils.mc_regions(cloud)
        regions.pop('header')
        rnames = regions.keys()
        for fband, imname in imname_map[cloud].iteritems():
            if imname is None:
                continue
            #im = build_observed_image(cloud, rnames, imname)
            im = build_observed_images_complicated(cloud, rnames, [imname])
            im = np.squeeze(im)
            h = pyfits.hdu.image.ImageHDU(im)
            h.header['BUNIT'] = 'AB maggies'
            h.header['Comment'] = "AXIS1, AXIS2 correspond to Harris and Zaritsky regions."
            h.header['ORIGFILE'] = imname
            h.header['OBJECT'] = cloud.upper()
            h.header['BAND'] = fband
            #h.header['DATE'] =
            print('wrote ' + istring.format(cloud, fband))
            pyfits.writeto(istring.format(cloud, fband), im, h.header, clobber=True)
            
