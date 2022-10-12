from siena3d.astrometry import Astrometry

datacube = 'Input/Mrk1044.fits'   # input file

astrometry = Astrometry(datacube)
astrometry.run()
