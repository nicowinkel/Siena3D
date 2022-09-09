from siena3d.astrometry import Astrometry

datacube = 'Mrk1044_example.fits'   # input file

astrometry = Astrometry(datacube)
astrometry.run()
