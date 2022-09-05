from siena.astrometry import Astrometry

datacube = 'Mrk1044_example.fits'   # input file

astrometry = Astrometry(datacube, cz)
astrometry.run()
