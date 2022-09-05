from siena.astrometry import Astrometry

cz = 4913.4

datacube = '/home/winkel/Data/MUSE/NFM/HE0227-0913/DataCube/Mrk_1044_DATACUBE_FINAL_2019-08-24T08:47:37.086.fits'
workdir = '.'

astrometry = Astrometry(datacube, cz)
astrometry.run()
