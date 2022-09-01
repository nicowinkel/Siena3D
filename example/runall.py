from siena import cube
from siena import Astrometry

cz = 4920

datacube = '/home/winkel/Data/MUSE/NFM/HE0227-0913/DataCube/Mrk_1044_DATACUBE_FINAL_2019-08-24T08:47:37.086.fits'
qsotable = 'par_table.fits'
workdir = '.'

astrometry = Astrometry(datacube, qsotable, cz)
astrometry.print_result()
astrometry.write(workdir)
