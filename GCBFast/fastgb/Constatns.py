import numpy as np

### Physical constants
clight = 299792458.0    # Speed of light in vacuum [m.s^-1] (CODATA 2014)
c = clight
G = 6.67408e-11         # Newtonian constant of graviation [m^3.kg^-1.s^-2] (CODATA 2014)
h = 6.626070040e-34     # Planck constant [J.s] (CODATA 2014)

### Derived constants
kg2s = G/(c*c*c)        # Kilogram to seconds

##### Astronomical units
pc = 3.08567758149136727e+16   # Parsec [m] (XXIX General Assembly of the International Astronomical Union, RESOLUTION B2 on recommended zero points for the absolute and apparent bolometric magnitude scales, 2015)
GMsun = 1.32712442099e+20   ## GMsun in SI (http://asa.usno.navy.mil/static/files/2016/Astronomical_Constants_2016.pdf)
#MsunKG = 1.98854695496e+30   # Solar mass [kg] (Stas ?)
MsunKG = GMsun/G
#MsunKG = 1.988475e30         # Solar mass [kg] (Luzum et al., The IAU 2009 system of astronomical constants: the report of the IAU working group on numerical standards for Fundamental Astronomy, Celestial Mechanics and Dynam- ical Astronomy, 2011)
ua = 149597870700.           # Astronomical unit [m](resolution B2 of IAU2012)
YRSID_SI = 31558149.763545600  ## siderial year [sec] (http://hpiers.obspm.fr/eop-pc/models/constants.html)


##### Conversion of masse
#MTsun = 4.92549102554e-06  # Solar mass in seconds (Stas ?)
MTsun = GMsun/c**3
#MTsun = MsunKG*kg2s #  Solar mass (consistent with other definition)

convMass = {'kg':1.,\
            'g':1e-3,\
            'msun':MsunKG,\
            'solarmass':MsunKG}


### Conversion of distance
kpc = 1e3*pc
Mpc = 1e6*pc
Gpc = 1e9*pc
pct = pc/clight  # parsec in seconds

convDistance = {'m':1.,\
                'meter':1.,\
                'km':1000.,\
                'pc':pc,\
                'kpc':kpc,\
                'mpc':Mpc,\
                'gpc':Gpc}


### Conversion of time
year = 365.25 * 24. * 3600

convTime = {'s':1.,\
            'sec':1.,\
            'second':1.,\
            'seconds':1.,\
            'mn':60.,\
            'minute':60.,\
            'h':3600.,\
            'hour':3600,\
            'day':86400.,\
            'year':year,\
            'yr':year}

### Conversion of frequency
convFreq = {'hertz':1.,\
            'hz':1.,\
            'mhz':1.e-3,\
            'khz':1.e3,\
            'muhz':1.e-6,\
            }


### Conversion of time
convAngle = {'rad':1.,\
             'radian':1.,\
             'r':1.,\
             'deg':np.pi/180.,\
             'degree':np.pi/180.
             }


### Convert everything in time
convT = convTime
for x in convDistance:
    convT.update( { x : convDistance[x]/clight } )
for x in convMass:
    convT.update( { x : convMass[x]*kg2s } )
