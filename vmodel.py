""" Module for applying the vectorial model """
__author__ = 'Shawn Oset, Lauren Lyons'
__version__ = '0.1'

import numpy as np
from scipy.interpolate import CubicSpline
import scipy.integrate as integrate
import sbpy.activity as sba
import astropy.units as u


class VectorialModel:
    def __init__(self, vModelParams):
        self.vModelParams = vModelParams
        self.par = vModelParams['ParentSpecies']
        self.frag = vModelParams['FragmentSpecies']
        self.comet = vModelParams['Comet']

        self.comet['TimeAtProductions'] = self.comet['TimeAtProductions'].to(u.s).value
        self.numProductionRates = len(self.comet['ProductionRates'])

        self.par['Velocity'] = self.par['Velocity'].to(u.km/u.s).value
        self.par['TotalLifetime'] = self.par['TotalLifetime'].to(u.s).value
        self.par['DissociativeLifetime'] = self.par['DissociativeLifetime'].to(u.s).value

        self.frag['Velocity'] = self.frag['Velocity'].to(u.km/u.s).value
        self.frag['TotalLifetime'] = self.frag['TotalLifetime'].to(u.s).value

        self.sigma = self.par['Sigma'].to(u.km**2).value

        """Initialize data structures to hold our calculations"""
        self.vModel = {}

        # Calculate up a few things
        self._setupCalculations()

        # Build the radial grid
        self.vModel['NumRadialGridpoints'] = vModelParams['Grid']['NumRadialGridpoints']
        self.vModel['FastRadialGrid'] = self.makeLogspaceGrid()
        self.vModel['RadialGrid'] = self.vModel['FastRadialGrid']*(u.km)

        # Angular grid
        self.vModel['NumAngularGridpoints'] = vModelParams['Grid']['NumAngularGridpoints']
        self.vModel['dAlpha'] = self.vModel['EpsilonMax']/self.vModel['NumAngularGridpoints']
        # Make array of angles adjusted up away from zero, to keep from calculating a radial line's contribution
        # to itself
        self.vModel['AngularAlphaGrid'] = np.linspace(0, self.vModel['EpsilonMax'], num=self.vModel['NumAngularGridpoints'], endpoint=False)
        # This maps addition over the whole array automatically
        self.vModel['AngularAlphaGrid'] += self.vModel['dAlpha']/2

        # makes a 2d array full of zero values
        self.vModel['DensityGrid'] = np.zeros((self.vModel['NumRadialGridpoints'],
                                               self.vModel['NumAngularGridpoints']))

        self._computeDensity()
        self._interpolateColumnDensity()

    def _setupCalculations(self):
        """ A few calculations before the heavy lifting starts """

        """ Calculate collision sphere radius based on oldest production rate, Eq. (5) in Festou 1981

            Note that this is only calculated with the first production rate, because it is assumed that the first
            production rate has had enough time to reach a steady state before letting production vary with time.
            We also use that vtherm = 0.25 * outflow velocity
        """
        vtherm = self.par['Velocity']*0.25
        q = self.comet['ProductionRates'][0]
        vp = self.par['Velocity']
        vf = self.frag['Velocity']

        # Eq. 5 of Festou 1981
        self.vModel['CollisionSphereRadius'] = ((self.sigma * q * vtherm)/(vp * vp))*u.km

        """ Calculates the radius of the coma given our input parameters """
        # NOTE: Equation (16) of Festou 1981 where alpha is the percent destruction of molecules
        parentBetaR = -np.log(1.0 - self.par['DestructionLevel'])
        parentR = parentBetaR * vp * self.par['TotalLifetime']
        fragR = vp * self.comet['TimeAtProductions'][0]
        self.vModel['ComaRadius'] = min(parentR, fragR)*u.km

        """ Calculates the time needed to hit a steady, permanent production """
        fragmentBetaR = -np.log(1.0 - self.frag['DestructionLevel'])
        # Permanent flow regime
        permFlowR = self.vModel['ComaRadius'].value + ((vp + vf) * fragmentBetaR * self.frag['TotalLifetime'])

        timeInSecs = self.vModel['ComaRadius'].value/vp + (permFlowR - self.vModel['ComaRadius'].value)/(vp + vf)
        self.vModel['TimeToPermanentFlowRegime'] = (timeInSecs * u.s).to(u.day)

        """ this is the total radial size that parents & fragments occupy, beyond which we assume zero density
        """
        # Calculate the lesser of the radii of two situations
        #       Permanent flow regime
        permFlowR = self.vModel['ComaRadius'].value + ((vp + vf) * fragmentBetaR * self.frag['TotalLifetime'])
        #       or Outburst situation
        outburstR = (vp + vf) * self.comet['TimeAtProductions'][0]
        self.vModel['MaxRadiusOfGrid'] = min(permFlowR, outburstR)*u.km

        # Two cases for angular range of ejection of fragment based on relative velocities of parent and fragment species
        if(vf < vp):
            self.vModel['EpsilonMax'] = np.arcsin(vf/vp)
        else:
            self.vModel['EpsilonMax'] = np.pi

    def productionRateAtTime(self, t):
        """ Get production rate at time t in s, with positive values representing the past, so
            productionRateAtTime(5) will provide the production 5 seconds ago

            For times in the past beyond the specified production rates, returns zero
        """
        if t > self.comet['TimeAtProductions'][0]:
            return 0.0

        for i in range(0, self.numProductionRates):
            binStartTime = self.comet['TimeAtProductions'][i]
            if i == (self.numProductionRates - 1):
                # We're at the end of the array, so stop time is zero seconds ago
                binStopTime = 0
            else:
                # Otherwise we go all the way to the start of the next one
                binStopTime = self.comet['TimeAtProductions'][i+1]

            # NOTE: remember that these times are in seconds ago, so the comparison is backward
            if t < binStartTime and t >= binStopTime:
                return self.comet['ProductionRates'][i]

    def makeLogspaceGrid(self):
        """ Creates a np array radial grid in km with np's logspace function that covers the expected radial size """

        rStartpointPower = np.log10(self.vModel['CollisionSphereRadius'].value * 2)
        rEndpointPower = np.log10(self.vModel['MaxRadiusOfGrid'].value)
        return np.logspace(rStartpointPower, rEndpointPower.astype(float), num=self.vModel['NumRadialGridpoints'], endpoint=True)

    def _computeDensity(self):
        """ Computes the density at different radii and due to each ejection angle, performing the
           radial integration of eq. (36), Festou 1981 with only one fragment velocity.
           The resulting units will be in 1/(km^3) because we work in km, s, and km/s.
        """
        vp = self.par['Velocity']
        vf = self.frag['Velocity']

        # Follow fragments until they have been totally destroyed
        timeLimit = 8.0 * self.frag['TotalLifetime']
        rComa = self.vModel['ComaRadius'].value
        rLimit = rComa

        # temporary radial array for when we loop through 0 to epsilonMax
        # TODO: magic numbers
        ejectionRadii = np.zeros(self.vModelParams['Grid']['SubgridRadialSteps'])

        pTotLifetime = self.par['TotalLifetime']
        fTotLifetime = self.frag['TotalLifetime']
        pDisLifetime = self.par['DissociativeLifetime']

        # Compute this once ahead of time
        # More factors to fill out integral similar to eq. (36) Festou 1981
        IntegrationFactor = (1/(4 * np.pi * pDisLifetime)) * self.vModel['dAlpha']/(4.0 * np.pi)

        # Build the 2d elementary density table
        # Loop through alpha
        for j in range(0, self.vModel['NumAngularGridpoints']):
            curAngle = self.vModel['AngularAlphaGrid'][j]
            # Loop through the radial points along this axis
            for i in range(0, self.vModel['NumRadialGridpoints']):

                curR = self.vModel['FastRadialGrid'][i]
                x = curR * np.sin(curAngle)
                y = curR * np.cos(curAngle)

                # Decide how granular our epsilon should be
                dEpsilonSteps = len(ejectionRadii)
                dEpsilon = (self.vModel['EpsilonMax'] - curAngle)/dEpsilonSteps

                # Maximum radius that contributes to point x,y when there is a a max ejection angle
                if(self.vModel['EpsilonMax'] < np.pi):
                    rLimit = y - (x/np.tan(self.vModel['EpsilonMax']))
                # Set the last element to be rComa or the above limit
                ejectionRadii[dEpsilonSteps-1] = rLimit

                for dE in range(0, dEpsilonSteps-1):  # We already filled out the very last element in the array above
                    ejectionRadii[dE] = y - x/np.tan((dE+1)*dEpsilon + curAngle)

                ejectionRadiiStart = 0
                # Number of slices along the contributing axis for each step
                NumRadialSlices = self.vModelParams['Grid']['SubgridAngularSteps']

                # Loop over radial chunk that contributes to x,y
                for ejectionRadiiEnd in ejectionRadii:

                    # We are slicing up this axis into pieces
                    dr = (ejectionRadiiEnd - ejectionRadiiStart)/NumRadialSlices

                    # Loop over tiny slices along this chunk
                    for m in range(0, NumRadialSlices):

                        # Current distance along contributing axis
                        R = (m + 0.5)*dr + ejectionRadiiStart
                        # This is the distance from the NP axis point to the current point on the ray, squared
                        sepDist = np.sqrt(x * x + (R - y)*(R - y))

                        cosEjection = (y - R)/sepDist
                        sinEjection = x/sepDist

                        # Calculate sqrt(vR^2 - u^2 sin^2 gamma)
                        vFactor = np.sqrt(vf * vf - (vp * vp)*sinEjection**2)

                        # The first (and largest) of the two solutions for the velocity when it arrives
                        vOne = vp*cosEjection + vFactor

                        # Time taken to travel from the dissociation point at v1, reject if the time is too large (and all
                        # fragments have decayed)
                        tFragmentOne = sepDist/vOne
                        if tFragmentOne > timeLimit:
                            continue

                        # This is the total time between parent emission from nucleus and fragment arriving at our point of
                        # interest, which we then use to look up Q at that time in the past
                        tTotalOne = (R/vp) + tFragmentOne

                        # Division by parent velocity makes this production per unit distance for radial integration
                        # q(r, epsilon) given by eq. 32, Festou 1981
                        prodOne = self.productionRateAtTime(tTotalOne)/vp
                        qREpsOne = (vOne*vOne*prodOne)/(vf * np.abs(vOne - vp*cosEjection))

                        # Parent extinction when traveling along to the dissociation site
                        pExtinction = np.e**(-R/(pTotLifetime * vp))
                        # Fragment extinction when traveling at speed v1
                        fExtinctionOne = np.e**(-tFragmentOne/fTotLifetime)

                        # First differential addition to the density integrating along dr, similar to eq. (36) Festou 1981,
                        # due to the first velocity
                        densityOne = (pExtinction * fExtinctionOne * qREpsOne)/(sepDist**2 * vOne)

                        # Add this contribution to the density grid
                        self.vModel['DensityGrid'][i][j] = self.vModel['DensityGrid'][i][j] + densityOne*dr

                        # Check if there is a second solution for the velocity
                        if vf > vp:
                            continue

                        # Compute the contribution from the second solution for v in the same way
                        vTwo = vp*cosEjection - vFactor
                        tFragmentTwo = sepDist/vTwo
                        if tFragmentTwo > timeLimit:
                            continue
                        tTotalTwo = (R/vp) + tFragmentTwo
                        prodTwo = self.productionRateAtTime(tTotalTwo)/vp
                        qREpsTwo = (vTwo * vTwo * prodTwo)/(vf * np.abs(vTwo - vp*cosEjection))
                        fExtinctionTwo = np.e**(-tFragmentTwo/fTotLifetime)
                        densityTwo = (pExtinction * fExtinctionTwo * qREpsTwo)/(vTwo * sepDist**2)
                        self.vModel['DensityGrid'][i][j] = self.vModel['DensityGrid'][i][j] + densityTwo*dr

                    # Next starting radial point is the current end point
                    ejectionRadiiStart = ejectionRadiiEnd

            if(self.vModelParams['Misc']['PrintDensityProgress'] is True):
                progressPercent = (j+1)*100/self.vModel['NumAngularGridpoints']
                print(f'Computing: {progressPercent:3.1f} %', end='\r')

        # Loops automatically over the 2d grid
        self.vModel['DensityGrid'] *= IntegrationFactor
        # phew

        """ Performs angular part of the integration to yield density in km^-3 as a function of radius.
            Assumes spherical symmetry of parent production.

            Fills vModel['RadialDensity'] and vModel['FastRadialDensity'] with and without units respectively
            Fills vModel['rDensInterpolator'] with cubic spline interpolation of the radial density,
                which takes radial coordinate in km and outputs the density at that coord in km^-3
        """

        # Make array to hold our data, no units
        self.vModel['FastRadialDensity'] = np.zeros(self.vModel['NumRadialGridpoints'])

        # loop through grid array
        for i in range(0, self.vModel['NumRadialGridpoints']):
            for j in range(0, self.vModel['NumAngularGridpoints']):
                # Current angle is theta
                theta = self.vModel['AngularAlphaGrid'][j]
                # Integration factors from angular part of integral, similar to eq. (36) Festou 1981
                densityToAdd = 2.0 * np.pi * np.sin(theta) * self.vModel['DensityGrid'][i][j]
                self.vModel['FastRadialDensity'][i] += densityToAdd

        # Tag with proper units
        self.vModel['RadialDensity'] = self.vModel['FastRadialDensity']/(u.km**3)

        # Interpolate this radial density grid with a cubic spline for lookup at non-grid radii, input in km, out in 1/km^3
        self.vModel['rDensInterpolator'] = CubicSpline(self.vModel['FastRadialGrid'], self.vModel['FastRadialDensity'], bc_type='natural')

        # Count up the number of fragments in the grid versus theoretical value
        self.vModel['NumFragmentsTheory'] = self.calcNumFragmentsTheory()
        self.vModel['NumFragmentsFromGrid'] = self.calcNumFragmentsFromGrid()
        self.vModel['FragmentRatio'] = self.vModel['NumFragmentsFromGrid']/self.vModel['NumFragmentsTheory']

    def calcNumFragmentsTheory(self):
        """ Returns the total number of fragment species we expect in the coma theoretically """
        vp = self.par['Velocity']
        vf = self.frag['Velocity']
        pTotLifetime = self.par['TotalLifetime']
        fTotLifetime = self.frag['TotalLifetime']
        pDisLifetime = self.par['DissociativeLifetime']
        pRates = self.comet['ProductionRates']
        pTimes = self.comet['TimeAtProductions']
        tPerm = self.vModel['TimeToPermanentFlowRegime'].to(u.s).value

        mR = self.vModel['MaxRadiusOfGrid'].value
        lastDensityElement = len(self.vModel['FastRadialDensity'])-1

        theoryTot = 0
        for i in range(0, len(pRates)):
            if(pTimes[i] > tPerm):
                t1 = tPerm/pTotLifetime
            else:
                t1 = pTimes[i]/pTotLifetime

            if i != (self.numProductionRates - 1):
                t2 = pTimes[i+1]/pTotLifetime
            else:
                t2 = 0
            theoryTot += pRates[i]*(-np.e**(-t1) + np.e**(-t2))

        theoryTot = theoryTot*(fTotLifetime*pTotLifetime/pDisLifetime) \
            - (np.pi * mR * mR * (vf + vp) * self.vModel['FastRadialDensity'][lastDensityElement])

        return theoryTot

    def calcNumFragmentsFromGrid(self):
        """ Returns the total number of fragment species by integrating the density grid over the volume """
        maxR = self.vModel['MaxRadiusOfGrid'].value

        def volIntegrand(r, rFunc):
            return (rFunc(r) * r**2)

        # Maybe stay away from r = 0 here, but spline type of 'natural' seems to handle origin well
        rInt = integrate.romberg(volIntegrand, 0, maxR, args=(self.vModel['rDensInterpolator'], ),
                                 rtol=0.0001, divmax=20)
        return 4*np.pi*rInt

    def calculateColumnDensity(self, impactParam):
        """ Returns the number of fragment species per km^2 at the given impact parameter, in km """
        rMax = self.vModel['MaxRadiusOfGrid'].value
        if(impactParam > rMax):
            return 0
        ipsq = impactParam**2
        zMax = np.sqrt(rMax**2 - ipsq)

        def columnDensityIntegrand(z):
            return self.vModel['rDensInterpolator'](np.sqrt(z**2 + ipsq))

        # Romberg is significantly slower for impact parameters near the nucleus, and becomes much faster at roughly 60 times
        # the collision sphere radius, after a few tests
        # The results of both were the same to within .1% or better, generally

        if impactParam < (60 * self.vModel['CollisionSphereRadius'].value):
            cDens = (integrate.quad(columnDensityIntegrand, -zMax, zMax, limit=1000))[0]
        else:
            cDens = 2 * integrate.romberg(columnDensityIntegrand, 0, zMax, rtol=0.0001, divmax=20)

        # result is in 1/km^2
        return cDens

    def _interpolateColumnDensity(self):
        """ computes the column density on a grid and produces an interpolation function based on it.
            The interpolator takes input in km from nucleus and return column density in km^-2
        """
        cDensGrid = self.makeLogspaceGrid()
        cdVec = np.vectorize(self.calculateColumnDensity)
        columnDensities = cdVec(cDensGrid)

        self.vModel['ColumnDensity'] = {}
        self.vModel['ColumnDensity']['FastCDGrid'] = cDensGrid
        self.vModel['ColumnDensity']['CDGrid'] = cDensGrid*u.km
        self.vModel['ColumnDensity']['Values'] = columnDensities/(u.km**2)
        # Interpolator gives column density in km^-2
        self.vModel['ColumnDensity']['Interpolator'] = CubicSpline(cDensGrid, columnDensities, bc_type='natural')

    def countInsideAperture(self, aperture):
        """ Takes an aperture and returns the number of fragment species inside the aperture area
            Aperture will be centered at (offset, 0), so horizontal offset only
            Comet centered at 0,0
        """

        totalInAperture = 0

        def ciaIntegrandRectangular(x, y):
            r = np.sqrt(x**2 + y**2)
            return self.vModel['ColumnDensity']['Interpolator'](r)

        def ciaIntegrandCircular(r):
            return self.vModel['ColumnDensity']['Interpolator'](r) * r

        if(isinstance(aperture, sba.CircularAperture)):
            apRadius = aperture.as_length(self.vModel['MaxRadiusOfGrid']).radius.to_value('km')
            totalInAperture = integrate.romberg(ciaIntegrandCircular, 0, apRadius, divmax=20)
            totalInAperture *= 2 * np.pi
        elif(isinstance(aperture, sba.AnnularAperture)):
            innerR = aperture.shape.to_value('km')[0]
            outerR = aperture.shape.to_value('km')[1]
            totOuter = self.countInsideAperture(sba.CircularAperture(outerR * u.km))
            totInner = self.countInsideAperture(sba.CircularAperture(innerR * u.km))
            totalInAperture = totOuter - totInner
        elif(isinstance(aperture, sba.RectangularAperture)):
            leftX = -aperture.shape.to_value('km')[0]/2
            rightX = aperture.shape.to_value('km')[0]/2
            topY = aperture.shape.to_value('km')[1]/2
            bottomY = -aperture.shape.to_value('km')[1]/2

            # integrate.dblquad gives poor results, this method recovers almost all fragments with a very large aperture
            nYSteps = 100
            dY = (topY - bottomY)/nYSteps
            for y in np.linspace(bottomY, topY, nYSteps):
                totalInAperture += integrate.romberg(ciaIntegrandRectangular, leftX, rightX, args=(y,), divmax=20)*dY

        return totalInAperture
