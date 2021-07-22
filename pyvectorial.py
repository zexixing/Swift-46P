#!/usr/bin/env python3

import copy
import numpy
import os
import sys
import yaml

import numpy as np
import astropy.units as u
from astropy.visualization import quantity_support
import matplotlib.pyplot as plt
import sbpy.activity as sba

from argparse import ArgumentParser

from vmodel import VectorialModel

__author__ = 'Shawn Oset, Lauren Lyons'
__version__ = '0.0'

solarbluecol = np.array([38, 139, 220]) / 255.
solarblue = (solarbluecol[0], solarbluecol[1], solarbluecol[2], 1)
solargreencol = np.array([133, 153, 0]) / 255.
solargreen = (solargreencol[0], solargreencol[1], solargreencol[2], 1)
solarblackcol = np.array([0, 43, 54]) / 255.
solarblack = (solarblackcol[0], solarblackcol[1], solarblackcol[2], 1)
solarwhitecol = np.array([238, 232, 213]) / 255.
solarwhite = (solarblackcol[0], solarblackcol[1], solarblackcol[2], 1)


def readParametersYAML(filepath):
    """Read the YAML file with all of the input parameters in it"""
    with open(filepath, 'r') as stream:
        try:
            paramYAMLData = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return paramYAMLData


def showRadialPlots(coma, rUnits, volUnits, fragName):
    """ Show the radial density of the fragment species """

    xMin_logplot = 2
    xMax_logplot = 8

    xMin_linear = 0 * u.km
    xMax_linear = 2000 * u.km

    linInterpX = np.linspace(xMin_linear.value, xMax_linear.value, num=200)
    linInterpY = coma.vModel['rDensInterpolator'](linInterpX)/(u.km**3)
    linInterpX *= u.km
    linInterpX.to(rUnits)

    logInterpX = np.logspace(xMin_logplot, xMax_logplot, num=200)
    logInterpY = coma.vModel['rDensInterpolator'](logInterpX)/(u.km**3)
    logInterpX *= u.km
    logInterpX.to(rUnits)

    plt.style.use('Solarize_Light2')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    ax1.set(xlabel=f'Distance from nucleus, {rUnits.to_string()}')
    ax1.set(ylabel=f"Fragment density, {volUnits.unit.to_string()}")
    ax2.set(xlabel=f'Distance from nucleus, {rUnits.to_string()}')
    ax2.set(ylabel=f"Fragment density, {volUnits.unit.to_string()}")
    fig.suptitle(f"Calculated radial density of {fragName}")

    ax1.set_xlim([xMin_linear, xMax_linear])
    ax1.plot(linInterpX, linInterpY, color="red",  linewidth=1.5, linestyle="-", label="cubic spline")
    ax1.plot(coma.vModel['RadialGrid'], coma.vModel['RadialDensity'].to(volUnits), 'bo', label="model")
    ax1.plot(coma.vModel['RadialGrid'], coma.vModel['RadialDensity'].to(volUnits), 'g--', label="linear interpolation")

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.loglog(coma.vModel['FastRadialGrid'], coma.vModel['RadialDensity'].to(volUnits), 'bo', label="model")
    ax2.loglog(coma.vModel['FastRadialGrid'], coma.vModel['RadialDensity'].to(volUnits), 'g--', label="linear interpolation")
    ax2.loglog(logInterpX, logInterpY, color="red",  linewidth=1.5, linestyle="-", label="cubic spline")

    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0.1)

    # Mark the beginning of the collision sphere
    ax1.axvline(x=coma.vModel['CollisionSphereRadius'], color=solarblue)
    ax2.axvline(x=coma.vModel['CollisionSphereRadius'], color=solarblue)

    # Mark the collision sphere
    plt.text(coma.vModel['CollisionSphereRadius']*2, linInterpY[0]*2, 'Collision Sphere Edge', color=solarblue)

    plt.legend(loc='upper right', frameon=False)
    plt.show()


def showColumnDensityPlot(coma, rUnits, cdUnits, fragName):
    """ Show the radial density of the fragment species """

    xMin_logplot = 2
    xMax_logplot = 8

    xMin_linear = 0 * u.km
    xMax_linear = 2000 * u.km

    linInterpX = np.linspace(xMin_linear.value, xMax_linear.value, num=200)
    linInterpY = coma.vModel['ColumnDensity']['Interpolator'](linInterpX)/(u.km**2)
    linInterpX *= u.km
    linInterpX.to(rUnits)

    logInterpX = np.logspace(xMin_logplot, xMax_logplot, num=200)
    logInterpY = coma.vModel['ColumnDensity']['Interpolator'](logInterpX)/(u.km**2)
    logInterpX *= u.km
    logInterpX.to(rUnits)

    plt.style.use('Solarize_Light2')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    ax1.set(xlabel=f'Distance from nucleus, {rUnits.to_string()}')
    ax1.set(ylabel=f"Fragment column density, {cdUnits.unit.to_string()}")
    ax2.set(xlabel=f'Distance from nucleus, {rUnits.to_string()}')
    ax2.set(ylabel=f"Fragment column density, {cdUnits.unit.to_string()}")
    fig.suptitle(f"Calculated column density of {fragName}")

    ax1.set_xlim([xMin_linear, xMax_linear])
    ax1.plot(linInterpX, linInterpY, color="red",  linewidth=2.5, linestyle="-", label="cubic spline")
    ax1.plot(coma.vModel['ColumnDensity']['CDGrid'], coma.vModel['ColumnDensity']['Values'], 'bo', label="model")
    ax1.plot(coma.vModel['ColumnDensity']['CDGrid'], coma.vModel['ColumnDensity']['Values'], 'g--', label="linear interpolation")

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.loglog(coma.vModel['ColumnDensity']['CDGrid'], coma.vModel['ColumnDensity']['Values'], 'bo', label="model")
    ax2.loglog(coma.vModel['ColumnDensity']['CDGrid'], coma.vModel['ColumnDensity']['Values'], 'g--',
               label="linear interpolation", linewidth=0.5)
    ax2.loglog(logInterpX, logInterpY, color="red",  linewidth=0.5, linestyle="-", label="cubic spline")

    ax1.set_ylim(bottom=0)

    ax2.set_xlim(right=coma.vModel['MaxRadiusOfGrid'])

    # Mark the beginning of the collision sphere
    ax1.axvline(x=coma.vModel['CollisionSphereRadius'], color=solarblue)
    ax2.axvline(x=coma.vModel['CollisionSphereRadius'], color=solarblue)

    # Only plot as far as the maximum radius of our grid on log-log plot
    ax2.axvline(x=coma.vModel['MaxRadiusOfGrid'])

    # Mark the collision sphere
    plt.text(coma.vModel['CollisionSphereRadius']*1.1, linInterpY[0]/10, 'Collision Sphere Edge', color=solarblue)

    plt.legend(loc='upper right', frameon=False)
    plt.show()


def showColumnDensity3D(coma, xMin, xMax, yMin, yMax, gridStepX, gridStepY, rUnits, cdUnits, fragName):
    """ 3D plot of column density """

    x = np.linspace(xMin.to(u.km).value, xMax.to(u.km).value, gridStepX)
    y = np.linspace(yMin.to(u.km).value, yMax.to(u.km).value, gridStepY)
    xv, yv = np.meshgrid(x, y)
    z = coma.vModel['ColumnDensity']['Interpolator'](np.sqrt(xv**2 + yv**2))
    # Interpolator spits out km^-2
    fz = (z/u.km**2).to(cdUnits)

    xu = np.linspace(xMin.to(rUnits), xMax.to(rUnits), gridStepX)
    yu = np.linspace(yMin.to(rUnits), yMax.to(rUnits), gridStepY)
    xvu, yvu = np.meshgrid(xu, yu)

    plt.style.use('Solarize_Light2')
    plt.style.use('dark_background')
    plt.rcParams['grid.color'] = "black"

    fig = plt.figure(figsize=(20, 20))
    ax = plt.axes(projection='3d')
    # ax.grid(False)
    surf = ax.plot_surface(xvu, yvu, fz, cmap='inferno', vmin=0, edgecolor='none')

    plt.gca().set_zlim(bottom=0)

    ax.set_xlabel(f'Distance, ({rUnits.to_string()})')
    ax.set_ylabel(f'Distance, ({rUnits.to_string()})')
    ax.set_zlabel(f"Column density, {cdUnits.unit.to_string()}")
    plt.title(f"Calculated column density of {fragName}")

    ax.w_xaxis.set_pane_color(solargreen)
    ax.w_yaxis.set_pane_color(solarblue)
    ax.w_zaxis.set_pane_color(solarblack)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(90, 90)
    plt.show()


def main(rh, delta):
    '''
    # Parse command-line arguments
    parser = ArgumentParser(
        usage='%(prog)s [options] [inputfile]',
        description=__doc__,
        prog=os.path.basename(sys.argv[0])
    )
    parser.add_argument('--version', action='version', version=__version__)
    parser.add_argument('--verbose', '-v', action='count', default=0,
                        help='increase verbosity level')
    parser.add_argument(
            'parameterfile', nargs=1, help='YAML file with observation and molecule data in it'
            )  # the nargs=? specifies 0 or 1 arguments: it is optional
    args = parser.parse_args()
    '''

    #print(f'Loading input from {args.parameterfile[0]} ...')
    # Read in our stuff
    #inputYAML = readParametersYAML(args.parameterfile[0])
    inputYAML = readParametersYAML('/Users/zexixing/Research/wvm/parameters.yaml')

    # Check if the length ProductionRates TimeAtProductions is the same, otherwise we're in trouble
    if(len(inputYAML['Comet']['ProductionRates']) != len(inputYAML['Comet']['TimeAtProductions'])):
        print("Mismatched lengths for ProductionRates and TimeAtProductions!  Exiting.")
        return 1

    # astropy units/quantities support in plots
    quantity_support()

    # save an unaltered copy
    vmInput = copy.deepcopy(inputYAML)

    # apply units to data loaded from file
    vmInput['Comet']['TimeAtProductions'] *= u.day
    vmInput['Comet']['GeocentricDistance'] *= u.AU
    vmInput['Comet']['HeliocentricDistance'] *= u.AU #~TODO:

    vmInput['ParentSpecies']['Velocity'] *= u.km/u.s
    vmInput['ParentSpecies']['TotalLifetime'] *= u.s
    vmInput['ParentSpecies']['DissociativeLifetime'] *= u.s
    vmInput['ParentSpecies']['Sigma'] *= u.cm**2

    vmInput['FragmentSpecies']['Velocity'] *= u.km/u.s
    vmInput['FragmentSpecies']['TotalLifetime'] *= u.s

    # temp
    vmInput['Comet']['GeocentricDistance'] = delta*u.AU
    vmInput['Comet']['HeliocentricDistance'] = rh*u.AU

    # Adjust some things for heliocentric distance
    rs2 = vmInput['Comet']['HeliocentricDistance'].value**2
    vmInput['ParentSpecies']['TotalLifetime'] *= rs2
    vmInput['ParentSpecies']['DissociativeLifetime'] *= rs2
    vmInput['FragmentSpecies']['TotalLifetime'] *= rs2
    # Cochran and Schleicher, 1993
    vmInput['ParentSpecies']['Velocity'] /= np.sqrt(vmInput['Comet']['HeliocentricDistance'].value)

    # Do the calculations
    print("Calculating density using vectorial model ...")
    coma = VectorialModel(vmInput)
    '''
    print("\n\nRadius (km) vs Fragment density (1/cm3)\n---------------------------------------")
    for i in range(0, coma.vModel['NumRadialGridpoints']):
        print(f"{coma.vModel['RadialGrid'][i]:10.1f} : {coma.vModel['RadialDensity'][i].to(1/(u.cm**3)):8.4f}")

    print("\nRadius (km) vs Column density (1/cm2)\n-------------------------------------")
    cds = list(zip(coma.vModel['ColumnDensity']['CDGrid'], coma.vModel['ColumnDensity']['Values']))
    for pair in cds:
        print(f'{pair[0]:7.0f} :\t{pair[1].to(1/(u.cm*u.cm)):5.3e}')

    fragmentTheory = coma.vModel['NumFragmentsTheory']
    fragmentGrid = coma.vModel['NumFragmentsFromGrid']
    print("Theoretical total number of fragments in coma: ", fragmentTheory)
    print("Total number of fragments from density grid integration: ", fragmentGrid)

    # Total counts inside apertures
    # Set aperture to entire comet to see if we get all of the fragments as an answer
    ap1 = sba.RectangularAperture((coma.vModel['MaxRadiusOfGrid'].value, coma.vModel['MaxRadiusOfGrid'].value) * u.km)
    print("Percent of total fragments recovered by integrating column density over")
    print("\tLarge rectangular aperture: ",
          coma.countInsideAperture(ap1)*100/fragmentTheory)

    # Large circular aperture
    ap2 = sba.CircularAperture((coma.vModel['MaxRadiusOfGrid'].value) * u.km)
    print("\tLarge circular aperture: ", coma.countInsideAperture(ap2)*100/fragmentTheory)

    # Try out annular
    ap3 = sba.AnnularAperture([500000, coma.vModel['MaxRadiusOfGrid'].value] * u.km)
    print("\tAnnular aperture, inner radius 500000 km, outer radius of entire grid:\n\t",
          coma.countInsideAperture(ap3)*100/fragmentTheory)

    # Show some plots
    fragName = vmInput['FragmentSpecies']['Name']

    # Volume and column density plots
    #showRadialPlots(coma, u.km, 1/u.cm**3, fragName)
    #showColumnDensityPlot(coma, u.km, 1/u.cm**3, fragName)

    # column density with nucleus in the corner of the plot
    #showColumnDensity3D(coma, -100000*u.km, 10000*u.km, -100000*u.km, 10000*u.km, 1000, 100, u.km, 1/u.cm**2, fragName)
    # column density with nucleus centered
    #showColumnDensity3D(coma, -100000*u.km, 100000*u.km, -100000*u.km, 100000*u.km, 1000, 1000, u.km, 1/u.cm**2, fragName)
    '''
    d = coma.vModel['ColumnDensity']['CDGrid']/(u.km)
    cd = coma.vModel['ColumnDensity']['Values'].to(1/(u.cm*u.cm))*(u.cm*u.cm)
    '''
    from scipy import optimize, interpolate
    f = interpolate.interp1d(d,cd,fill_value=(cd[0],cd[-1]),bounds_error=False)
    step = 1 #km
    def fov_circle(r):
        s = np.pi*(r*1e5)**2
        if np.ceil(d[0]) < r:
            x = np.arange(np.ceil(d[0]),r+step,step)
            x = (x[:-1]+x[1:])/2
            y = f(x)
            num = np.sum((2*np.pi*x*1e5)*(step*1e5)*y)
            num += np.pi*(np.ceil(d[0])*1e5)**2*cd[0]
        else:
            num = s*cd[0]
        return num, num/s
    def fov_regt(a,b):
        num1,den = fov_circle(a/2)
        x = np.arange(a/2,b/2+step,step)
        x = (x[:-1]+x[1:])/2
        y = f(x)
        theta = np.arccos(a/(2*x))
        num2 = np.sum((2*np.pi*x-4*x*theta)*1e5*(step*1e5)*y)
        c = np.sqrt(a**2+b**2)/2
        x = np.arange(b/2,c+step,step)
        x = (x[:-1]+x[1:])/2
        y = f(x)
        alpha = 4*(np.pi/2-np.arccos(b/(2*x))-np.arccos(a/(2*x)))
        num3 = np.sum(x*alpha*1e5*step*1e5*y)
        num = num1+num2+num3
        s = (a*1e5)*(b*1e5)
        return num, num/s
    #r = 500
    #num, den = fov_circle(r)
    #num, den = fov_regt(80,200)
    #print(num,den)
    #print(num/s)
    '''
    return d, cd


#if __name__ == '__main__':
#    sys.exit(main())

def pyvm(rh,delta):
    d,cd = main(rh, delta)
    return d, cd