#!/usr/bin/python3

# =============================================================
# Plot logged data from the APL Iver2 vehicle
# =============================================================

# System imports
import matplotlib.pyplot as plot
import sys
import numpy as np
import scipy.stats

# =============================================================
# Function to detect and correct outliers in a dataset
# =============================================================
def fillOutliers( rawData ):
    # Determine good entries (abs zscore < 3)
    zScores = scipy.stats.zscore(rawData)
    good = np.where(np.abs(zScores)<3)
    # Generate mean of this
    goodMean = np.mean(rawData[good])

    # Extract bad entries
    mask = np.ones(rawData.size,dtype=bool)
    mask[good] = False
    bad = np.where(mask)
    # Set bad entries to good mean
    filteredData = rawData
    filteredData[bad] = goodMean

    # Return the result
    return filteredData

# =============================================================
# App Start
# =============================================================

# Must supply filename as an arg
nArgs = len(sys.argv)
if nArgs < 2 :
    print(' - Not enough args')
    sys.exit()
dataFile = sys.argv[1];

# Load the file
print(' - Loading file: %s' % dataFile)
data = np.loadtxt(dataFile)

# Extract the data
print(' - Extracting data')
elapsedTime = data[:,0] # sec
lat = data[:,1] # deg
lon = data[:,2] # deg
depth = data[:,3] # m
roll = data[:,4] # deg
pitch = data[:,5] # deg
heading = data[:,6] # deg
targetHeading = data[:,7] # deg
targetSpeed = data[:,8] # m/s  (what is this?)
targetDepth = data[:,9] # m
vX = data[:,10] # m/s
vY = data[:,11] # m/s
vZ = data[:,12] # m/s
upperRudder = data[:,13] # deg
lowerRudder = data[:,14] # deg
portElevator = data[:,15] # deg
stbdElevator = data[:,16] # deg
propPower = data[:,17] # percent
propCurrent = data[:,18] # amps
propSpeed = data[:,19] # RPM (motor mechanical)
rudderMode = data[:,20] # 0=manual,1=headingControl
elevMode = data[:,21]   # 0=manual,1=depthControl
propMode = data[:,22]   # 0=manual,1=speedControl
leakDetect = data[:,23] # 0/1 = no leak/leak
battTemp = data[:,24] # battery temp, degC
battVolt = data[:,25] # battery voltage, V
battCurr = data[:,26] # battery current, A
battSOC = data[:,27]  # state of charge
totalCurrent = data[:,28]
temp1 = data[:,29]
temp2 = data[:,30]
temp3 = data[:,31]

# Generate total velocities
print(' - Generating total velocities')
vTotal = np.linalg.norm(np.array([vX,vY,vZ]),axis=0)

# Filter battery data for outliers
print(' - Filtering outliers')
battVolt = fillOutliers(battVolt)
battTemp = fillOutliers(battTemp)
battCurr = fillOutliers(battCurr)
battSOC = fillOutliers(battSOC)
# Likewise for target values
targetDepth = fillOutliers(targetDepth)
targetHeading = fillOutliers(targetHeading)

print(' - Plotting')

# ---------------------------------------------------------------
# Depth & speed
# ---------------------------------------------------------------
dsFig,dsPlots = plot.subplots(2,1)
dsFig.suptitle('Depth & Speed', fontsize=14,fontweight='bold')
# Depth
depthPlot = dsPlots[0]
depthPlot.plot(elapsedTime, depth, color='r', label='actual')
depthPlot.plot(elapsedTime, targetDepth, color='b', label='target')
depthPlot.legend(loc="upper right")
depthPlot.set_title('Depth')
depthPlot.set(xlabel='sec',ylabel='meters')
depthPlot.grid(which='both',axis='both')
depthPlot.invert_yaxis()
# Speed
speedPlot = dsPlots[1]
speedPlot.plot(elapsedTime, vX, color='b',label='X')
speedPlot.plot(elapsedTime, vY, color='r',label='Y')
speedPlot.plot(elapsedTime, vZ, color='g',label='Z')
speedPlot.plot(elapsedTime, vTotal, color='k',label='Total')
speedPlot.legend(loc="upper right")
speedPlot.set_title('Speed')
speedPlot.set(xlabel='sec',ylabel='m/s')
speedPlot.grid(which='both',axis='both')
# Space out the plots a bit in height
plot.subplots_adjust(hspace=0.5)

# ---------------------------------------------------------------
# Attitude
# ---------------------------------------------------------------
attFig,attPlots = plot.subplots(2,1)
attFig.suptitle('Attitude', fontsize=14,fontweight='bold')
# Pitch & Roll
pitchRollPlot = attPlots[0]
pitchRollPlot.plot(elapsedTime, pitch, color='r',label='roll')
pitchRollPlot.plot(elapsedTime, roll, color='b',label='pitch')
pitchRollPlot.legend(loc="upper right")
pitchRollPlot.set_title('Pitch & Roll')
pitchRollPlot.set(xlabel='sec',ylabel='deg')
pitchRollPlot.grid(which='both',axis='both')
# Heading
hdgPlot = attPlots[1]
hdgPlot.plot(elapsedTime, heading, color='r', label='actual')
hdgPlot.plot(elapsedTime, targetHeading, color='b', label='target')
hdgPlot.legend(loc="upper right")
hdgPlot.set_title('Heading')
hdgPlot.set(xlabel='sec',ylabel='deg')
hdgPlot.grid(which='both',axis='both')
plot.subplots_adjust(hspace=0.5)

# ---------------------------------------------------------------
# Fins
# ---------------------------------------------------------------
finsFig,finsPlots = plot.subplots(2,1)
finsFig.suptitle('Fins', fontsize=14,fontweight='bold')
# Rudders
ruddersPlot = finsPlots[0]
ruddersPlot.plot(elapsedTime, upperRudder, color='b',label='upper')
ruddersPlot.plot(elapsedTime, lowerRudder, color='r',label='lower')
ruddersPlot.legend(loc="upper right")
ruddersPlot.set_title('Rudders')
ruddersPlot.set(xlabel='sec',ylabel='deg')
ruddersPlot.grid(which='both',axis='both')
# Elevators
elevatorsPlot = finsPlots[1]
elevatorsPlot.plot(elapsedTime, portElevator, color='b',label='port')
elevatorsPlot.plot(elapsedTime, stbdElevator, color='r',label='stbd')
elevatorsPlot.legend(loc="upper right")
elevatorsPlot.set_title('Elevators')
elevatorsPlot.set(xlabel='sec',ylabel='deg')
elevatorsPlot.grid(which='both',axis='both')
plot.subplots_adjust(hspace=0.5)

# ---------------------------------------------------------------
# Prop
# ---------------------------------------------------------------
propFig,propPlots = plot.subplots(3,1)
propFig.suptitle('Prop', fontsize=14,fontweight='bold')
# RPM
rpmPlot = propPlots[0]
rpmPlot.plot(elapsedTime, propSpeed)
rpmPlot.set_title('Prop Speed (Motor RPM)')
rpmPlot.set(xlabel='sec',ylabel='RPM')
rpmPlot.grid(which='both',axis='both')
# Current
propCurrentPlot = propPlots[1]
propCurrentPlot.plot(elapsedTime, propCurrent)
propCurrentPlot.set_title('Prop Current')
propCurrentPlot.set(xlabel='sec',ylabel='amps')
propCurrentPlot.grid(which='both',axis='both')
# Power (throttle level)
powerPlot = propPlots[2]
powerPlot.plot(elapsedTime, propPower)
powerPlot.set_title('Prop Power (Throttle Level)')
powerPlot.set(xlabel='sec',ylabel='percent')
powerPlot.grid(which='both',axis='both')
plot.subplots_adjust(hspace=1)

# ---------------------------------------------------------------
# Batteries
# ---------------------------------------------------------------
battFig,battPlots = plot.subplots(2,2)
battFig.suptitle('Batteries', fontsize=14,fontweight='bold')
# Temperature
tempPlot = battPlots[0,0]
tempPlot.plot(elapsedTime, battTemp)
tempPlot.set_title('Battery Temperature')
tempPlot.set(xlabel='sec',ylabel='degC')
tempPlot.grid(which='both',axis='both')
# Voltage
voltPlot = battPlots[0,1]
voltPlot.plot(elapsedTime, battVolt)
voltPlot.set_title('Battery Voltage')
voltPlot.set(xlabel='sec',ylabel='volts')
voltPlot.grid(which='both',axis='both')
# Current
battCurrentPlot = battPlots[1,0]
battCurrentPlot.plot(elapsedTime, battCurr)
battCurrentPlot.set_title('Battery Current')
battCurrentPlot.set(xlabel='sec',ylabel='amps')
battCurrentPlot.grid(which='both',axis='both')
# State of Charge
socPlot = battPlots[1,1]
socPlot.plot(elapsedTime, battSOC)
socPlot.set_title('Battery State of Charge')
socPlot.set(xlabel='sec',ylabel='percent')
socPlot.grid(which='both',axis='both')
plot.subplots_adjust(hspace=0.5,wspace=0.5)  # space in height & width

# Share the x-axes across all plots.
# This means zooming in time in one zooms them all to the same timespan
socPlot.get_shared_x_axes().join(socPlot,battCurrentPlot,voltPlot,tempPlot,
                                 powerPlot,propCurrentPlot,rpmPlot,
                                 elevatorsPlot, ruddersPlot,
                                 hdgPlot,pitchRollPlot,
                                 speedPlot,depthPlot)

# Put 'em up
plot.show()

print('\n === DONE === \n')
