
# Library for plotting data from APL Iver2 vehicle
# Can also be used as a script with
# python3 plot_iver.py [data.log]
# ^ spits out a folder full of plots
# 2022 Pete Brodsky, Joe Sammartino, Tim Player
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from typing import List, Tuple
from scipy.spatial.transform import Rotation as R

def prepare_data(log_file: str) -> pd.DataFrame:
    """Load, clean, and preprocess Pandas DataFrame from log file."""

    df = log_to_df(log_file)

    # Generate total velocities
    # print(' - Generating total velocities')
    df['vTotal'] = np.linalg.norm(np.array([df.vX, df.vY, df.vZ]), axis=0)

    # Filter battery data for outliers
    # print(' - Filtering outliers')
    df.battVolt = fillOutliers(df.battVolt.to_numpy())
    df.battTemp = fillOutliers(df.battTemp.to_numpy())
    df.battCurr = fillOutliers(df.battCurr.to_numpy())
    df.battSOC = fillOutliers(df.battSOC.to_numpy())

    # Likewise for target values
    df.targetDepth = fillOutliers(df.targetDepth.to_numpy())
    df.targetHeading = fillOutliers(df.targetHeading.to_numpy())

    return df

def log_to_df(log_file: str) -> pd.DataFrame:

    names = [ # these should actually live in the CSV
        "elapsedTime", # sec
        "lat", # deg
        "lon", # deg
        "depth", # m
        "roll", # deg
        "pitch", # deg
        "heading", # deg
        "targetHeading", # deg
        "targetSpeed", # m/s  (what is this?)
        "targetDepth", # m
        "vX", # m/s
        "vY", # m/s
        "vZ", # m/s
        "upperRudder", # deg
        "lowerRudder", # deg
        "portElevator", # deg
        "stbdElevator", # deg
        "propPower", # percent
        "propCurrent", # amps
        "propSpeed", # RPM (motor mechanical)
        "rudderMode", # 0=manual,1=headingControl
        "elevMode", # 0=manual,1=depthControl
        "propMode", # 0=manual,1=speedControl
        "leakDetect", # 0/1 = no leak/leak
        "battTemp", # battery temp, degC
        "battVolt", # battery voltage, V
        "battCurr", # battery current, A
        "battSOC",  # state of charge
        "totalCurrent",
        "temp1",
        "temp2",
        "temp3"
    ]

    # Read from a space-separated CSV with no header and bad lines
    df = pd.read_csv(log_file, header=None, sep=' ', on_bad_lines='skip')

    # There are 35 columns in the DataFrame, but we only know the names of 32 of
    # them.
    df.columns = pd.Index(
        names + 
        df.columns[len(names):].map(str).to_list()) # cast to str

    df = df.interpolate() # fill in NaNs in the middle of data 

    return df # may contain NaN, at start of vX, vY, vZ

def fillOutliers(rawData: np.ndarray) -> np.ndarray:

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

def plot_inputs_outputs(df: pd.DataFrame) -> None:
    """ Plot inputs and outputs.
    Left column: inputs. Rudders, fins, thrust.
    Right column: outputs. Pitch/roll, heading, speed.
    """
    fig, axs = plt.subplots(3, 2, figsize=(10, 5))

    ruddersPlot = axs[0, 0]
    ruddersPlot.plot(df.elapsedTime, df.upperRudder, color='b',label='upper')
    ruddersPlot.plot(df.elapsedTime, df.lowerRudder, color='r',label='lower')
    ruddersPlot.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ruddersPlot.set_title('Rudders')
    ruddersPlot.set(xlabel='sec',ylabel='deg')
    ruddersPlot.grid(which='both',axis='both')

    elevatorsPlot = axs[1, 0]
    elevatorsPlot.plot(df.elapsedTime, df.portElevator, color='b',label='port')
    elevatorsPlot.plot(df.elapsedTime, df.stbdElevator, color='r',label='stbd')
    elevatorsPlot.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    elevatorsPlot.set_title('Elevators')
    elevatorsPlot.set(xlabel='sec',ylabel='deg')
    elevatorsPlot.grid(which='both',axis='both')

    # RPM
    rpmPlot = axs[2, 0]
    rpmPlot.plot(df.elapsedTime, df.propSpeed)
    rpmPlot.set_ylim([0, 20_000])
    rpmPlot.set_title('Prop Speed (Motor RPM)')
    rpmPlot.set(xlabel='sec',ylabel='RPM')
    rpmPlot.grid(which='both',axis='both')

    # Heading
    hdgPlot = axs[0, 1]
    hdgPlot.plot(df.elapsedTime, df.heading, color='r', label='actual')
    hdgPlot.plot(df.elapsedTime, df.targetHeading, color='b', label='target')
    hdgPlot.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    hdgPlot.set_title('Heading')
    hdgPlot.set(xlabel='sec',ylabel='deg')
    hdgPlot.grid(which='both',axis='both')

    # Pitch & Roll
    pitchRollPlot = axs[1, 1]
    pitchRollPlot.plot(df.elapsedTime, df.pitch, color='r',label='roll')
    pitchRollPlot.plot(df.elapsedTime, df.roll, color='b',label='pitch')
    pitchRollPlot.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    pitchRollPlot.set_title('Pitch & Roll')
    pitchRollPlot.set(xlabel='sec',ylabel='deg')
    pitchRollPlot.grid(which='both',axis='both')

    # Speed
    speedPlot = axs[2, 1]
    speedPlot.plot(df.elapsedTime, df.vX, color='b',label='X')
    speedPlot.plot(df.elapsedTime, df.vY, color='r',label='Y')
    speedPlot.plot(df.elapsedTime, df.vZ, color='g',label='Z')
    speedPlot.plot(df.elapsedTime, df.vTotal, color='k',label='Total')
    speedPlot.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    speedPlot.set_title('Speed')
    speedPlot.set(xlabel='sec',ylabel='m/s')
    speedPlot.grid(which='both',axis='both')

    fig.subplots_adjust(hspace=0.5)
    fig.tight_layout()

    return fig, axs

def plot_data(df: pd.DataFrame) -> None:

    # ---------------------------------------------------------------
    # Depth & speed
    # ---------------------------------------------------------------
    dsFig,dsPlots = plt.subplots(2,1)
    dsFig.suptitle('Depth & Speed', fontsize=14,fontweight='bold')
    # Depth
    depthPlot = dsPlots[0]
    depthPlot.plot(df.elapsedTime, df.depth, color='r', label='actual')
    depthPlot.plot(df.elapsedTime, df.targetDepth, color='b', label='target')
    depthPlot.legend(loc="upper right")
    depthPlot.set_title('Depth')
    depthPlot.set(xlabel='sec',ylabel='meters')
    depthPlot.grid(which='both',axis='both')
    depthPlot.invert_yaxis()
    # Speed
    speedPlot = dsPlots[1]
    speedPlot.plot(df.elapsedTime, df.vX, color='b',label='X')
    speedPlot.plot(df.elapsedTime, df.vY, color='r',label='Y')
    speedPlot.plot(df.elapsedTime, df.vZ, color='g',label='Z')
    speedPlot.plot(df.elapsedTime, df.vTotal, color='k',label='Total')
    speedPlot.legend(loc="upper right")
    speedPlot.set_title('Speed')
    speedPlot.set(xlabel='sec',ylabel='m/s')
    speedPlot.grid(which='both',axis='both')
    # Space out the plots a bit in height
    plt.subplots_adjust(hspace=0.5)

    # ---------------------------------------------------------------
    # Attitude
    # ---------------------------------------------------------------
    attFig,attPlots = plt.subplots(2,1)
    attFig.suptitle('Attitude', fontsize=14,fontweight='bold')
    # Pitch & Roll
    pitchRollPlot = attPlots[0]
    pitchRollPlot.plot(df.elapsedTime, df.pitch, color='r',label='roll')
    pitchRollPlot.plot(df.elapsedTime, df.roll, color='b',label='pitch')
    pitchRollPlot.legend(loc="upper right")
    pitchRollPlot.set_title('Pitch & Roll')
    pitchRollPlot.set(xlabel='sec',ylabel='deg')
    pitchRollPlot.grid(which='both',axis='both')
    # Heading
    hdgPlot = attPlots[1]
    hdgPlot.plot(df.elapsedTime, df.heading, color='r', label='actual')
    hdgPlot.plot(df.elapsedTime, df.targetHeading, color='b', label='target')
    hdgPlot.legend(loc="upper right")
    hdgPlot.set_title('Heading')
    hdgPlot.set(xlabel='sec',ylabel='deg')
    hdgPlot.grid(which='both',axis='both')
    plt.subplots_adjust(hspace=0.5)

    # ---------------------------------------------------------------
    # Fins
    # ---------------------------------------------------------------
    finsFig,finsPlots = plt.subplots(2,1)
    finsFig.suptitle('Fins', fontsize=14,fontweight='bold')
    # Rudders
    ruddersPlot = finsPlots[0]
    ruddersPlot.plot(df.elapsedTime, df.upperRudder, color='b',label='upper')
    ruddersPlot.plot(df.elapsedTime, df.lowerRudder, color='r',label='lower')
    ruddersPlot.legend(loc="upper right")
    ruddersPlot.set_title('Rudders')
    ruddersPlot.set(xlabel='sec',ylabel='deg')
    ruddersPlot.grid(which='both',axis='both')
    # Elevators
    elevatorsPlot = finsPlots[1]
    elevatorsPlot.plot(df.elapsedTime, df.portElevator, color='b',label='port')
    elevatorsPlot.plot(df.elapsedTime, df.stbdElevator, color='r',label='stbd')
    elevatorsPlot.legend(loc="upper right")
    elevatorsPlot.set_title('Elevators')
    elevatorsPlot.set(xlabel='sec',ylabel='deg')
    elevatorsPlot.grid(which='both',axis='both')
    plt.subplots_adjust(hspace=0.5)

    # ---------------------------------------------------------------
    # Prop
    # ---------------------------------------------------------------
    propFig,propPlots = plt.subplots(3,1)
    propFig.suptitle('Prop', fontsize=14,fontweight='bold')
    # RPM
    rpmPlot = propPlots[0]
    rpmPlot.plot(df.elapsedTime, df.propSpeed)
    rpmPlot.set_title('Prop Speed (Motor RPM)')
    rpmPlot.set(xlabel='sec',ylabel='RPM')
    rpmPlot.grid(which='both',axis='both')
    # Current
    propCurrentPlot = propPlots[1]
    propCurrentPlot.plot(df.elapsedTime, df.propCurrent)
    propCurrentPlot.set_title('Prop Current')
    propCurrentPlot.set(xlabel='sec',ylabel='amps')
    propCurrentPlot.grid(which='both',axis='both')
    # Power (throttle level)
    powerPlot = propPlots[2]
    powerPlot.plot(df.elapsedTime, df.propPower)
    powerPlot.set_title('Prop Power (Throttle Level)')
    powerPlot.set(xlabel='sec',ylabel='percent')
    powerPlot.grid(which='both',axis='both')
    plt.subplots_adjust(hspace=1)

    # ---------------------------------------------------------------
    # Batteries
    # ---------------------------------------------------------------
    battFig,battPlots = plt.subplots(2,2)
    battFig.suptitle('Batteries', fontsize=14,fontweight='bold')
    # Temperature
    tempPlot = battPlots[0,0]
    tempPlot.plot(df.elapsedTime, df.battTemp)
    tempPlot.set_title('Battery Temperature')
    tempPlot.set(xlabel='sec',ylabel='degC')
    tempPlot.grid(which='both',axis='both')
    # Voltage
    voltPlot = battPlots[0,1]
    voltPlot.plot(df.elapsedTime, df.battVolt)
    voltPlot.set_title('Battery Voltage')
    voltPlot.set(xlabel='sec',ylabel='volts')
    voltPlot.grid(which='both',axis='both')
    # Current
    battCurrentPlot = battPlots[1,0]
    battCurrentPlot.plot(df.elapsedTime, df.battCurr)
    battCurrentPlot.set_title('Battery Current')
    battCurrentPlot.set(xlabel='sec',ylabel='amps')
    battCurrentPlot.grid(which='both',axis='both')
    # State of Charge
    socPlot = battPlots[1,1]
    socPlot.plot(df.elapsedTime, df.battSOC)
    socPlot.set_title('Battery State of Charge')
    socPlot.set(xlabel='sec',ylabel='percent')
    socPlot.grid(which='both',axis='both')
    plt.subplots_adjust(hspace=0.5,wspace=0.5)  # space in height & width

    # Share the x-axes across all plots.
    # This means zooming in time in one zooms them all to the same timespan
    socPlot.get_shared_x_axes().join(socPlot,battCurrentPlot,voltPlot,tempPlot,
                                    powerPlot,propCurrentPlot,rpmPlot,
                                    elevatorsPlot, ruddersPlot,
                                    hdgPlot,pitchRollPlot,
                                    speedPlot,depthPlot)

    # Put 'em up
    plt.show()


def rpy_to_rotmat(r, p, y):
    return R.from_euler('zyx', [r, p, y], degrees=True).as_matrix()

def calculate_6dof_traj(df: pd.DataFrame) -> None:

    N = len(df)
    poses = np.zeros((4, 4, N))
    for i in range(len(df)):
        poses[:3,:3,i] = rpy_to_rotmat(df.roll[i], df.pitch[i], df.heading[i])

    poses[0, 3, :] = np.linspace(0,1,N)

    return poses

def plot_poses(poses: np.ndarray):
    ims = []
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    _4, __4, N = poses.shape
    for i in range(N):
        ax.quiver(poses[0,3,i],poses[1,3,i],poses[2, 3,i], *poses[0:3,0,i], color='r', length=1, arrow_length_ratio=0.01) # x
        ax.quiver(poses[0,3,i],poses[1,3,i],poses[2, 3,i], *poses[0:3,1,i], color='g', length=1, arrow_length_ratio=0.01) # y
        ax.quiver(poses[0,3,i],poses[1,3,i],poses[2, 3,i], *poses[0:3,2,i], color='b', length=1, arrow_length_ratio=0.01) # z

    ax.auto_scale_xyz([0, 1], [-1, 1], [-1, 1])
    plt.show()

if __name__ == "__main__":

    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", type=str)
    args = parser.parse_args()

    print(f" - Loading file: {args.log_file}")
    df = prepare_data(args.log_file)

    print(' - Plotting')

    # plot_data(df)

    print(' - Calculating Trajectory')
    calculate_6dof_traj(df)

    print('\n === DONE === \n')