"""
Display video and track simultaneously
"""
import numpy as np
import numba as nb
import pandas as pd
import cv2
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui

# path to csv track file
track_file = "path/to/track.csv"

# path to video file
file = "path/to/video"

start = '00:00:00' # time to start the video
start_real = 'hh:mm:ss' # real time to get data from track, where video is set to start
dive = file[-10:-6] # current dive - can be a substring of file

ps = 2 # size of plot, in the same scale of the coordinates
ws = 15*ps # number of reads to take before and after current location

# Parameters for time display over the image
text_params = dict(org = (47,72), # position
                   fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                   fontScale = 0.6,
                   color = (0,0,255), # BRG
                   thickness = 1)

# Parameters to resize images when displayed
resize_params = dict(dsize = (0,0),
                     fx = 0.5, fy = 0.5, # scale factor
                     interpolation = cv2.INTER_AREA)

# press Esc to exit (window 'frame' must be active)
###########################
class NonScientific(pg.AxisItem):
    # This is to avoid y axis to be displayed as scientific notation (happens when using UTM)
    def __init__(self, *args, **kwargs):
        super(NonScientific, self).__init__(*args, **kwargs)

    def tickStrings(self, values, scale, spacing):
        return [int(value*1) for value in values]

@nb.njit
def rotate(x, y, origin=(0, 0), degrees=0):
    # Rotate x, y coordinates so the ROV heading is always facing upwards in the plot
    angle = np.deg2rad(degrees)
    asin = np.sin(angle)
    acos = np.cos(angle)
    R = np.array([[acos, -asin], [asin,  acos]])
    o = np.array([[origin[0]],[origin[1]]])
    p = np.vstack((x,y))
    rotated = R @ (p - o) + o
    return rotated[0], rotated[1]

# get data from file
track = pd.read_csv(r"G:\Meu Drive\Doutorado\Videos HyBIS\scripts\track clean\track_clean_python.csv")
i = track.index[(track.Dive == dive) & (track.Time == start_real)].tolist()[0]

# class to plot the coordinates
class PlotCoords:
    def __init__(self):
        self.app = QtGui.QApplication([])
        self.win = pg.GraphicsWindow(title='track')
        self.win.resize(600,550) # widows size

        self.win.addLabel('', row=0, col=0)
        self.win.addLabel('longitude X latitude', row=1, col=0, angle=-90)
        self.win.addLabel('', row=2, col=0)
        self.p1 = self.win.addPlot(row=0, col=1, rowspan=3, axisItems={'left': NonScientific(orientation='left')})
        self.curve11 = self.p1.plot(pen='r')
        self.curve12 = self.p1.plot(pen='g')
        self.win.addLabel('depth X distance', row=3, col=0, angle=-90)
        self.p2 = self.win.addPlot(row=3, col=1)
        self.curve21 = self.p2.plot(pen='r')
        self.curve22 = self.p2.plot(pen='b')

        self.a1 = pg.ArrowItem(angle=90, headLen=25)
        self.p1.addItem(self.a1)
        self.a2 = pg.ArrowItem(angle=180, headLen=25)
        self.p2.addItem(self.a2)
    
    def update(self, data_x, data_y, data_z):
        xc, yc = data_x[ws], data_y[ws]

        x_dif = np.diff(data_x)
        y_dif = np.diff(data_y)
        data_dist = (x_dif**2 + y_dif**2)**0.5
        data_dist = np.cumsum(data_dist)
        zc, dc = data_z[ws], data_dist[ws]

        data_xr, data_yr = rotate(data_x, data_y, origin=(xc, yc), degrees=track.Heading[i])

        self.curve11.setData(data_xr[:(ws+1)], data_yr[:(ws+1)])
        self.curve12.setData(data_xr[ws:-1], data_yr[ws:-1])
        self.curve21.setData(data_dist[:(ws+1)], data_z[:(ws+1)])
        self.curve22.setData(data_dist[ws:], data_z[ws:])

        self.p1.setRange(xRange=[xc-ps, xc+ps],yRange=[yc-ps, yc+ps])
        self.p2.setRange(xRange=[dc-ps, dc+ps],yRange=[zc-ps, zc+ps])
        self.a1.setPos(xc,yc)
        self.a2.setPos(dc,zc)

        QtGui.QApplication.processEvents()

# open video capture
cap = cv2.VideoCapture(file)
fps = cap.get(cv2.CAP_PROP_FPS)

plot = PlotCoords()

# jump to specified time
s = start.split(':')
jump = (3600*int(s[0]) + 60*int(s[1]) + int(s[2]))*1000
retval = cap.set(cv2.CAP_PROP_POS_MSEC, jump)

retval, frame = cap.read()
t = 0
while retval:
    frame_small = cv2.resize(frame, **resize_params)
    retval, frame = cap.read() # read for next interation

    if t % fps == 0:
        # plot is updated every second
        data_x = track.Lon[(i-ws):(i+ws+2)].ravel()
        data_y = track.Lat[(i-ws):(i+ws+2)].ravel()
        data_z = track.Depth[(i-ws):(i+ws+1)].ravel()
        plot.update(data_x, data_y, data_z)
        time = track.Time[i]
        i += 1
        
    cv2.putText(frame_small, time, **text_params)
    cv2.imshow('frame', frame_small)
    
    t += 1
    k = cv2.waitKey(1)
    if  k == 27: # press Esc to exit
        break

cap.release()
cv2.destroyWindow('frame')
plot.app.exit()
