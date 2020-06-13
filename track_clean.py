"""
A script example to clean and smooth USBL track data of ROV
"""
from datetime import timedelta
import numpy as np
import numba as nb
import pandas as pd
from scipy.interpolate import Akima1DInterpolator
from rdp import rdp

# path\name to input and output csv
input = "path/to/input.csv"
output = "path/to/output.csv"

# precision of the USBL eg. a value of 0.01 means 1% precision per meter depth
depth_multiplier = 0.01

###########################
@nb.njit
def dist_thr(xyz, result):
    # Distance threshold filter
    # Remove all sequential points that are within a threshold
    result[0] = True; result[-1] = True
    il = 0
    for i in range(1, xyz.shape[0]):
        dst = ((xyz[il,0] - xyz[i,0])**2 + \
               (xyz[il,1] - xyz[i,1])**2 + \
               (xyz[il,2] - xyz[i,2])**2)**0.5
        if dst >= abs(xyz[i,2])*depth_multiplier:
            result[i] = True
            il = i
    return result

track = pd.read_csv(input, parse_dates = [['Date', 'Time']], infer_datetime_format=True)

# remove duplicated rows with same time, lat, long, etc
track = track.loc[track['Date_Time'].shift() != track['Date_Time']]

# create a colum for each consecutive is_stopped
track['nstopped'] = (track.is_stopped != track.is_stopped.shift()).cumsum()

# remove rows with NA values
track_c = track[track['Lon'].notnull()]

# remove duplicated lat, lon and depth, and shift the time in these rows
ss = timedelta(seconds=0.5)
group = track_c.groupby(by=['Dive','nstopped','Lon','Lat','Depth'], sort=False)
i = group.size() - 1
dt = group['Date_Time'].first() + i*ss
dt = dt.reset_index(name='Date_Time')

# loop for each time is stopped and for each dive
final_df = []
for i, g in dt.groupby(by=['Dive','nstopped']):
    heading_df = track[(track['Dive'] == i[0]) & (track['nstopped'] == i[1])]
    time_new = pd.date_range(heading_df['Date_Time'].iloc[0], heading_df['Date_Time'].iloc[-1], freq='S')

    if heading_df['is_stopped'].iloc[0]:
        df1 = pd.DataFrame({'Date_Time': time_new, 'Lon': g['Lon'].mean(), \
            'Lat': g['Lat'].mean(), 'Depth': g['Depth'].mean()})

    else:
        # remove points too close
        index = dist_thr(np.array(g[['Lon','Lat','Depth']], dtype=float), np.zeros(len(g.index), dtype=bool))
        filtered = g.loc[index]

        # apply Ramer-Douglas-Peucker algorithm
        eps = abs(filtered['Depth'].mean())*depth_multiplier
        index = rdp(filtered[['Lon','Lat','Depth']], epsilon=eps, return_mask=True)
        filtered = filtered.loc[index]

        # interpolate for each second
        fit = Akima1DInterpolator(filtered['Date_Time'], filtered[['Lon','Lat','Depth']])
        xyz_new = fit(time_new, extrapolate=True)
        df1 = pd.DataFrame({'Date_Time': time_new, 'Lon': xyz_new[:,0], 'Lat': xyz_new[:,1], 'Depth': xyz_new[:,2]})

    # now join the heading
    df2 = pd.merge(df1,heading_df[['Date_Time','Heading']], on='Date_Time', how='left')

    # interpolate missing heading
    df2['head_rad'] = np.deg2rad(df2['Heading'])
    df2['cos'] = np.cos(df2['head_rad'])
    df2['sin'] = np.sin(df2['head_rad'])
    df2[['cos','sin']] = df2[['cos','sin']].interpolate(method='linear')
    df2['Heading'] = df2.apply(lambda x: np.round(np.rad2deg(np.arctan2(x['sin'], x['cos']))) \
                                if pd.isnull(x['Heading']) else x['Heading'], axis=1)
    df2 = df2.drop(columns=['head_rad','cos','sin'])

    # append data frame
    df2['Dive'] = i[0]
    final_df.append(df2)

# Concatenate data frames and save csv
final_df = pd.concat(final_df)
final_df['Date'] = final_df['Date_Time'].dt.date
final_df['Time'] = final_df['Date_Time'].dt.time
final_df = final_df[['Dive','Date', 'Time','Lon','Lat','Depth','Heading']]
final_df.to_csv(output, index=False)
