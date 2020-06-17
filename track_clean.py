"""
A script example to clean and smooth USBL/GPS track data of ROV
"""
import time
from datetime import timedelta
import numpy as np
import numba as nb
import pandas as pd
from scipy import interpolate
from sklearn.neighbors import NearestNeighbors
from rdp import rdp

# path\name to input and output csv
input = "path/to/input.csv"
output = "path/to/input.csv"

# precision of the USBL eg. a value of 0.01 means 1% precision per meter depth
depth_multiplier = 0.005

# number of neighbors to compute distances. see function dist_filter below.
knn = 10

# distance from the mean for a point to be considered a outlier (same unit as lat, lon, depth)
outlier_thr = 20
# window size to compute the rolling mean
mean_window = 5

###########################
start_time = time.time()
knn2 = knn//2

@nb.njit
def dist_filter(xyz, dists, result):
    """ Distance threshold filter
    Remove all sequential points that are within a threshold
    xyz = array of shape (N, 3) with long, lat and depth
    dists = sum of distance from k Nearest Neighbors, for each point. array of shape (N,)
            the point with lowest dists will be kept
    result = output array indicating points to keep or to remove
             first and last points must already be set to true
    """
    i = knn2
    il = 0
    end = xyz.shape[0] - knn2
    while i < end:
        dst = ((xyz[il,0] - xyz[i,0])**2 + \
               (xyz[il,1] - xyz[i,1])**2 + \
               (xyz[il,2] - xyz[i,2])**2)**0.5

        if dst >= abs(xyz[i,2])*depth_multiplier:
            il = i - knn2 + np.argmin(dists[i-knn2:i+knn2+1])
            result[il] = True
            i += knn2 - 1
        else:
            i += 1

    return result

@nb.njit
def heading_calc(mat):
    """
    Convert sin and cos back to degree where heading is nan
    mat - array of shape (M, 3) with heading and interpolated sin and cos
    """
    result = np.zeros(mat.shape[0])
    for i in range(mat.shape[0]):
        if np.isnan(mat[i,0]):
            result[i] = np.round(np.rad2deg(np.arctan2(mat[i,1], mat[i,2])), 1)
        else:
            result[i] = mat[i,0]
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

df_tmp = []
col_stopped = dt.columns.get_loc('nstopped')
for i, g_dive in dt.groupby(by=['Dive']):
    # get first and last group of the dive
    first = g_dive.iloc[0, col_stopped]
    last = g_dive.iloc[-1, col_stopped]

    # remove outliers by calculating the difference from the value and the rolling mean
    means = g_dive[['Lon','Lat','Depth']].rolling(mean_window, center=True).mean()
    means = means.interpolate(method='pad')
    diffs = np.abs(means - g_dive[['Lon','Lat','Depth']]) < outlier_thr
    g_dive = g_dive[diffs['Lon'] & diffs['Lat'] & diffs['Depth']]

    for i2, g_stopped in g_dive.groupby(by=['nstopped']):
        # check if current group is_stopped or not
        track_dive = track[track['Dive'] == i]
        track_g = track_dive[track_dive['nstopped'] == i2]
        
        if track_g['is_stopped'].iloc[0]:
            # generate a time for every second
            # also get missing timestamps between groups
            start = max(track_g.index[0] - 1, track_dive.index[0]) # avoid to get values from another dive
            end = min(track_g.index[-1] + 1, track_dive.index[-1])
            time_new = pd.date_range(track_dive['Date_Time'].loc[start], track_dive['Date_Time'].loc[end], freq='S')
            time_new = time_new[1:-1]

            # generate a df with mean values
            df1 = pd.DataFrame({'Dive': i, 'nstopped': i2, \
                               'Date_Time': time_new, 'Lon': g_stopped['Lon'].mean(), \
                               'Lat': g_stopped['Lat'].mean(), 'Depth': g_stopped['Depth'].mean()})
            
            # set the fist and last value from stopped to moving group
            # this is necessary for the filters
            col_stopped = df1.columns.get_loc('nstopped')
            if i2 == first:
                df1.iloc[-1, col_stopped] = i2 + 1
            elif i2 == last:
                df1.iloc[0, col_stopped] = i2 - 1
            else:
                df1.iloc[-1, col_stopped] = i2 + 1
                df1.iloc[0, col_stopped] = i2 - 1
        
        else:
            # if is moving, simple store the group
            df1 = g_stopped.copy()

        df_tmp.append(df1)

# concatenate data frames
dt = pd.concat(df_tmp)

# loop again, now applying the filters
df_tmp = []
for i, g in dt.groupby(by=['Dive','nstopped']):
    # track_dive = track[track['Dive'] == i[0]]
    # track_g = track_dive[track_dive['nstopped'] == i[1]]
    track_g = track[(track['Dive'] == i[0]) & (track['nstopped'] == i[1])]
    if track_g['is_stopped'].iloc[1]:
        df1 = g.drop(columns=['nstopped'])
    else:
        time_new = pd.date_range(track_g['Date_Time'].iloc[0], track_g['Date_Time'].iloc[-1], freq='S')
        
        # get the first and last value of the track
        xyz = np.array(g[['Lon','Lat','Depth']], dtype=float)
        result = np.zeros(xyz.shape[0], dtype=bool)
        result[0] = True; result[-1] = True
        
        
        if xyz.shape[0] <= knn + 1:
            # segment is too short, just get the first and last value
            filtered = g.loc[result]
        
        else:
            # set the score for each point
            nbrs = NearestNeighbors(n_neighbors=knn, algorithm='kd_tree')
            nbrs.fit(xyz)
            distances, indices = nbrs.kneighbors(xyz)
            dists = np.sum(distances, axis=1)

            # remove points too close
            index = dist_filter(xyz, dists, result)
            filtered = g.loc[index]

            # apply Ramer-Douglas-Peucker algorithm
            eps = abs(filtered['Depth'].mean())*depth_multiplier/2
            index = rdp(filtered[['Lon','Lat','Depth']], epsilon = eps, return_mask = True)
            filtered = filtered.loc[index]

        # interpolate for each second
        ### uncomment below lines for linear interpolation
        # fit = interpolate.interp1d(filtered['Date_Time'].values.astype(float), filtered[['Lon','Lat','Depth']], \
        #                             kind = 'linear', axis = 0, fill_value = 'extrapolate')
        # xyz_new = fit(time_new.values.astype(float))
        ###
        ### or uncomment these for pchip interpolation
        fit = interpolate.PchipInterpolator(filtered['Date_Time'], filtered[['Lon','Lat','Depth']])
        xyz_new = fit(time_new, extrapolate = True)
        ###

        df1 = pd.DataFrame({'Date_Time': time_new, 'Lon': xyz_new[:,0], 'Lat': xyz_new[:,1], 'Depth': xyz_new[:,2]})
        df1['Dive'] = i[0]
    
     # append data frame
    df_tmp.append(df1)

df1 = pd.concat(df_tmp)


# now join the heading
df2 = pd.merge(df1, track[['Dive','Date_Time','Heading']], on=['Dive','Date_Time'], how='left')

# interpolate heading missing values
df_tmp = []
for i, g in df2.groupby(by=['Dive']):
    df3 = g.copy()
    df3['head_rad'] = np.deg2rad(df3['Heading'])
    df3['cos'] = np.cos(df3['head_rad'])
    df3['sin'] = np.sin(df3['head_rad'])
    df3[['cos','sin']] = df3[['cos','sin']].interpolate(method='linear')
    df3['Heading'] = heading_calc(df3[['Heading','cos','sin']].to_numpy(float))
    # df3['Heading'] = df3.apply(lambda x: np.round(np.rad2deg(np.arctan2(x['sin'], x['cos'])), 1) \
    #                             if pd.isnull(x['Heading']) else x['Heading'], axis=1)
    df3 = df3.drop(columns = ['head_rad','cos','sin'])
    df_tmp.append(df3)

# Concatenate data frames and save csv
final_df = pd.concat(df_tmp)
final_df['Date'] = final_df['Date_Time'].dt.date
final_df['Time'] = final_df['Date_Time'].dt.time
final_df = final_df[['Dive','Date', 'Time','Lon','Lat','Depth','Heading']]
final_df.to_csv(output, index=False, float_format='%.2f')

finish_time = time.time()
duration = finish_time - start_time
print(f'Elapsed time: {duration:.2f}s')