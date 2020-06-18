# smooth rov track

### Python script to clean and smooth track data (GPS/USBL) of ROVs

This is a script example to clean and smooth USBL track data of ROVs.
It will remove invalid lines and outliers, apply a distance threshold filter, followed by 3D Ramer-Douglas-Peucker algorithm.
See [here](https://www.gpsvisualizer.com/tutorials/track_filters.html) for more a detailed explanation of the algorithms.
The output are interpolated readings for each second.

Feel free to modify these scripts as you need!

### Usage
There are two scripts in the folder:
- `track_clean.py` - clean and smooth track data
- `track_view.py` - simple GUI to display video and track simultaneously

Check and edit parameters in the beginning of each script.

Input track must be a csv file as below:

|Dive|Date|Time|Lon|Lat|Depth|Heading|is_stopped|
|---|---|---|---|---|---|---|---|
|Dive_X|dd/mm/aaaa|hh:mm:ss|X.X|X.X|-X.X|degrees|True/False|

- Lon and Lat must be in te same scale as depth (eg. coordinates in UTM and depth in meters)
- is_stopped is 'True' when ROV is stopped and 'False' when is moving.

#### Observations
Heading is not cleaned by the filters, but missing values are interpolated. You can fill a column with zeroes to ignore it.<br>
Seconds where is_stopped are grouped with mean values of lat, long and depth. You can set all to 'False' if ROV is always moving.
You should always check the tracks in a GIS software. The track_view.py does not intend to substitute that.

#### Requirements
`numpy`, `numba`, `pandas`, `scipy`, `sklearn` and `rdp`<br>
`track_clean.py` also requires `opencv`, `pyqtgraph` and `PyQt5` <br>
All can be installed using **pip**
