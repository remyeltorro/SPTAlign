import pandas as pd
from os import sep
from natsort import natsorted
from glob import glob
import numpy as np
from tifffile import imread

def locate_frames(path, load=False):

	"""
	Locate and optionally load TIFF image frames from a specified directory.

	This function searches for all `.tif` image files in the given directory and returns a 
	sorted list of their file paths. If the `load` parameter is set to `True`, the function 
	reads the image files and returns a list of loaded image arrays.

	Parameters
	----------
	path : str
		Path to the directory containing the TIFF image frames.
	load : bool, optional
		If `True`, loads the image frames as numpy arrays and returns them. If `False`, 
		only returns the sorted list of file paths. Default is `False`.

	Returns
	-------
	list
		If `load` is `False`, returns a sorted list of file paths to the TIFF image frames.
		If `load` is `True`, returns a list of loaded image arrays.

	Notes
	-----
	- The function uses the `natsorted` function to ensure natural sorting of file names, 
	  which is useful when the frame file names contain numerical indices.
	- If `load` is `True`, the function loads the images using the `imread` function from 
	  the `skimage.io` module.

	Examples
	--------
	>>> frames = locate_frames('/path/to/frames/')
	>>> print(frames)
	['/path/to/frames/frame_0000.tif', '/path/to/frames/frame_0001.tif', ...]

	>>> frames_data = locate_frames('/path/to/frames/', load=True)
	>>> print(frames_data[0].shape)
	(512, 512)

	# This example locates and loads all TIFF frames from the specified directory,
	# then prints the shape of the first frame's array.
	"""

	if not path.endswith(sep):
		path += sep

	frames = natsorted(glob(path+"*.tif"))
	if load:
		frames = [imread(f) for f in frames]

	return frames
	

def is_float(element: any) -> bool:
	
	"""
	Check if an element can be converted to a float.

	This function determines whether a given element can be safely converted 
	to a float without raising a `ValueError`. If `None` is passed, the function 
	returns `False`. The method is particularly useful for validating user input 
	or data parsing.

	Parameters
	----------
	element : any
		The element to check, which can be of any type (e.g., string, integer, 
		float, or None).

	Returns
	-------
	bool
		Returns `True` if the element can be converted to a float, otherwise `False`.

	Examples
	--------
	>>> is_float('3.14')
	True
	
	>>> is_float('not a number')
	False
	
	>>> is_float(None)
	False

	Notes
	-----
	- The function is robust against `None` inputs, returning `False` immediately.
	- It uses a `try-except` block to catch `ValueError` exceptions when attempting 
	  to convert the element to a float.
	- Adapted from a Stack Overflow solution: 
	  https://stackoverflow.com/questions/736043/checking-if-a-string-can-be-converted-to-float-in-python
	"""

	if element is None: 
		return False
	try:
		float(element)
		return True
	except ValueError:
		return False


def filter_by_tracklength(trajectories, minimum_tracklength, track_label="TRACK_ID"):
	
	"""
	Filter trajectories based on the minimum track length.

	Parameters
	----------
	trajectories : pandas.DataFrame
		The input DataFrame containing trajectory data.
	minimum_tracklength : int
		The minimum length required for a track to be included.
	track_label : str, optional
		The column name in the DataFrame that represents the track ID.
		Defaults to "TRACK_ID".

	Returns
	-------
	pandas.DataFrame
		The filtered DataFrame with trajectories that meet the minimum track length.

	Notes
	-----
	This function removes any tracks from the input DataFrame that have a length
	(number of data points) less than the specified minimum track length.

	Examples
	--------
	>>> filtered_data = filter_by_tracklength(trajectories, 10, track_label="TrackID")
	>>> print(filtered_data.head())

	"""
	
	if minimum_tracklength>0:
		
		leftover_tracks = trajectories.groupby(track_label, group_keys=False).size().index[trajectories.groupby(track_label, group_keys=False).size() > minimum_tracklength]
		trajectories = trajectories.loc[trajectories[track_label].isin(leftover_tracks)]
	
	trajectories = trajectories.reset_index(drop=True)
	
	return trajectories


def load_tracks(csv, minimum_tracklength=0, column_labels={'track': "TRACK_ID", 'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'}):
	
	"""
	Load and preprocess tracking data from a CSV file.

	This function reads a CSV file containing tracking data and performs basic 
	preprocessing, including dropping rows with missing track IDs, removing 
	invalid rows based on position data, and converting data types for specified 
	columns. The data is then sorted by track ID and time for easier analysis.

	Parameters
	----------
	csv : str
		Path to the CSV file containing tracking data.
	column_labels : dict, optional
		A dictionary mapping expected column names in the CSV file to the 
		corresponding labels in the DataFrame. Keys should be:
		- 'track': Column representing track or object IDs.
		- 'time': Column representing time or frame numbers.
		- 'x': Column representing the x-coordinate of positions.
		- 'y': Column representing the y-coordinate of positions.
		Defaults to:
		{'track': "TRACK_ID", 'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'}.

	Returns
	-------
	pd.DataFrame
		A preprocessed pandas DataFrame containing tracking data, with columns 
		for track ID, time, and position (x, y), sorted by track ID and time.

	Raises
	------
	FileNotFoundError
		If the specified CSV file does not exist.
	KeyError
		If the specified column labels do not match the columns in the CSV file.

	Examples
	--------
	>>> df = load_tracks("tracking_data.csv")
	>>> print(df.head())
	   TRACK_ID  FRAME  POSITION_X  POSITION_Y
	0         1      0         0.0         0.0
	1         1      1         1.0         1.0
	2         2      0         2.0         2.0
	3         2      1         3.0         3.0

	Notes
	-----
	- The function removes rows with NaN values in the track ID column to ensure 
	  valid tracking data.
	- Rows containing non-numeric x or y positions are dropped to avoid errors 
	  during type conversion.
	- The function sorts the DataFrame by 'track' and 'time' columns for better 
	  organization and subsequent analysis.

	"""

	df = pd.read_csv(csv, low_memory=False)
	df = df.dropna(subset=[column_labels['track']])

	# Drop label rows if any
	df = df.loc[df[column_labels['x']].apply(lambda x: is_float(x)),:]
	
	df[column_labels['track']] = df[column_labels['track']].astype(int)
	df[column_labels['time']] = df[column_labels['time']].astype(int)
	df[column_labels['x']] = df[column_labels['x']].astype(float)
	df[column_labels['y']] = df[column_labels['y']].astype(float)

	# Sort
	df = df.sort_values(by=[column_labels['track'],column_labels['time']])
	df = filter_by_tracklength(df, minimum_tracklength, track_label=column_labels['track'])

	return df

def estimate_timeline(stack, tracks,column_labels={'track': "TRACK_ID", 'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'}):

	"""
	Estimate the timeline of frames based on image stack and tracking data.

	This function generates a timeline, represented as a numpy array, 
	which corresponds to the number of frames in the provided image stack. 
	It also checks for consistency between the number of frames in the 
	image stack and the maximum time frame in the tracking data.

	Parameters
	----------
	stack : list or np.ndarray
		List or array of image frames representing the stack.
	tracks : pd.DataFrame
		DataFrame containing tracking data, with columns specified in the 
		`column_labels` parameter.
	column_labels : dict, optional
		A dictionary mapping expected column names in the DataFrame `tracks` 
		to the corresponding labels. Keys should be:
		- 'track': Column representing track or object IDs.
		- 'time': Column representing time or frame numbers.
		- 'x': Column representing the x-coordinate of positions.
		- 'y': Column representing the y-coordinate of positions.
		Defaults to:
		{'track': "TRACK_ID", 'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'}.

	Returns
	-------
	np.ndarray
		A numpy array representing the timeline of frames, ranging from 0 
		to the maximum frame index.

	Raises
	------
	AssertionError
		If the number of frames in the stack does not match the maximum 
		time frame in the tracking data.

	Examples
	--------
	>>> stack = [np.zeros((10, 10)) for _ in range(50)]
	>>> tracks = pd.DataFrame({
	...     'TRACK_ID': [1, 1, 2, 2],
	...     'FRAME': [0, 1, 0, 1],
	...     'POSITION_X': [5, 6, 1, 2],
	...     'POSITION_Y': [5, 6, 1, 2]
	... })
	>>> timeline = estimate_timeline(stack, tracks)
	>>> print(timeline)
	[ 0  1  2  3 ... 49]

	Notes
	-----
	- The function checks if the number of frames in the image stack matches 
	  the maximum frame index in the tracking data to ensure synchronization.
	- The timeline is simply an array of integers representing frame indices.
	"""

	max_frame = len(stack)
	max_frame_in_df = int(tracks[column_labels['time']].max()) + 1
	assert max_frame==max_frame_in_df,"The number of frames in the stack is not equal to the maximum time in the tracked data..."
	timeline = np.arange(max_frame)

	return timeline

