import numpy as np

from tifffile import imread,imwrite

from scipy.ndimage.interpolation import shift
from scipy.ndimage import fourier_shift
from scipy.ndimage import shift
from scipy.interpolate import interp1d
from scipy.stats import norm

import os
import shutil
import time
import gc
from tqdm import tqdm

def fourier_shift_frame(frame, shift_x, shift_y, PxToUm=1):

	"""
	Apply a sub-pixel shift to an image frame using the Fourier shift theorem.

	This function shifts an image by a specified number of pixels in both the x and y directions 
	using Fourier transformations. The Fourier shift theorem allows for precise, sub-pixel image 
	translations, which is useful in applications such as image registration or alignment.

	Parameters
	----------
	frame : np.ndarray
		A 2D numpy array representing the image frame to be shifted. The image should be grayscale, 
		with pixel intensities encoded as a floating-point or integer type.
	shift_x : float
		The shift in the x-direction (columns).
	shift_y : float
		The shift in the y-direction (rows). 
	PxToUm : float, optional
		The conversion factor between pixels and micrometers (um). This factor is used to convert 
		the shifts to the correct units if needed. Default is 1, assuming shifts are given in pixels.

	Returns
	-------
	np.ndarray
		A 2D numpy array of the same shape as `frame`, containing the shifted image. The output is 
		in uint16 format, with pixel values clipped to the valid range [0, 65535].

	Notes
	-----
	- This method performs a Fourier transform of the image, applies the shift in Fourier space, 
	  and then performs an inverse Fourier transform to obtain the shifted image.
	- Sub-pixel shifts are achieved by modifying the phase of the Fourier-transformed image.
	"""

	#Fourier transform, apply shift, Fourier invert
	to_align = np.copy(frame)
	fft = np.fft.fft2(to_align)
	fft_shift = fourier_shift(fft,shift=[-shift_y/PxToUm,-shift_x/PxToUm])
	fftm1 = np.fft.ifft2(fft_shift)
	
	return np.array(np.absolute(fftm1),dtype='uint16')


def estimate_displacement(df, timeline, reference_time=0, nbr_tracks_threshold=30):
	
	"""
	Estimate the displacement of tracked objects over time relative to a reference frame.

	This function computes the average displacement (drift) of tracked objects in the x and y 
	directions relative to a specified reference frame. It uses a DataFrame containing tracking 
	data to calculate the shift for each time point in the timeline, considering only tracks 
	that are present both in the reference frame and the current frame.

	Parameters
	----------
	df : pandas.DataFrame
		A DataFrame containing tracking data with columns:
		- 'FRAME': The time frame of the track.
		- 'TRACK_ID': The unique identifier for each track.
		- 'POSITION_X': The x-coordinate of the tracked object.
		- 'POSITION_Y': The y-coordinate of the tracked object.
	timeline : list or array-like
		A sequence of time points over which the displacement will be estimated. Each entry 
		in the timeline corresponds to a frame index for which the displacement is calculated.
	reference_time : int, optional
		The reference time frame against which displacements are calculated. Default is 0, 
		meaning the first time point in the tracking data is used as the reference.
	nbr_tracks_threshold : int, optional
		The minimum number of tracks that must be present in both the reference and current 
		frames to compute displacement. If fewer tracks are found, the displacement is set 
		to NaN for that time point. Default is 30.

	Returns
	-------
	numpy.ndarray
		A 2D array of shape (len(timeline), 2) containing the estimated displacement in the 
		x and y directions for each time point in the timeline. Rows represent time points, 
		and columns represent displacement in x and y, respectively. Missing values are 
		represented as NaN.

	Notes
	-----
	- The displacement for each time frame is calculated only if the number of common tracks 
	  between the reference frame and the current frame exceeds the `nbr_tracks_threshold`.
	- Displacement is estimated using a normal distribution fit (`norm.fit`) to the shifts 
	  in x and y coordinates, providing a robust mean (mu) and standard deviation (std).
	- Missing or insufficient data at any time point will result in NaN values in the 
	  returned displacement array.

	Examples
	--------
	>>> import pandas as pd
	>>> import numpy as np
	>>> data = {'FRAME': [0, 0, 1, 1, 2, 2],
				'TRACK_ID': [1, 2, 1, 2, 1, 2],
				'POSITION_X': [0, 0, 1, 1, 2, 2],
				'POSITION_Y': [0, 0, 1, 1, 2, 2]}
	>>> df = pd.DataFrame(data)
	>>> timeline = [0, 1, 2]
	>>> estimate_displacement(df, timeline, reference_time=0, nbr_tracks_threshold=1)
	array([[ 0.,  0.],
		   [ 1.,  1.],
		   [ 2.,  2.]])
	"""

	displacement = np.zeros((len(timeline), 2), dtype=float)
	displacement[:,:] = np.nan
	
	for _, reference_group in df.loc[df['FRAME']==reference_time].groupby('FRAME'):

		for time, group in df.groupby('FRAME'):

			if time>=reference_time:

				tracks_reference = reference_group['TRACK_ID'].to_numpy()
				tracks = group['TRACK_ID'].to_numpy()

				track_intersection = np.intersect1d(tracks_reference, tracks)

				if len(track_intersection) < nbr_tracks_threshold:
					displacement[time, :] = np.nan
					continue

				# Use track matches to compute the drift
				reference_positions = reference_group.loc[reference_group['TRACK_ID'].isin(track_intersection),['POSITION_X','POSITION_Y']].to_numpy()
				positions = group.loc[group['TRACK_ID'].isin(track_intersection),['POSITION_X','POSITION_Y']].to_numpy()

				shift = positions - reference_positions
				
				# Outlier detection
				q75 = np.percentile(shift,75,axis=0,method="inverted_cdf")
				q25 = np.percentile(shift,25,axis=0,method="inverted_cdf")
				iqr = q75 - q25

				whi_upper = q75 + 1.5*iqr
				whi_lower = q25 - 1.5*iqr
				
				safe_x = shift[:,0]
				safe_x = safe_x[(safe_x>=whi_lower[0])*(safe_x<=whi_upper[0])]

				safe_y = shift[:,1]
				safe_y = safe_y[(safe_y>=whi_lower[1])*(safe_y<=whi_upper[1])]

				if len(safe_x)>0 and len(safe_y)>0:

					mu_x, std_x = norm.fit(safe_x)
					mu_y, std_y = norm.fit(safe_y)
					displacement[time,:] = [mu_x, mu_y]
	
	return displacement

def fill_by_shifting_reference_time(df, timeline, displacement, nbr_tracks_threshold=30, initial_reference_time=0, from_origin=True, extrapolation_kind='linear', interpolate=False):
	
	"""
	Fill missing displacement values in a displacement array by dynamically adjusting the reference time.

	This function corrects missing or invalid displacement values by shifting the reference time to 
	the nearest valid time point. It calculates the displacement relative to this new reference time 
	and updates the displacement array. If no valid reference can be found, it uses interpolation 
	to estimate the displacement.

	Parameters
	----------
	df : pandas.DataFrame
		DataFrame containing tracking data with columns:
		- 'FRAME': The time frame of the track.
		- 'TRACK_ID': The unique identifier for each track.
		- 'POSITION_X': The x-coordinate of the tracked object.
		- 'POSITION_Y': The y-coordinate of the tracked object.
	timeline : list or array-like
		A sequence of time points for which the displacement is calculated.
	displacement : numpy.ndarray
		A 2D array of shape (len(timeline), 2) containing displacement values in x and y directions.
		Missing values should be represented as NaN.
	nbr_tracks_threshold : int, optional
		Minimum number of tracks required in both reference and current frames for a valid 
		displacement calculation. Default is 30.
	initial_reference_time : int, optional
		The initial time point used as a reference for displacement calculation. Default is 0.
	from_origin : bool, optional
		If True, the search for a new reference time starts from the beginning of the timeline. 
		If False, the search starts from the current frame and goes backward. Default is True.
	extrapolation_kind : str, optional
		The kind of extrapolation used by `scipy.interpolate.interp1d` when estimating missing values.
		Options include 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', etc. Default is 'linear'.

	Returns
	-------
	numpy.ndarray
		The corrected displacement array with the same shape as the input `displacement`. Missing 
		values are replaced with interpolated or newly calculated displacements based on adjusted references.

	Notes
	-----
	- The function tries to find a new reference time if the current reference time does not have valid 
	  displacement values. It iterates through the timeline until a valid reference time is found.
	- If a new reference time is set, it checks the displacement up to this new reference time. If the 
	  displacement values are not valid, it uses interpolation to fill in the missing values.
	"""
	

	displacement_matrix = np.zeros((len(timeline),len(timeline),2))
	displacement_matrix[:,:,:] = np.nan
	displacement_matrix[initial_reference_time, :, :] = displacement

	for t in tqdm(timeline):

		dx,dy = tuple(displacement[t])

		if dx!=dx:
			print(f"Time {t}: No displacement with reference t={initial_reference_time} found... Attempt to change the reference...")
			
			if from_origin:
				s=1
			else:
				s=(t-1)
			while dx!=dx:
				if from_origin:
					if s==t:
						print(f"Time {t}: Failed to find a valid reference time...")
						s = None
						break					
				else:  
					if s<0:
						print(f"Time {t}: Failed to find a valid reference time...")
						s = None
						break
			
				# Check if the displacement profile was never computed before
				if np.all([val!=val for val in displacement_matrix[s,:,0]]):
					displacement_new_ref = estimate_displacement(df, timeline, reference_time=s, nbr_tracks_threshold=nbr_tracks_threshold)
					displacement_matrix[s,:,:] = displacement_new_ref
				else:
					displacement_new_ref = displacement_matrix[s,:,:]

				dx,dy = tuple(displacement_new_ref[t])
				
				# Continue going backwards
				if from_origin:
					s += 1
				else:
					s-=1
			
			# Exit
			if s is not None:
				if from_origin:
					s-=1
				else:
					s+=1
				print(f"Time {t}: New reference time set to t={s}...")
		
				# Check if the displacement up to s exists... Extrapolate the disp otherwise...
				previous_displacements_x = displacement[:(s+1),0]
				previous_displacements_y = displacement[:(s+1),1]
				timeline_truncated = timeline[:(s+1)]

				if previous_displacements_x[-1]!=previous_displacements_x[-1]:
					
					print(f"Time {t}: Displacement(t={s}) not valid... Extrapolating this value...")
					timeline_safe = timeline_truncated[previous_displacements_x==previous_displacements_x]
					
					# need to extrapolate
					fx = interp1d(timeline_safe, previous_displacements_x[previous_displacements_x==previous_displacements_x], kind=extrapolation_kind,fill_value='extrapolate')
					fy = interp1d(timeline_safe, previous_displacements_y[previous_displacements_x==previous_displacements_x], kind=extrapolation_kind,fill_value='extrapolate')
					interp_x = fx(timeline[s]); interp_y = fy(timeline[s])
					displacement[s,:] = tuple([interp_x, interp_y])					

				dx += displacement[s,0]
				dy += displacement[s,1]
				displacement[t,:] = tuple([dx,dy])
				print(f"Time {t}: Displacement_x(t={s}) = {displacement[s,0]}\nTotal displacement_x = {displacement[t,0]}")

	if interpolate:		
		disp_x = displacement[:,0]
		nan_x = [i for i in range(len(timeline)) if disp_x[i]!=disp_x[i]]
		fx = interp1d(timeline[disp_x==disp_x], disp_x[disp_x==disp_x], kind="linear", fill_value='extrapolate')
		
		disp_y = displacement[:,1]
		nan_y = [i for i in range(len(timeline)) if disp_y[i]!=disp_y[i]]		
		fy = interp1d(timeline[disp_y==disp_y], disp_y[disp_y==disp_y], kind="linear", fill_value='extrapolate')
		
		for t in nan_x:
			displacement[t,0] = fx(timeline[t])
		for t in nan_y:
			displacement[t,1] = fy(timeline[t])

	return displacement.copy()


def align_frames(frames, displacement, PxToUm=1, output_dir=None, return_stack=True):
	
	"""
	Aligns a sequence of image frames based on provided displacement values using Fourier shift.

	This function reads image frames, applies a Fourier shift transformation to correct for the 
	specified displacements in the x and y directions, and optionally saves the aligned frames 
	to a specified directory. The displacement values are provided in a 2D array, where each row 
	corresponds to the displacement (dx, dy) for the respective frame.

	Parameters
	----------
	frames : list of str
		List of file paths to the image frames to be aligned.
	displacement : numpy.ndarray
		A 2D array of shape (n_frames, 2) containing displacement values for each frame in the x 
		and y directions. If displacement is NaN for a frame, it is replaced with a zero-filled image.
	PxToUm : float, optional
		Pixel-to-micron conversion factor used to scale the displacement values. Default is 1.
	output_dir : str, optional
		Path to the directory where the aligned images should be saved. If None, the images are 
		not saved to disk. If specified, a subdirectory 'aligned' will be created to store the 
		aligned images. Default is None.
	return_stack : bool, optional
		If True, the function returns the stack of aligned images as a numpy array. If False, 
		it only saves the aligned images to disk (if `output_dir` is specified) and returns None. 
		Default is True.

	Returns
	-------
	numpy.ndarray or None
		If `return_stack` is True, returns a numpy array of the aligned image stack with shape 
		(n_frames, height, width). If `return_stack` is False, returns None.

	Notes
	-----
	- If `output_dir` is specified, the aligned frames are saved as 16-bit TIFF images in a 
	  subdirectory named 'aligned' within `output_dir`. Any existing 'aligned' subdirectory is 
	  removed before saving the new images.
	- The function applies a Fourier shift to each frame based on the corresponding displacement 
	  values. If the displacement value is NaN, a zero-filled image of the same size is used instead.
	- The `fourier_shift_frame` function is used to perform the alignment of each frame.

	Examples
	--------
	>>> frames = ['frame_0000.tif', 'frame_0001.tif', 'frame_0002.tif']
	>>> displacement = np.array([[1.0, -2.0], [0.5, 1.0], [np.nan, np.nan]])
	>>> aligned_images = align_frames(frames, displacement, PxToUm=1.0, output_dir='aligned_frames', return_stack=True)
	>>> print(aligned_images.shape)
	(3, 512, 512)

	# This example reads three frames, aligns them based on the specified displacements,
	# saves them to the 'aligned_frames/aligned/' directory, and returns the aligned stack.
	"""
	
	if output_dir is not None:
		if not output_dir.endswith(os.sep):
			output_dir += os.sep
		if not os.path.exists(output_dir):
			os.mkdir(output_dir)
		output_dir = output_dir + "aligned"
		if os.path.exists(output_dir):
			shutil.rmtree(output_dir)
		os.mkdir(output_dir)
	
	aligned_stack = []
	for t in range(len(frames)):
		padded_t = str(int(t)).zfill(4)
		dx,dy = displacement[t]
		frame = imread(frames[t])
		if dx==dx:
			fftm1 = fourier_shift_frame(frame.astype(float), dx, dy, PxToUm=PxToUm)
		else:
			fftm1 = np.zeros_like(frame)
		
		if output_dir is not None:
			imwrite(os.sep.join([output_dir,f"out_{padded_t}.tif"]),fftm1,dtype='uint16')
		
		if return_stack:
			aligned_stack.append(fftm1)
		else:
			del frame
			del fftm1
			gc.collect()	
	if return_stack:
		return np.array(aligned_stack)