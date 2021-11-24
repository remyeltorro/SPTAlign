import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tifffile import imread,imwrite
from scipy.ndimage.interpolation import shift
from scipy.ndimage import fourier_shift
from scipy.ndimage import shift
from scipy.interpolate import interp1d
from natsort import natsorted
from glob import glob
import gc
from PIL import Image
import os
import shutil
from numba import njit
import time

# Following https://github.com/remyeltorro/subpixel_registration_spt 
directory = "../data/images/"
trajectories = "../data/trajectories.csv"
output_dir = "../output/"

if os.path.isdir(output_dir):
	shutil.rmtree(output_dir)
os.mkdir(output_dir)
os.mkdir(output_dir+"aligned/")

#colTID should be named "TRACK_ID" in the table
colX = "POSITION_X"; colY = "POSITION_Y"; colTID = "TRACK_ID"; colF = "FRAME";

# set pixel calibration used during tracking
PxToUm = float(input('What is the pixel calibration? '))

@njit
def mean_displacement(x,y,times):
	"""
	This function takes a series of positions x and y and computes the average displacement.
	Function to be refined.
	"""
	fullrange = list(np.linspace(min(times),max(times),int(max(times)-min(times))+1)) #to include gaps in average
	s=0
	for p in range(0,len(x)-1):
		s+= np.sqrt((x[p+1]-x[p])**2+(y[p+1]-y[p])**2)
	return(s/len(fullrange))


def check_consecutive(frames,dt=1):
	"""
	This function looks for consecutive values in the array frames
	and organises them in chunks of consecutive values [[val1,val1+1,...,val1+n],[val2,val2+1,...]]
	with val2!=(val1+n)
	"""
	frames = np.sort(frames)
	slices = np.zeros(len(frames))
	key = 0
	for i in range(1,len(frames)):
		coef = int((frames[i] - frames[i-1]) / dt)
		if coef!=1:
			key+=1
			slices[i] = key

		elif coef==1:
			slices[i] = key
	return(np.array(slices),np.array(frames))


def fourier_shift_frame(frame,shift_x,shift_y,calibration):
	#Fourier transform, apply shift, Fourier invert
	to_align = np.copy(frame)
	fft = np.fft.fft2(to_align)
	fft_shift = fourier_shift(fft,shift=[-shift_y/calibration,-shift_x/calibration])
	fftm1 = np.fft.ifft2(fft_shift)

	# Export aligned frame
	#imwrite(filename,np.array(np.absolute(fftm1),dtype='uint16'))

	del to_align
	del fft
	del fft_shift
	
	return(np.array(np.absolute(fftm1),dtype='uint16'))
	
def plot_displacement(frames,x_disp,y_disp,filename):
	plt.figure(figsize=(4,3))
	plt.plot(frames,x_disp,label=r"$x$ displacement")
	plt.plot(frames,y_disp,label=r"$y$ displacement")
	plt.xlabel('frame')
	plt.ylabel(r'displacement [$\mu$m]')
	plt.legend()
	plt.tight_layout()
	plt.savefig(filename,dpi=300)
	plt.pause(3)
	plt.close()


img_paths = natsorted(glob(directory+"*.tif")) #load images to align
ref = imread(img_paths[0]) #set reference frame
data = pd.read_csv(trajectories) #read trajectory file

mean_displacement_x = []; mean_displacement_y = [];

times = np.unique(data[colF].to_numpy()) #extract times for which tracks are available
fullrange = list(np.linspace(min(times),max(times),int(max(times)-min(times))+1)) #all times
framediff = list(set(fullrange) - set(list(times))) #missing frames

# Find tracks existing in the reference frame
exists_at_ref = data[colF] == 0
tracks_ref = data[exists_at_ref].TRACK_ID.unique()

#initialize displacement fields to 0.0
mean_displacement_x_at_t = np.zeros(len(fullrange)); mean_displacement_y_at_t = np.zeros(len(fullrange));

#Compute the mean displacement of all tracks and filter out outliers
tracklist = np.unique(data[colTID].to_numpy())
confinement_number = []
for tid in tracklist:
    trackid = data[colTID] == tid
    x = data[trackid][colX].to_numpy()   #x associated to track 'tid'
    y = data[trackid][colY].to_numpy()   #y associated to track 'tid'
    frames = data[trackid][colF].to_numpy()
    c = mean_displacement(x,y,frames)
    confinement_number.append(c)

c_threshold = np.percentile(confinement_number,90)
print(f"Max accepted displacement = {c_threshold}")

#Loop over all frames and align when tracking information is available
for t in times:
	gc.collect()
	
	ref = 0
	print(f"Aligning frame {t} out of {max(times)}... ")

	exists_at_t = data[colF] == t # Subtable of spots existing at time t
	tracks_t = data[exists_at_t].TRACK_ID.unique() # Get tracks associated to those spots
	track_intersect = np.intersect1d(tracks_ref,tracks_t) # Find intersection of tracks between ref and t
	
	#Routine to align frames with no intersect to the last frame that was aligned, by chain rule
	if len(track_intersect)<1:
		s=1
		while len(track_intersect)==0:
			exists_at_new_ref = data[colF] == int(t-s)
			tracks_new_ref = data[exists_at_new_ref].TRACK_ID.unique()
			#print(tracks_new_ref,tracks_t)
			track_intersect = np.intersect1d(tracks_new_ref,tracks_t)
			#print(t,s,t-s,track_intersect)
			s+=1
			
			if s==20:
				print("No reference time could be found... The algorithm will not function properly... ")
				break
		ref = int(t-s+1)
		print("New reference time = ",ref)

	# Use track matches to compute the drift
	dx = []; dy = []; 
	for tid in track_intersect:
		track_at_t = data[(data[colTID]==tid) & (data[colF]==t)]
		track_at_ref = data[(data[colTID]==tid) & (data[colF]==ref)]
		c_value = confinement_number[int(np.where(tracklist==tid)[0])]

		xt = track_at_t[colX].to_numpy()[0]
		yt = track_at_t[colY].to_numpy()[0]
		if ref==0:
			x0 = track_at_ref[colX].to_numpy()[0]
			y0 = track_at_ref[colY].to_numpy()[0]
		else:
			#print(ref)
			#print(len(mean_displacement_x_at_t))
			x0 = track_at_ref[colX].to_numpy()[0] - mean_displacement_x_at_t[int(np.where(times==ref)[0])]
			y0 = track_at_ref[colY].to_numpy()[0] - mean_displacement_y_at_t[int(np.where(times==ref)[0])]
		if c_value<c_threshold:
			dx.append(xt - x0)
			dy.append(yt - y0)
	
	if len(dx)>1:
		mean_dx = np.mean(dx); mean_dy = np.mean(dy);
		mean_displacement_x_at_t[int(t)] = mean_dx
		mean_displacement_y_at_t[int(t)] = mean_dy	
		padded_t = str(t).zfill(4)
		img_t = imread([s for s in img_paths if padded_t in s][0])
		fftm1 = fourier_shift_frame(img_t,mean_dx,mean_dy,PxToUm)
		imwrite(f"{output_dir}aligned/out_{padded_t}.tif",fftm1,dtype='uint16')
		
		del img_t
		del fftm1
		
	else:
		framediff = np.append(framediff,t)
		framediff = np.sort(framediff)

#Plot the displacement field before the interpolation pass
plot_displacement(np.array(fullrange),mean_displacement_x_at_t,mean_displacement_y_at_t,output_dir+"displacement_profile.png")
	
# END OF FIRST PASS #####
print("Interpolating the shift of the frames for which we have no information...")
print("Frames ",framediff," are missing from the full range of frames...")

ref=0 #reference frame is the first		
slices,framediff = check_consecutive(framediff) #check if the missing frames are consecutive
unique_groups = np.unique(slices) #group by consecutiveness

for k in range(len(unique_groups)):
	disp_x = np.delete(mean_displacement_x_at_t, np.array(framediff,dtype=int)) #prep the displacement arrays to have the same shape as "times"
	disp_y = np.delete(mean_displacement_y_at_t, np.array(framediff,dtype=int))
	lower_bound = np.amin(framediff[slices==unique_groups[k]]) #identify the bounds
	upper_bound = np.amax(framediff[slices==unique_groups[k]])
	print(f"Processing chunk {k} lower bound = {lower_bound}; upper_bound = {upper_bound}...")
	loclb = int(np.where(times==(lower_bound-1))[0]) #find the position of the nearest neighbors to the bounds in "times"
	locub = int(np.where(times==(upper_bound+1))[0])
	nbrpoints = 10 #define number of points taken on both sides for the interpolation
	lower_x = times[loclb-nbrpoints:(loclb+1)]; upper_x = times[(locub):locub+nbrpoints]
	interpolate_x = np.concatenate([lower_x,upper_x])
	lower_y1 = disp_x[loclb-nbrpoints:(loclb+1)]; upper_y1 = disp_x[(locub):locub+nbrpoints]
	interpolate_y1 = np.concatenate([lower_y1,upper_y1])
	lower_y2 = disp_y[loclb-nbrpoints:(loclb+1)]; upper_y2 = disp_y[(locub):locub+nbrpoints]
	interpolate_y2 = np.concatenate([lower_y2,upper_y2])
		
	interp_disp_x = interp1d(interpolate_x, interpolate_y1,fill_value="extrapolate")
	interp_disp_y = interp1d(interpolate_x, interpolate_y2,fill_value="extrapolate")

	x_to_interpolate = framediff[slices==unique_groups[k]]
	frames_to_modify = np.array(framediff[slices==unique_groups[k]],dtype=int)

	mean_displacement_x_at_t[frames_to_modify] = np.array([interp_disp_x(x) for x in x_to_interpolate])
	mean_displacement_y_at_t[frames_to_modify] = np.array([interp_disp_y(x) for x in x_to_interpolate])

#Apply interpolated alignment to those frames
for k in range(len(framediff)):
	target_frame = int(framediff[k])
	mean_dx = disp_x[target_frame]; mean_dy = disp_y[target_frame];
	padded_t = str(target_frame).zfill(4)
	img_t = imread([s for s in img_paths if padded_t in s][0])
	fftm1 = fourier_shift_frame(img_t,mean_dx,mean_dy,PxToUm)
	imwrite(f"{output_dir}aligned/out_{padded_t}.tif",fftm1,dtype='uint16')
	

#Plot the displacement field after the interpolation
plot_displacement(np.array(fullrange),mean_displacement_x_at_t,mean_displacement_y_at_t,output_dir+"displacement_profile_corrected.png")

np.save(output_dir+"dispx_interp.npy",mean_displacement_x_at_t)
np.save(output_dir+"dispy_interp.npy",mean_displacement_y_at_t)

	
		
		
	

		
	


		
	

  
