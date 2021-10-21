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


directory = "../data/images/"
trajectories = "../data/trajectories.csv"
output_dir = "../output/"
PxToUm = 0.1

def confinement_ratio(x,y):
	s=0
	for p in range(0,len(x)-1):
		s+= np.sqrt((x[p+1]-x[p])**2+(y[p+1]-y[p])**2)
	return(s/len(x))

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


img_paths = natsorted(glob(directory+"*.tif"))

ref = imread(img_paths[0])
data = pd.read_csv(trajectories)

mean_displacement_x = []; mean_displacement_y = [];

times = np.unique(data["FRAME"].to_numpy())
fullrange = list(np.linspace(min(times),max(times),int(max(times)-min(times))+1))
framediff = list(set(fullrange) - set(list(times)))

# Find tracks existing in the reference frame
exists_at_ref = data["FRAME"] == 0
tracks_ref = data[exists_at_ref].TRACK_ID.unique()

mean_displacement_x_at_t = np.zeros(len(fullrange)); mean_displacement_y_at_t = np.zeros(len(fullrange));

tracklist = np.unique(data["TRACK_ID"].to_numpy())
confinement_number = []
for tid in tracklist:
    trackid = data["TRACK_ID"] == tid
    x = data[trackid]["POSITION_X"].to_numpy()   #x associated to track 'tid'
    y = data[trackid]["POSITION_Y"].to_numpy()   #y associated to track 'tid'
    frames = data[trackid]["FRAME"].to_numpy()
    c = confinement_ratio(x,y)
    confinement_number.append(c)
 
c_threshold = np.percentile(confinement_number,90)
print("Max accepted displacement = ",c_threshold)
for t in times:
	gc.collect()
	
	ref = 0
	print("Aligning frame ",t," out of ",max(times),"... ")
	# Subtable of spots existing at time t
	exists_at_t = data["FRAME"] == t
	# Get tracks associated to those spots
	tracks_t = data[exists_at_t].TRACK_ID.unique()
	# Find intersection of tracks between ref and t
	track_intersect = np.intersect1d(tracks_ref,tracks_t)


	
	#Routine to align frames with no intersect to the last frame that was aligned
	if len(track_intersect)<1:
		s=1
		while len(track_intersect)==0:
			exists_at_new_ref = data["FRAME"] == int(t-s)
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

	# Use track match to compute a drift
	dx = []; dy = []; 
	for tid in track_intersect:
		track_at_t = data[(data["TRACK_ID"]==tid) & (data["FRAME"]==t)]
		track_at_ref = data[(data["TRACK_ID"]==tid) & (data["FRAME"]==ref)]
		c_value = confinement_number[int(np.where(tracklist==tid)[0])]

		xt = track_at_t["POSITION_X"].to_numpy()[0]
		yt = track_at_t["POSITION_Y"].to_numpy()[0]
		if ref==0:
			x0 = track_at_ref["POSITION_X"].to_numpy()[0]
			y0 = track_at_ref["POSITION_Y"].to_numpy()[0]
		else:
			#print(ref)
			#print(len(mean_displacement_x_at_t))
			x0 = track_at_ref["POSITION_X"].to_numpy()[0] - mean_displacement_x_at_t[int(np.where(times==ref)[0])]
			y0 = track_at_ref["POSITION_Y"].to_numpy()[0] - mean_displacement_y_at_t[int(np.where(times==ref)[0])]
		if c_value<c_threshold:
			dx.append(xt - x0)
			dy.append(yt - y0)
	
	if len(dx)>1:
		mean_dx = np.mean(dx); mean_dy = np.mean(dy);
		mean_displacement_x_at_t[int(t)] = mean_dx
		mean_displacement_y_at_t[int(t)] = mean_dy	
		
		padded_t = str(t).zfill(4)
		padded_ref = str(ref).zfill(4)
		img_t = imread([s for s in img_paths if padded_t in s][0])
		img_ref = imread([s for s in img_paths if padded_ref in s][0])
		
		to_align = np.copy(img_t)
		input_ = np.fft.fft2(to_align)
		out = fourier_shift(input_,shift=[-mean_dy/PxToUm,-mean_dx/PxToUm])
		out = np.fft.ifft2(out)

		imwrite(output_dir+"aligned/out_"+padded_t+".tif",np.array(np.absolute(out),dtype='uint16'))
	
		del to_align
		del input_
		del img_ref
		del img_t
		del out
	else:
		framediff = np.append(framediff,t)
		framediff = np.sort(framediff)

plt.plot(np.array(fullrange),mean_displacement_x_at_t)
plt.plot(np.array(fullrange),mean_displacement_y_at_t)
plt.xlabel('frame')
plt.ylabel(r'displacement [$\mu$m]')
plt.savefig(output_dir+"displacement_profile.png",dpi=300)
plt.pause(5)
plt.close()
	
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
	print("Processing chunk ",k," lower bound = ",lower_bound,"; upper_bound = ",upper_bound,"...")
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
	ref=0
	target_frame = int(framediff[k])
	mean_dx = disp_x[target_frame]; mean_dy = disp_y[target_frame];

	padded_t = str(target_frame).zfill(4)
	padded_ref = str(ref).zfill(4)

	img_t = imread([s for s in img_paths if padded_t in s][0])
	img_ref = imread([s for s in img_paths if padded_ref in s][0])

	to_align = np.copy(img_t)
	input_ = np.fft.fft2(to_align)
	out = fourier_shift(input_,shift=[-mean_dy/PxToUm,-mean_dx/PxToUm])
	out = np.fft.ifft2(out)

	imwrite(output_dir+"aligned/out_"+padded_t+".tif",np.array(np.absolute(out),dtype='uint16'))
		

plt.plot(fullrange,mean_displacement_x_at_t)
plt.plot(fullrange,mean_displacement_y_at_t)
plt.xlabel('frame')
plt.ylabel(r'displacement [$\mu$m]')
plt.savefig(output_dir+"displacement_profile_corrected.png",dpi=300)
plt.pause(5)
plt.close()

np.save(output_dir+"dispx_interp.npy",mean_displacement_x_at_t)
np.save(output_dir+"dispy_interp.npy",mean_displacement_y_at_t)

	
		
		
	

		
	


		
	

  
