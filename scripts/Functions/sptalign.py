import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from natsort import natsorted
from glob import glob

from tifffile import imread,imwrite
from PIL import Image
from scipy.ndimage.interpolation import shift
from scipy.ndimage import fourier_shift
from scipy.ndimage import shift
from scipy.interpolate import interp1d

import gc
import os
import shutil
from numba import njit
import time
import os,sys,platform
import glob
import shutil
from datetime import datetime
import itertools
import datetime, time
import json


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

