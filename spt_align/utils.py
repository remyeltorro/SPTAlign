import matplotlib.pyplot as plt

def plot_displacement(timeline, displacement, save_path=None, ax=None, auto_close=False):
	
	"""
	Plots the drift along the X and Y axes of the image based on displacement data from trajectories.

	This function visualizes the X and Y displacement of tracked objects over time. It allows for 
	optional saving of the plot and customization of the plotting axis.

	Parameters
	----------
	timeline : 1D numpy array
		Array representing the time frames associated with each displacement value.
	displacement : 2D numpy array
		Array of shape (n_frames, 2) containing X and Y displacements. Each row represents a frame, 
		and the two columns correspond to X and Y displacements in micrometers (µm).
	save_path : str, optional
		Path to save the plot as an image file. If None, the plot is not saved. Default is None.
	ax : matplotlib.axes.Axes, optional
		Matplotlib axis object to plot on. If None, a new figure and axis are created. Default is None.
	auto_close : bool, optional
		If True, the plot window will automatically close after displaying. Default is False.

	Returns
	-------
	matplotlib.figure.Figure or None
		Returns the matplotlib figure object if `ax` is None. If `ax` is provided, it returns None.

	Notes
	-----
	- The function plots the X and Y displacement over the given timeline.
	- If `save_path` is specified, the plot is saved as an image file at the given path.
	- If `auto_close` is set to True, the plot window will close automatically after a short pause.

	Examples
	--------
	>>> timeline = np.arange(0, 100, 1)
	>>> displacement = np.random.randn(100, 2)  # Random displacements
	>>> plot_displacement(timeline, displacement, save_path='displacement_plot.png')

	"""
	
	if ax is None:
		fig,ax = plt.subplots(1,1,figsize=(4,3))
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
		
	ax.plot(timeline, displacement[:,0], label=r"$x$ displacement")
	ax.plot(timeline, displacement[:,1], label=r"$y$ displacement")
	ax.set_xlabel('timeline [frame]')
	ax.set_ylabel(r'displacement [$\mu$m]')
	ax.set_xlim(min(timeline),max(timeline))

	ax.legend()
	plt.tight_layout()
	if save_path is not None:
		plt.savefig(filename,dpi=300)
	if not auto_close:
		plt.show()
	else:
		plt.pause(1)
		plt.close()
