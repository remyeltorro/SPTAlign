# Sub-pixel image registration using single particle tracking
A pipeline that combines SPT and Fourier shift to perform a sub-pixel registration on traction force microscopy and other fluorescence movies.

<div align="center">
  
Original TFM stack             |  Drift corrected stack
:-------------------------:|:-------------------------:
![](_figures/drift.gif)  |  ![](_figures/drift_corrected.gif)
  
</div>

## Dependencies

To clone the repository, go to the folder of your choice and run in a terminal:

```bash
git clone https://github.com/remy13127/subpixel_registration_spt registration
```

In order to install the required python packages:

```bash
pip install -r requirements.txt
```

## Acquisition of tracks

The reference tracks are acquired using TrackMate, but any CSV table containing <img src="https://render.githubusercontent.com/render/math?math=x, \ y, \ t"> position and a track ID should work. TrackMate’s LoGdetector applies a Laplacian of Gaussian filter, giving a strong response for gaussian-like spots of radius <img src="https://render.githubusercontent.com/render/math?math=\sqrt{2}\sigma">, which must be tuned to the average size of the beads.  For each object, the detector applies a Gaussian fit, which allows for the determination of a subpixel centroid. Once the objects of interest are detected, TrackMate’s LAP particle linking algorithm, which combines a nearest  neighbour  penalty  function  with  similitude  criteria (intensity, shape, size...).  Tuning parameters such as the maximum expected displacement of a spot between one frame and the next and the maximum number of missed detections within a trajectory, we can obtain high quality tracks.

## Alignment

```bash
python align.py
```

## Results

<div align="center">
  
Displacement field before interpolation             |  Interpolated displacement
:-------------------------:|:-------------------------:
![](output/displacement_profile.png)  |  ![](output/displacement_profile_corrected.png)
  
</div>
