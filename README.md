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
