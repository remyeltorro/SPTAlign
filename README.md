<img src="https://www.univ-amu.fr/system/files/2021-01/DIRCOM-Logo_AMU_CMJN.png" alt="drawing" width="150"/> &nbsp;&nbsp; <img src="https://centuri-livingsystems.org/wp-content/uploads/2018/02/logo-CENTURI-horizontal-azur-retina.png" width="150"/>

![ico17](https://github.com/remyeltorro/SPTAlign/actions/workflows/test.yml/badge.svg)
![GitHub repo size](https://img.shields.io/github/repo-size/remyeltorro/SPTAlign)
![ico2](https://img.shields.io/github/forks/remyeltorro/SPTAlign?link=https%3A%2F%2Fgithub.com%2Fremyeltorro%2FSPTAlign%2Fforks)
![ico3](https://img.shields.io/github/stars/remyeltorro/SPTAlign?link=https%3A%2F%2Fgithub.com%2Fremyeltorro%2FSPTAlign%2Fstargazers)


`spt_align` is a python package and register traction force microscopy (TFM) stacks using a tracking table of the fluorescent beads. Go check the tutorial notebook for more information!

-   [Report a bug or request a new feature](https://github.com/remyeltorro/celldetective/issues/new/choose)


## System requirements

### Hardware requirements

The registration is done frame per frame, with a garbage collect at each loop to save memory. Therefore, the requirements are very low.

### Software requirements

To use the package, you must install Python, *e.g.* through
[Anaconda](https://www.anaconda.com/download). The `spt_align` package is routinely tested on both Ubuntu and Windows for Python versions between 3.7 and 3.12.


## Package installation

You may clone the repository to your local machine (or download/extract the `zip` file), then install the python package `spt_align`. 

``` bash
    # creates "SPTAlign" folder
    git clone git://github.com/remyeltorro/SPTAlign.git
    cd SPTAlign

    # install the package in editable/development mode
    pip install -r requirements.txt
    pip install -e .
```

## Quick use

Open a Python shell or a Jupyter Notebook. For detailed a detailed explanation about the steps of the method, check the tutorial notebook provided in the `notebook` folder of the repository. 

The movie stack must be stored as an image sequence in `.tif` format in a folder. Here the tracking is assumed to have been performed with TrackMate[^1].


``` python
	df = load_tracks("path/to/track/csv")
	frames = locate_frames("path/to/frames/folder")
	timeline = estimate_timeline(frames, df)
	PxToUm = 1 # calibration used in TrackMate
	output_dir = "output"

	displacement = estimate_displacement(df, timeline, reference_time=0, nbr_tracks_threshold=30)
	displacement = fill_by_shifting_reference_time(df, timeline, displacement, nbr_tracks_threshold=30, from_origin=True)
	align_frames(frames, displacement, PxToUm=PxToUm, output_dir=output_dir,return_stack=False)
	# registered stack created in folder "output/aligned"

```

<div align="center">
  
Displacement field before interpolation             |  Interpolated displacement | Re-tracking of aligned movie
:-------------------------:|:-------------------------:|:---------------------------------------:|
![](output/displacement_profile.png)  |  ![](output/displacement_profile_corrected.png) | ![](_figures/retracking.png)
  
</div>


<div align="center">
  
Original TFM stack             |  Drift corrected stack
:-------------------------:|:-------------------------:
![](_figures/drift.gif)  |  ![](_figures/drift_corrected.gif)
  
</div>


## How to cite?

If you use the notebook in your research, please cite the work for which it was developed (currently preprint):

``` raw
@article {Mustapha2022.02.11.480084,
	author = {Mustapha, Farah and Pelicot-Biarnes, Martine and Torro, Remy and Sengupta, Kheya and Puech, Pierre-henri},
	title = {Cellular forces during early spreading of T lymphocytes on ultra-soft substrates},
	elocation-id = {2022.02.11.480084},
	year = {2022},
	doi = {10.1101/2022.02.11.480084},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {T cells use forces to read out and act on the mechanical parameters of their microenvironment, which includes antigen presenting cells (APCs). Here we explore the early interaction of T cells with an APC-mimicking ultra-soft polymer gel exhibiting physiologically relevant stiffness in the range of 350-450 Pa. We quantify the dependence of cell spreading and stiffness on gel elasticity, and measure early time traction forces. We find that coating the surface with an antibody against the CD3 region of the TCR-complex elicits small but measurable gel deformation in the early recognition phase, which we quantify in terms of stress or energy. We show that the time evolution of the energy follows one of three distinct patterns: active fluctuation, intermittent, or sigmoidal signal. Addition of either anti-CD28 or anti-LFA1 has little impact on the total integrated energy or the maximum stress. However, the relative distribution of the energy patterns does depend on the additional ligands. Remarkably, the forces are centrifugal at very early times, and only later turn into classical in-ward pointing centripetal traction.},
	URL = {https://www.biorxiv.org/content/early/2022/02/11/2022.02.11.480084},
	eprint = {https://www.biorxiv.org/content/early/2022/02/11/2022.02.11.480084.full.pdf},
	journal = {bioRxiv}
}
```

## Bibliography

[^1]: Ershov, Dmitry, Minh-Son Phan, Joanna W. Pylvänäinen, Stéphane U. Rigaud, Laure Le Blanc, Arthur Charles-Orszag, James R. W. Conway, et al. “TrackMate 7: Integrating State-of-the-Art Segmentation Algorithms into Tracking Pipelines.” Nature Methods 19, no. 7 (July 2022): 829–32. https://doi.org/10.1038/s41592-022-01507-1.
