<img src="https://www.univ-amu.fr/system/files/2021-01/DIRCOM-Logo_AMU_CMJN.png" alt="drawing" width="150"/> &nbsp;&nbsp; <img src="https://centuri-livingsystems.org/wp-content/uploads/2018/02/logo-CENTURI-horizontal-azur-retina.png" width="150"/>


# Badges 

# Brief description and links?

# Overview


# System requirements

## Hardware requirements

Tested on Ubuntu and Windows.

## Software requirements

This package was developed in Python 3.9.

# Package installation

You may clone the repository to your local machine (or download/extract the `zip` file), then install the python package `spt_align`:

``` bash
    # creates "SPTAlign" folder
    git clone git://github.com/remyeltorro/SPTAlign.git
    cd SPTAlign

    # install the package in editable/development mode
    pip install -r requirements.txt
    pip install -e .
```

# Quick use

Open a Python shell or a Jupyter Notebook. For detailed a detailed explanation about the steps of the method, check the tutorial notebook provided in the `notebook` folder of the repository. 

The movie stack must be stored as an image sequence in `.tif` format in a folder.

``` python
	df = load_tracks("path/to/track/csv")
	frames = locate_frames("path/to/frames/folder")
	timeline = estimate_timeline(frames, df)
	PxToUm = 1 # calibration used in TrackMate
	output_dir = "output"

	displacement = estimate_displacement(df, timeline, reference_time=0, nbr_tracks_threshold=30)
	displacement = fill_by_shifting_reference_time(df, timeline, displacement, nbr_tracks_threshold=30, from_origin=True)
	stack = align_frames(frames, displacement, PxToUm=PxToUm, output_dir=output_dir,return_stack=True)
```


# How to cite?

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