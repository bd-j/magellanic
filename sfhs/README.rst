Installation
=====

1. Install FSPS.
   a. `svn checkout http://fsps.googlecode.com/svn/trunk/ fsps`
   b. `export SPS_HOME="/Path/to/fsps/directory/"`
   c. `cd fsps/src/`
   d. `make all`
   e. smoke test by running `./simple.exe`

2. Install python-FSPS (specifically my fork of it, which works with
   the newest FSPS version, R140)
   a. `git clone https://github.com/bd-j/python-fsps.git`
   b. `cd python-fsps`
   c. `[sudo] python setup.py build_fsps`
   d. `[sudo] python setup.py develop` (puts a link to this directory
      in your default site-packages directory so import can find it)

3. Install sedpy.  Sedpy does all the filter projections and has
   attenuation functions
   a. `git clone https://github.com/bd-j/sedpy.git`
   b. `cd sedpy`
   c. `[sudo] python setup.py develop`
   d. copy over any filters you want to use that aren't already in
      sedpy/data/filters. The filter file format is the same as
      k_correct, so you can nick any of those.
   
4. Download the relevant `scombine` modules to your working directory
   a. You will need `bursty_sps.py` and `sfh_utils.py` (available from
      https://github.com/bd-j/scombine , but I included them in this
      tarball for convenience)

5. Final touches.
   a. Downlaod the Harris and Zaritsky SFH files from
      http://djuma.as.arizona.edu/~dennis/mcsurvey/Data_Products.html
      or use the provided files.
   b. change the path to the Harris and Zaritsky SFH files in
   `hzutils.py`, choose your cloud and filters in
   `predicted_image.py`, and run that script.

6. Profit!  last step will produce pngs, but if you want the images,
   you can write the `im` varibale as a fits image.  The WCS is a
   little uncertain, the Harris and Zaritsky files given a region
   coordinate, but I'm not sure if that's for the center of the region
   or what.
