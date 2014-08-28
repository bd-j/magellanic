Installation
=====

Get the Magellanic code
  1. ``git clone https://github.com/bd-j/magellanic.git``
  
You will only be concerned with stuff in the ``sfhs/`` subdirectory of the
magellanic repo.  The LF files go in the ``lf_data/`` subdirectory.  You
should make a ``sfhs/composite_lfs/`` directory.  Then, the key module
is ``predicted_total.py``.  This generates composite LFs optionally
for multiple bands and agb_dust parameters, for both clouds.  The
important bit is to modify the ``lfst``, ``outst``, and ``basti``
variables in the ``__main__`` branch to correspond to the LF files you
want to use, output name, and whether the ischrones are Basti or not.
Then ``python predicted_total.py`` should put some lf text files in
the ``composite_lfs`` directory .


Install FSPS.
 1. ``svn checkout http://fsps.googlecode.com/svn/trunk/ fsps``
 2. ``export SPS_HOME="/Path/to/fsps/directory/"``
 3. ``cd fsps/src/``
 4. ``make all``
 5. smoke test by running ``./simple.exe``

Install python-FSPS 
 1. ``pip install fsps``
 2. You'll probably have to add your revision of FSPS to the accepted
    list.  Alternatively, just comment out the lines in
    ``__init__.py`` that start
    ``if not accepted: raise ImportError(Your FSPS revision, {0}, is not known to work with``
 
Optional. Install sedpy.  
Sedpy does all the filter projections and has attenuation functions.
If you don't install it, make sure you set ``filters=None`` in the
main branch of ``predicted_total.py``
  1. ``git clone https://github.com/bd-j/sedpy.git``
  2. ``cd sedpy``
  3. ``[sudo] python setup.py develop``
  4. copy over any filters you want to use that aren't already in
      sedpy/data/filters. The filter file format is the same as
      k_correct, so you can nick any of those.
   
