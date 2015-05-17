import numpy as np
from aplpy import FITSFigure
import matplotlib.pyplot as pl
from matplotlib.patches import Circle

im  = np.random.randn(100,100)

gc = FITSFigure(im)
gc.show_grayscale()
gc.show_circles(20, 20, 10, alpha = 0.3, edgecolor = 'y', linewidth = 10)
gc.show_circles(40,40,10, facecolor = 'y', alpha = 0.3, linewidth = 10)
gc.show_circles(60,60,10, facecolor = 'y', alpha = 0.3, edgecolor = 'none')
gc.save('aplpy_circles.png')

pl.figure(2)
pl.imshow(im, interpolation = 'nearest', origin = 1, cmap = 'gray')
circle = Circle((20,20), 10,alpha=0.3, edgecolor = 'y',facecolor = 'none', linewidth = 10)
pl.gca().add_artist(circle)
circle = Circle((40,40), 10,alpha=0.3, facecolor = 'y', linewidth = 10)
pl.gca().add_artist(circle)
circle = Circle((60,60), 10,alpha=0.3, facecolor = 'y', linewidth = 0)
pl.gca().add_artist(circle)
pl.savefig('mpl_circles.png')


import matplotlib
import aplpy

assert matplotlib.__version__ == '1.3.0'
assert aplpy.__version__ == '0.9.9'
