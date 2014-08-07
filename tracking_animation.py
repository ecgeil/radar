import trackblobs

import numpy as np
from PIL import Image
from skimage import feature
from skimage.morphology import watershed
from skimage import measure, filter
from scipy import ndimage
from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from glob import glob

reload(trackblobs)

files = glob('/Users/ethan/insight/nexrad/data/kmlb_grid/firstn/*.png' )
frames = []
storm_list = []
for f in files:
	im = Image.open(f)
	z = np.flipud(np.array(im))*75.0/255.0
	z1 = filter.gaussian_filter(1*z, sigma=1.1)
	im.close()
	frames.append(z1)
	storms = trackblobs.find_storms(z1)
	storm_list.append(storms)

fig = plt.figure()
ax = fig.add_subplot(111)
axim = ax.imshow(z1, origin='lower')

def init():
	pass


def animate(i):
	#ax.cla()
	ax.artists = []
	axim.set_data(frames[i])
	trackblobs.draw_storms(storm_list[i], ax=ax)


anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(files), interval=20, blit=False)

plt.show()