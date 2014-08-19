import matplotlib.pyplot as plt

def drawseq(images, fps=4, **kwargs):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.show()
	for i in range(len(images)):
		ax.imshow(images[i], **kwargs)
		plt.pause(1.0/fps)
		plt.draw()


def blink(images, fps=4, times=5, **kwargs):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.show()
	i = 0
	j = 0
	while j < times:
		ax.cla()
		ax.imshow(images[i], **kwargs)
		plt.pause(1.0/fps)
		plt.draw()
		i += 1
		if i >= len(images):
			i = 0
			j += 1
			plt.pause(1.0/fps)
