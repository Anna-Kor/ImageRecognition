import matplotlib.pyplot as pyplot
import numpy

def show(image):
    image = image / 2 + 0.5
    npimg = image.numpy()
    pyplot.imshow(numpy.transpose(npimg, (1, 2, 0)))
    pyplot.show()
