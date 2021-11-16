import mat73
from matplotlib import pyplot as plot

mat = mat73.loadmat('/home/georg_mosh/Documents/gtsegs_ijcv.mat')
plot.imshow(mat['value']['img'][0])
plot.show()