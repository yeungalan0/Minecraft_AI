
import h5py
import numpy as np
import Image
from game_globals import *

folder = "datasets/"
hdf5_dataset_filename = folder + 'tmp_training_dataset.hdf5'
txt_dataset_filename = folder + 'tmp_training_dataset.txt'

#input_array = np.array(Image.open('screenshots/test.png'), dtype=np.float64)
input_array = np.ones((1, FEATURE_VECTOR_SIZE), dtype=np.float64)

# Scale the grayscale pixel values because caffe won't do it for the inputs only
input_array *= (1/255.)
input_array = input_array[:, np.newaxis, np.newaxis, :]

# Make a fake target vector with 64 values to test the network
# Make 'em all 1.0
# Set caffe's 4-D array to have the 64 values in the "width" (last) dimension
label_array = np.ones((1, OUTPUT_VECTOR_SIZE), dtype=np.float64)
label_array = label_array[:, np.newaxis, np.newaxis, :]

#print(input_array.shape)
#print(label_array.shape)

# Write out the inputs and targets in hdf5 format
f = h5py.File(hdf5_dataset_filename, 'w')
f.create_dataset('data', (1, 1, 1, FEATURE_VECTOR_SIZE), data=input_array, dtype='f8')
f.create_dataset('label', (1, 1, 1, OUTPUT_VECTOR_SIZE), data=label_array, dtype='f8')
f.close()

# Write the text file that points to the hdf5 binary data file (caffe requires this)
# This (the textfile) is the filename to include in the .prototxt file
t = open(txt_dataset_filename, 'w')
t.write("%s\n" % hdf5_dataset_filename)
t.close()

print("new temp hdf5 dataset created in %s" % hdf5_dataset_filename)