import os
import csv
import argparse
import numpy as np
import scipy.misc
import imageio


w, h = 48, 48
image = np.zeros((h, w), dtype=np.uint8)
id = 1

with open("fer2013/fer2013.csv") as csvfile:
    datareader = csv.reader(csvfile, delimiter=',')
    next(datareader, None)

    for row in datareader:

        usage = row[2]
        pixels = row[1].split()
        pixels_array = np.asarray(pixels, dtype=np.int)
        emotion = row[0]
        image = pixels_array.reshape(w, h)
        stacked_image = np.dstack((image,) * 3)

        image_folder = os.path.join("images", usage)

        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        image_file = os.path.join(image_folder, str(id) + '_' + emotion + '.jpg')
        imageio.imsave(image_file, stacked_image)
        id += 1

        if id % 100 == 0:
            print('Processed {} images'.format(id))

print("Finished processing {} images".format(id))
