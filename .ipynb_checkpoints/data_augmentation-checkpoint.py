import tensorflow
import imgaug as ia
from imgaug import augmenters as iaa
import imageio
import numpy as np
import skimage
import cv2
# import PIL
from PIL import Image
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join

n_classes = 1+6

images_path = "/scratch2/peilun/SpaceXYZ/originalImages"
labels_path = "/scratch2/peilun/SpaceXYZ/GT"
image_out_path = images_path + "_new_augmented/"
label_out_path = labels_path + "_new_augmented/"
show_on_image = '/scratch2/peilun/SpaceXYZ/o2/'
image_files = [f for f in listdir(images_path) if isfile(join(images_path, f))]
label_files = [f for f in listdir(labels_path) if isfile(join(labels_path, f))]

if '.DS_Store' in image_files:
    image_files.remove('.DS_Store')
if '.DS_Store' in label_files:
    label_files.remove('.DS_Store')
image_files.sort()
label_files.sort()
n_items = len(image_files)

## PROCESSING

for i in range(n_items):
# for i in range(1):
    print("Processing image ", i, ": ", image_files[i])
    image = cv2.imread(join(images_path, image_files[i]))
    label = cv2.imread(join(labels_path, label_files[i]))
    segmap = label.astype(np.int32)
    segmap = segmap[:,:,0]
    segmap = ia.SegmentationMapOnImage(segmap, shape=image.shape, nb_classes=n_classes)
    
    # transformation types
    seq = iaa.Sequential([
    iaa.Fliplr(0.5), 
    iaa.Flipud(0.5),      
    iaa.Affine(rotate=[90, 180, 270], 
               scale={"x": (0.75, 1.25), "y": (0.75, 1.25)}, 
               translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, mode='constant', cval=255)], random_order=True)
    
    for j in range(10):
        print("  Transformation ", j)
        seq_det = seq.to_deterministic()
        img = seq_det.augment_image(image)
#         img[img == 0] = 255
        seg_out = seq_det.augment_segmentation_maps([segmap])[0].get_arr_int()
#         print(seg_out.shape)
        seg_overlay = seq_det.augment_segmentation_maps([segmap])[0].draw_on_image(img)
        
        image_out = image_out_path + image_files[i][:-4] + '_' + str(j) + '.jpg'
        label_out = label_out_path + image_files[i][:-4] + '_' + str(j) + '.png'
        overlay_out = show_on_image + image_files[i][:-4] + '_' + str(j) + '.png'
        
        # save images
        im = Image.fromarray(img.astype(np.uint8))
        im.save(image_out)
        
        # save segmentations
        sg = Image.fromarray(seg_out.astype(np.uint8))
        sg.save(label_out)
        
        # save overlay for verification
        sv = Image.fromarray(seg_overlay.astype(np.uint8))
        sv.save(overlay_out)

