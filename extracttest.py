import pickle
import numpy as np
import cv2
import os
import mxnet as mx

bin_path = '/media/jag/volD/BID_DATA/faces_vgg_112x112/cfp_ff.bin'
output_dir = 'extracted_cfp_ff'
os.makedirs(output_dir, exist_ok=True)

with open(bin_path, 'rb') as f:
    bins, issame_list = pickle.load(f, encoding='bytes')

print(f"bins type: {type(bins)}, len: {len(bins)}")
for group_idx, group in enumerate(bins):
    print(f"group {group_idx} type: {type(group)}, len: {len(group)}")
    for idx, img_bin in enumerate(group):
        img = mx.image.imdecode(img_bin)
        if img is None:
            print(f"Failed to decode image at group {group_idx}, idx {idx}")
            continue
        img_arr = img.asnumpy()
        cv2.imwrite(os.path.join(output_dir, f'{group_idx}_{idx}.jpg'), img_arr)