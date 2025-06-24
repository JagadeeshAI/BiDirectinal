import mxnet as mx
import cv2
import os

rec_path = '/media/jag/volD/BID_DATA/faces_vgg_112x112/train.rec'
idx_path = '/media/jag/volD/BID_DATA/faces_vgg_112x112/train.idx'
output_dir = 'extracted_train_images'
os.makedirs(output_dir, exist_ok=True)

record = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')
keys = list(record.keys)  # Get all available keys (indices)

for i in keys:
    header, img = mx.recordio.unpack(record.read_idx(i))
    img_arr = mx.image.imdecode(img).asnumpy()  # (H, W, C) in BGR
    img_name = f"{i}.jpg"
    cv2.imwrite(os.path.join(output_dir, img_name), img_arr)