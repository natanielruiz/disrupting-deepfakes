import os
import shutil

output_dir = './images'

os.mkdir(output_dir)

for root, dirs, files in os.walk('./my-processed-dataset'):
    for filename in files:
        if 'jpg' in filename:
            img_name = root.split('/')[-1].split('_')[0] + '.jpg'
            shutil.copy2(os.path.join(root, filename), os.path.join(output_dir, img_name))