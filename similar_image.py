import os
from PIL import Image
import imagehash

cutoff = 5  # maximum bits that could be different between the hashes.
count = 0
for i in range(1000):
    fileList = os.listdir('/home/alhelal/Thesis/Data/unlabeled_48x48_images')
    os.chdir('/home/alhelal/Thesis/Data/unlabeled_48x48_images')
    if len(fileList) <= i:
        break
    file1 = fileList[i]
    hash0 = imagehash.average_hash(Image.open(file1))
    for file2 in fileList:
        if file1 == file2:
            continue
        hash1 = imagehash.average_hash(Image.open(file2))
        if hash0 - hash1 < cutoff:
            count += 1
            print(f'{count} : {file2}')
            os.rename(file2, '/home/alhelal/Thesis/Data/similar_image_48x48/'+file2)
