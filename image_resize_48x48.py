import os, numpy, cv2, re
numpy.set_printoptions(threshold=numpy.inf)
filelist = os.listdir('../Data/test/angry')
os.chdir('../Data/test/angry')
for f in filelist:
    print(f)
    f1 = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    f2 = cv2.resize(f1, (48, 48), cv2.IMREAD_GRAYSCALE)
    cv2.imwrite('../../../Data/modified_48x48_train/test/angry/'+f, f2)