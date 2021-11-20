import os, numpy, cv2, re
numpy.set_printoptions(threshold=numpy.inf)
filelist = os.listdir('Traine_DATA')
os.chdir('Traine_DATA')
for f in filelist:
    print(f)
    f1 = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    f2 = cv2.resize(f1, (48, 48), cv2.IMREAD_GRAYSCALE)
    cv2.imwrite('../Modified_Traine_DATA/'+f, f2)
    if(re.search("angry", f, re.IGNORECASE)):
        print('0, '+str(f2.flatten())+', Training', file=open('../Array_Data/data.csv','a'))
    elif(re.search("happy", f, re.IGNORECASE)):
        print('3, '+str(f2.flatten())+', Training', file=open('../Array_Data/data.csv','a'))
    elif(re.search("sad", f, re.IGNORECASE)):
        print('4, '+str(f2.flatten())+', Training', file=open('../Array_Data/data.csv','a'))
    elif(re.search("Neutral|normal", f, re.IGNORECASE)):
        print('6, '+str(f2.flatten())+', Training', file=open('../Array_Data/data.csv','a'))