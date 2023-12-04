import os


def listToConvenientStr(lst):
    finalStr = ""
    for element in lst:
        finalStr += str(element) + " "
    return finalStr + "\n"
    

    
import numpy as np
import idx2numpy
import matplotlib.pyplot as plt

imagefile = 'train-images.idx3-ubyte'
imagearray = idx2numpy.convert_from_file(imagefile)

labelfile = 'train-labels.idx1-ubyte'
labelarray = idx2numpy.convert_from_file(labelfile)

np.set_printoptions(linewidth=200)


#ind = 2136
#plt.imshow(imagearray[ind], cmap=plt.cm.binary)
#print(imagearray[ind])

#os.system("rm -r readableTrainingData")
#os.system("mkdir readableTrainingData")
#for i in range(1, 61):
#    file = open("readableTrainingData/trainingData"+str(i), "w")
#    for image in imagearray[(i-1)*1000:i*1000]:
#        imagelist = image.tolist()
#        flat_image_list = [item for sublist in imagelist for item in sublist]
#        file.write(listToConvenientStr(flat_image_list))
#    file.close()


testimagefile = 't10k-images.idx3-ubyte'
testimagearray = idx2numpy.convert_from_file(testimagefile)

testlabelfile = 't10k-labels.idx1-ubyte'
testlabelarray = idx2numpy.convert_from_file(testlabelfile)

plt.imshow(testimagearray[3], cmap=plt.cm.binary)
print(testimagearray[3])
plt.show()
file = open("readableTestingData/testingAnswers", "w")
file.write(listToConvenientStr(testlabelarray.tolist()))

file = open("readableTestingData/testingData", "w")
for image in testimagearray:
    imagelist = image.tolist()
    flat_image_list = [item for sublist in imagelist for item in sublist]
    file.write(listToConvenientStr(flat_image_list))
file.close()

