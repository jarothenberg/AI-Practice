# Import the necessary libraries
from tkinter.filedialog import askopenfilename

from PIL import Image
from numpy import asarray


def visualizeNpy(numpydata):
    for i in range(len(numpydata)):
        line = ""
        for j in range(len(numpydata[i])):
            val = numpydata[i][j][0]
            if val < 255:
                line += "X"
            else:
                line += " "
        print(line)


def visualizeArry(dataArray):
    count = 0
    for i in range(28):
        line = ""
        for j in range(28):
            val = dataArray[count]
            count += 1
            if val > 0:
                line += "X"
            else:
                line += " "
        print(line)


def getData():
    imgFile = askopenfilename(initialdir="C:/Users/Jacka/Desktop")
    img = Image.open(imgFile)
    numpydata = asarray(img)

    dataArray = []
    for i in range(len(numpydata)):
        for j in range(len(numpydata[i])):
            dataArray.append(1 - numpydata[i][j][0] / 255)

    return dataArray
