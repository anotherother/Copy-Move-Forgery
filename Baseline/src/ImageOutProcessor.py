import os
import datetime as dt
import cv2

class ImageOutProcessor:
    """
    Base class for writing and openning images
    (operation for addition image to vector, writing to file)
    """
    def __init__(self, file_name):
        self.vector = []
        self.images = []
        self.name = []
        self.image_name = []
        self.now = dt.datetime.now()
        file_name = file_name[0:len(file_name) - 4]
        self.foldername = "%s_%s_%s_%s_%s_%s_%s" % (
        file_name, self.now.day, self.now.month, self.now.year, self.now.hour, self.now.minute, self.now.second)
        self.PATH = 'output/%s' % (self.foldername)
        if not os.path.exists("output"):
            os.makedirs("output")
        os.mkdir(self.PATH)
        self.PATH += "/"

    def addObject(self, object1, name):
        self.vector.append(object1)
        self.name.append(name)

    def addImage(self, image, name):
        self.images.append(image)
        self.image_name.append(name)

    def printImage(self):
        len_of_images = len(self.images)
        for i in range(0, len_of_images):
            cv2.imwrite('%s/%s.png' % (self.PATH, self.image_name[i]), self.images[i])

    def printFile(self):
        if not os.path.exists("output"):
            os.makedirs("output")
        length_of_vector = len(self.vector)
        for i in range(0, length_of_vector):
            string = "%s/%s.txt" % (self.PATH, self.name[i])
            file = open(string, 'w+')
            file.write(str(self.vector[i]))

