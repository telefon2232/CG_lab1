# Importing Image from PIL package
from PIL import Image
import numpy
# creating a image object
im = Image.open(r"cat.jpg")
px = im.load()
im1 = Image.open(r"cat.jpg")
px1 = im.load()
import math


class Inversion:
    def __init__(self):

        self.width, self.height = im.size
        im.show()

    def start(self):

        for i in range(self.height):
            for j in range(self.height):
                px[i,j] = (255-px[i,j][0],255-px[i,j][1],255-px[i,j][2])


    def show(self):

        im.show()

class GrayScaleFilter(Inversion):
    def start(self):
        for i in range(self.height):
            for j in range(self.height):
                px[i,j]=(int(px[i,j][0]*0.299) + int(px[i,j][1]*0.587) +int(px[i,j][2]*0.114),int(px[i,j][0]*0.299) + int(px[i,j][1]*0.587) +int(px[i,j][2]*0.114),int(px[i,j][0]*0.299) + int(px[i,j][1]*0.587) +int(px[i,j][2]*0.114))
        im.show()

class MatrixFilter:
    def __init__(self):
        self.vector = []
        mRadius = 3
        self.size = 2 * mRadius + 1
        self.width, self.height = im.size
      #  im.show()


    def clam(self,x,max,min):
        if x < min:
            return min
        if x > max:
            return max
        return x


    def color(self,x,y,radius):
        returnR = 0
        returnG = 0
        returnB = 0
        size = 2 * radius + 1
        for i in range(-radius,radius+1):
            for j in range(-radius,radius+1):
                idx = (i+radius) * size + j + radius

                color = px[self.clam(x+i,self.width-1,0),self.clam(y+j,self.height-1,0)]
                returnR+=color[0] * self.vector[idx]

                returnG += color[1] * self.vector[idx]
                returnB += color[2] * self.vector[idx]
            print(returnR)
        return tuple(map(int,((self.clam(returnR,255,0),self.clam(returnG,255,0),self.clam(returnB,255,0)))))

    def pixel(self,radius):
        for i in range(self.width):
            for j in range(self.height):
                color = self.color(i,j,radius)
              #  print(color)
                px[i,j]=tuple(map(int,color))
        im.show()

   # def show(self):
     #   im1.show()


class BlurFilter(MatrixFilter):
    def __init__(self):
        super().__init__()
        self.vector =[0] * (self.size * self.size)

        for i in range(self.size):
            for j in range(self.size):

                self.vector[i*self.size+j] = 1/(self.size ** 2)

    def __del__(self):
        pass




class GauseFilter(MatrixFilter):
    def __init__(self):
        super().__init__()
        self.create_gause_vector(3,2)

    def create_gause_vector(self,radius,sigma):
        size = 2 * radius + 1
        norma = 0
        self.vector = [0] * (self.size * self.size)
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                idx = (i + radius) * size + j + radius
              #  print(math.exp(-(i*i+j*j)/(sigma*sigma)))
                self.vector[idx] =math.exp(-(i*i+j*j)/(sigma*sigma))
                norma +=self.vector[idx]

        for i in range(size):
            for j in range(size):
                self.vector[i*size+j] /=norma


#q=GrayScaleFilter()
#q.start()
#ww = Inversion()
#ww.start()
#ww.show()
#z = GauseFilter()
#z.pixel(3)
b=BlurFilter()
b.pixel(1) #change 33 line
#im.show()

