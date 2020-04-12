# Importing Image from PIL package
from PIL import Image
import numpy
# creating a image object
im = Image.open(r"chup.jpg")
im1 = Image.open(r"chup.jpg")
im2 = Image.open(r"bin.jpg")
im22 = Image.open(r"bin.jpg")
im3 = Image.open(r"bin2.jpg")
im33 = Image.open(r"bin2.jpg")
im4 = Image.open(r"ball3.jpg")
im44 = Image.open(r"ball3.jpg")
import math

MH = 3
MW = 3


def clam(x, max, min):
    if x < min:
        return min
    if x > max:
        return max
    return x

class Filter:
    def __init__(self):
        pass
class Inversion(Filter):
    def __init__(self):
        super().__init__()

    def start(self, new_image):
        px1 = new_image.load()
        px = im.load()
        self.width, self.height = new_image.size
        for i in range(self.width):
            for j in range(self.height):
                px[i,j] = (255-px1[i,j][0],255-px1[i,j][1],255-px1[i,j][2])
        return im


class GrayScaleFilter(Filter):
    def __init__(self):
        super().__init__()

    def start(self, new_image):
        px1 = new_image.load()
        px = im.load()
        self.width, self.height = new_image.size
        for i in range(self.width):
            for j in range(self.height):
                px[i,j]=(int(px1[i,j][0]*0.299) + int(px1[i,j][1]*0.587) +int(px1[i,j][2]*0.114),int(px1[i,j][0]*0.299) + int(px1[i,j][1]*0.587) +int(px1[i,j][2]*0.114),int(px1[i,j][0]*0.299) + int(px1[i,j][1]*0.587) +int(px1[i,j][2]*0.114))
        return im

class SeppiaFilter(Filter):
    def start(self, new_image):
        px1 = new_image.load()
        px = im.load()
        self.width, self.height = new_image.size
        for i in range(self.width):
            for j in range(self.height):
                Intencity = 0.299 * px1[i,j][0] + 0.587 * px1[i,j][1] + 0.114 * px1[i,j][2]
                k = 4
                R = clam(Intencity + 2 * k, 255, 0)
                G = clam(Intencity + 0.5 * k, 255, 0)
                B = clam(Intencity - 1 * k, 255, 0)
                px[i,j]=(int(R), int(G), int(B))
        return im

class BrightnessFilter(Filter):
    def start(self, new_image):
        px1 = new_image.load()
        px = im.load()
        self.width, self.height = new_image.size
        for i in range(self.width):
            for j in range(self.height):
                R = clam(px1[i,j][0] + 50, 255, 0)
                G = clam(px1[i,j][1] + 50, 255, 0)
                B = clam(px1[i,j][2] + 50, 255, 0)
                px[i,j]=(int(R), int(G), int(B))
        return im



class MatrixFilter:
    def __init__(self,):
        self.vector = []
        self.mRadius = 3
        self.size = 2 * self.mRadius + 1

    def color(self,x,y,radius, px1):
        returnR = 0
        returnG = 0
        returnB = 0
        size = 2 * radius + 1
        for i in range(-radius,radius+1):
            for j in range(-radius,radius+1):
                idx = (i+radius) * size + j + radius
                p_x = int(clam(x+i,self.width-1,0))
                p_y = int(clam(y+j,self.height-1,0))
                color = px1[p_x, p_y]
                returnR += color[0] * self.vector[idx]
                returnG += color[1] * self.vector[idx]
                returnB += color[2] * self.vector[idx]

        return tuple(map(int,((clam(returnR,255,0),clam(returnG,255,0),clam(returnB,255,0)))))

    def pixel(self,radius, new_image):
        px1 = new_image.load()
        px = im.load()
        self.width, self.height = new_image.size
        for i in range(self.width):
            for j in range(self.height):
                color = self.color(i,j,radius, px1)
                px[i,j]=color
        return im



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

class SobelXFilter(MatrixFilter):
    def __init__(self):
        super().__init__()
        self.vector =[0] * (self.size * self.size)
        self.vector[0] = -1
        self.vector[1] = 0
        self.vector[2] = 1
        self.vector[3] = -2
        self.vector[4] = 0
        self.vector[5] = 2
        self.vector[6] = -1
        self.vector[7] = 0
        self.vector[8] = 1

    def __del__(self):
        pass


class SobelYFilter(MatrixFilter):
    def __init__(self):
        super().__init__()
        self.vector =[0] * (self.size * self.size)
        self.vector[0] = -1
        self.vector[1] = -2
        self.vector[2] = -1
        self.vector[3] = 0
        self.vector[4] = 0
        self.vector[5] = 0
        self.vector[6] = 1
        self.vector[7] = 2
        self.vector[8] = 1

    def __del__(self):
        pass

class SharpnessFilter(MatrixFilter):
    def __init__(self):
        super().__init__()
        self.vector =[0] * (self.size * self.size)
        self.vector[0] = 0
        self.vector[1] = -1
        self.vector[2] = 0
        self.vector[3] = -1
        self.vector[4] = 5
        self.vector[5] = -1
        self.vector[6] = 0
        self.vector[7] = -1
        self.vector[8] = 0

    def __del__(self):
        pass

class Gray_world(Filter):
    def start(self, new_image):
        px1 = new_image.load()
        px = im44.load()
        self.width, self.height = new_image.size
        R = 0
        G = 0
        B = 0
        for x in range(self.width):
            for y in range(self.height):
                R += px1[x, y][0]
                G += px1[x, y][1]
                B += px1[x, y][2]

        kol = self.width * self.height
        Rav = R / kol
        Gav = G / kol
        Bav = B / kol
        average = (Rav + Gav + Bav) / 3

        for x in range(self.width):
            for y in range(self.height):
                colorred = clam(px1[x, y][0] * average / Rav, 255, 0)
                colorgreen = clam(px1[x, y][1] * average / Gav, 255, 0)
                colorblue = clam(px1[x, y][2] * average / Bav, 255, 0)
                color = tuple(map(int,(colorred, colorgreen, colorblue)))
                px[x, y] = color
        return im44

class Transfer(Filter):
    def start(self, new_image):
        px1 = new_image.load()
        px = im.load()
        self.width, self.height = new_image.size
        for x in range(self.width):
            for y in range(self.height):
                if x + 50 < self.width:
                    color = px1[x + 50, y]
                    px[x, y] = color
                else:
                    color = (0, 0, 0)
                    px[x ,y] = color
        return im

class Waves(Filter):
    def start(self, new_image):
        px1 = new_image.load()
        px = im.load()
        self.width, self.height = new_image.size
        for x in range(self.width):
            for y in range(self.height):
                value = x + 20 * math.sin((2 * math.pi * x) / 30)
                if value < self.width:
                    color = px1[value, y]
                    px[x, y] = color
                else:
                    px[x, y] = px1[x,y]
        return im

class SharpnessLab(MatrixFilter):
    def __init__(self):
        super().__init__()
        self.vector =[0] * (self.size * self.size)
        self.vector[0] = -1
        self.vector[1] = -1
        self.vector[2] = -1
        self.vector[3] = -1
        self.vector[4] = 9
        self.vector[5] = -1
        self.vector[6] = -1
        self.vector[7] = -1
        self.vector[8] = -1

    def __del__(self):
        pass


class Motion_blur(MatrixFilter):
    def __init__(self):
        super().__init__()
        self.vector =[0] * (self.size * self.size)

        for i in range(self.size):
            for j in range(self.size):
                if i == j:
                    self.vector[i * self.size + j] = 1/(self.size)
                else:
                    self.vector[i * self.size + j] = 0

    def __del__(self):
        pass


class Median_filter(MatrixFilter):
    def __init__(self):
        super().__init__()

    def __del__(self):
        pass

    def color(self, x, y, radius, px1):
        returnR = 0
        returnG = 0
        returnB = 0
        size = 2 * radius + 1
        Intencity_list = []
        coordinates_list = []
        for i in range(-radius,radius+1):
            for j in range(-radius,radius+1):
                p_x = int(clam(x + i, self.width - 1, 0))
                p_y = int(clam(y + j, self.height - 1, 0))
                Intencity = 0.299 * px1[p_x, p_y][0] + 0.587 * px1[p_x, p_y][1] + 0.114 * px1[p_x, p_y][2]
                Intencity_list.append(Intencity)
                coordinates_list.append(tuple((p_x, p_y)))

        for k in range(len(Intencity_list)):#сортировка пузырьком
            for m in range(1, len(Intencity_list)):
                if Intencity_list[m-1] > Intencity_list[m]:
                    tmp = Intencity_list[m-1]
                    Intencity_list[m - 1] = Intencity_list[m]
                    Intencity_list[m] = tmp
                    tmp = coordinates_list[m - 1]
                    coordinates_list[m - 1] = coordinates_list[m]
                    coordinates_list[m] = tmp
        number = int(len(Intencity_list) / 2)
        coordinates = coordinates_list[number]
        color = px1[coordinates[0], coordinates[1]]
        return tuple(map(int, color))


class Linear_tension(Filter):
    def start(self, new_image):
        px1 = new_image.load()
        px = im.load()
        self.width, self.height = new_image.size
        maxR = 0
        maxG = 0
        maxB = 0
        minR = 255
        minG = 255
        minB = 255
        for x in range(self.width): #ищем min max rgb
            for y in range(self.height):
                if px1[x, y][0] > maxR:
                    maxR = px1[x, y][0]
                if px1[x, y][0] < minR:
                    minR = px1[x, y][0]
                if px1[x, y][1] > maxG:
                    maxG = px1[x, y][1]
                if px1[x, y][1] < minG:
                    minG = px1[x, y][1]
                if px1[x, y][2] > maxB:
                    maxB = px1[x, y][2]
                if px1[x, y][2] < minB:
                    minB= px1[x, y][2]
        for x in range(self.width):
            for y in range(self.height):
                px[x, y] = tuple((int((px1[x, y][0] - minR) * 255 / (maxR - minR)), int((px1[x, y][1] - minG) * 255 / (maxG - minG)), int((px1[x, y][2] - minB) * 255 / (maxB - minB))))
        return im

def create():
    mask = []
    print('Введите структурный элемент:')
    for i in range(MW):  # MH - высота, MW - ширина
        for j in range(MH):
            t = input()
            mask.append(t)
    return mask

class Dilation(Filter):
    def start(self, mask, old_image, new_image):
        px1 = new_image.load()
        px = old_image.load()
        self.width, self.height = new_image.size
        hieght = self.height - int(MH / 2)
        width = self.width - int(MW / 2)
        for y in range(int(MH / 2), hieght):
            for x in range(int(MW / 2), width):
                maxR = 0
                maxG = 0
                maxB = 0
                for j in range(int(-MH / 2), int(MH / 2) + 1):
                    for i in range(int(-MW / 2), int(MW / 2) + 1):
                        p_x = int(clam(x + i, self.width - 1, 0))
                        p_y = int(clam(y + j, self.height - 1, 0))
                        if mask[i * MW + j] and px1[p_x, p_y][0] > maxR:
                            maxR = px1[p_x, p_y][0]
                        if mask[i * MW + j] and px1[p_x, p_y][1] > maxG:
                            maxG = px1[p_x, p_y][1]
                        if mask[i * MW + j] and px1[p_x, p_y][2] > maxB:
                            maxB = px1[p_x, p_y][2]
                        px[x, y] = tuple((maxR, maxG, maxB))
        return old_image

class Erosion(Filter):
    def start(self, mask, old_image, new_image):
        px1 = new_image.load()
        px = old_image.load()
        self.width, self.height = new_image.size
        hieght = self.height - int(MH / 2)
        width = self.width - int(MW / 2)
        for y in range(int(MH / 2), hieght):
            for x in range(int(MW / 2), width):
                minR = 255
                minG = 255
                minB = 255
                for j in range(int(-MH / 2), int(MH / 2) + 1):
                    for i in range(int(-MW / 2), int(MW / 2) + 1):
                        p_x = int(clam(x + i, self.width - 1, 0))
                        p_y = int(clam(y + j, self.height - 1, 0))
                        if mask[i * MW + j] and px1[p_x, p_y][0] < minR:
                            minR = px1[p_x, p_y][0]
                        if mask[i * MW + j] and px1[p_x, p_y][1] < minG:
                            minG = px1[p_x, p_y][1]
                        if mask[i * MW + j] and px1[p_x, p_y][2] < minB:
                            minB = px1[p_x, p_y][2]
                        px[x, y] = tuple((minR, minG, minB))
        return old_image

class Opening(Filter):
    def start(self, mask, old_image, new_image):
        erosion = Erosion()
        eros_im = erosion.start(mask, old_image, new_image)
        eros_im.save('test.jpg')
        imt = Image.open(r"test.jpg")
        imt1 = Image.open(r"test.jpg")
        dilation = Dilation()
        dilate_im = dilation.start(mask, imt, imt1)
        return dilate_im

class Closing(Filter):
    def start(self, mask, old_image, new_image):
        dilation = Dilation()
        dilate_im = dilation.start(mask, old_image, new_image)
        dilate_im.save('test.jpg')
        imt = Image.open(r"test.jpg")
        imt1 = Image.open(r"test.jpg")
        erosion = Erosion()
        eros_im = erosion.start(mask, imt, imt1)
        return eros_im

class Grad(Filter):
    def start(self, mask, old_image, new_image):
        dilation = Dilation()
        dilate_im = dilation.start(mask, old_image, new_image)
        dilate_im.save('grad_dil.jpg')

        erosion = Erosion()
        eros_im = erosion.start(mask, old_image, new_image)
        eros_im.save('grad_er.jpg')

        self.width, self.height = new_image.size

        dil_im = Image.open(r"grad_dil.jpg")
        er_im = Image.open(r"grad_er.jpg")
        er = er_im.load()
        dil = dil_im.load()

        for x in range(self.width):
            for y in range(self.height):
                red = dil[x, y][0] - er[x, y][0]
                green = dil[x, y][1] - er[x, y][1]
                blue = dil[x, y][2] - er[x, y][2]
                dil[x, y] = tuple((red, green, blue))
        return dil_im

#---------------------------задачи для здачи лабораторной работы----------------

#---1ое задание, фильтр серый мир---
#grayworld = Gray_world()
#grayworld_image = grayworld.start(im4)
#grayworld_image.save('Gray_world.jpg')

#---2ое задание, линейное растяжение---
#этот метод не работает если на картинке одновременно присутствуют белые (255) и чёрные (0)
# пиксели, в этом случае можно применять нелинейную (gamma) коррекцию.
#linear_tension=Linear_tension() #сюда лучше картинку chup.jpg
#linear_tension_image = linear_tension.start(im1)
#linear_tension_image.save('Linear_tension.jpg')

#---3е задание: перенос, волны, матричный резкость, матричный motion blur---
#transfer=Transfer()
#transfer_image = transfer.start(im1)
#transfer_image.save('Transfer.jpg')

#waves=Waves()
#waves_image = waves.start(im1)
#waves_image.save('Waves.jpg')

#sharpnessLab=SharpnessLab()
#sharpness_image = sharpnessLab.pixel(1, im1) #mRadius таким же надо делать в классе MatrixFilter
#sharpness_image.save('SharpnessLab.jpg')
'''
motion_blur=Motion_blur()
motion_image = motion_blur.pixel(3, im1) #mRadius таким же надо делать в классе MatrixFilter
motion_image.save('Motion_blur.jpg')

#---4ое, 6ое задания, операции математической морфологии + задание структурного элемента---
mask = create()
dilation = Dilation()
dilation_image = dilation.start(mask, im2, im22)
dilation_image.save('Dilation.jpg')

erosion = Erosion()
erosion_image = erosion.start(mask, im2, im22)
erosion_image.save('Erosion.jpg')

opening = Opening()
opening_image = opening.start(mask, im2, im22)
opening_image.save('Opening.jpg')

closing = Closing()
closing_image = closing.start(mask, im3, im33)
closing_image.save('Closing.jpg')

grad = Grad()
grad_image = grad.start(mask, im, im1)
grad_image.save('Grad.jpg')
'''
#ниже все сделано


#---5ое задание, медианный фильтр---
median_filter = Median_filter()
median_image = median_filter.pixel(3, im1) #mRadius таким же надо делать в классе MatrixFilter
median_image.save('Median_filter.jpg')

#---------------------------задачи для самостоятельного решения----------------
'''sobelx=SobelXFilter()
sobelx_image = sobelx.pixel(1, im1)
sobelx_image.save('SobelXFilter.jpg')

sobely=SobelYFilter()
sobely_image = sobely.pixel(1, im1)
sobely_image.save('SobelYFilter.jpg')

sharpness=SharpnessFilter()
sharpness_image = sharpness.pixel(1, im1)
sharpness_image.save('SharpnessFilter.jpg')
'''
#brightness=BrightnessFilter()
#brightness_image = brightness.start(im1)
#brightness_image.save('BrightnessFilter.jpg')

#seppia=SeppiaFilter()
#seppia_image = seppia.start(im1)
#seppia_image.save('SeppiaFilter.jpg')

#gray = GrayScaleFilter()
#gray_image = gray.start(im1)
#gray_image.save('GrayScaleFilter.jpg')


#-----------------------------самые первые в методичке--------------------------
#inversion = Inversion()
#inv_image = inversion.start(im1)
#inv_image.save('Inversion.jpg')
#inv_image.show()
#z = GauseFilter()
#gause_image = z.pixel(3, im1) #mRadius таким же надо делать в классе MatrixFilter
#gause_image.save('GauseFilter.jpg')
#gause_image.show()
#b = BlurFilter()
#blur_image = b.pixel(3, im1) #mRadius таким же надо делать в классе MatrixFilter
#blur_image.save('BlurFilter.jpg')
#blur_image.show()

