import cv2 as cv 
import numpy as np


#Basic printed document processing
def preprocess_image(image,
                    grey=1,
                    threshold=180,
                    adapt=0,
                    blur=0,
                    thresh=0,
                    sharp=0,
                    edge_cascade=0,
                    edge1=50,
                    edge2=200):

    newImg=image
    if grey:
        newImg=Grey(newImg)
    if edge_cascade:
        newImg=EdgeCascade(newImg,edge1,edge2)
    if blur:
        newImg=Blur(newImg)
    if thresh:
        newImg=Threshold(newImg,threshold)
    if adapt:
        newImg=adaptiveThreshold(newImg)
    if sharp:
        newImg=Sharpen(newImg)
    return newImg

#Rescaling images
def Rescale(img,dim_height=1280,dim_width=720):
    height_scale=dim_height/img.shape[0]
    width_scale=dim_width/img.shape[1]
    height=int(img.shape[0]*height_scale)
    width=int(img.shape[1]*width_scale)
    dimensions=(height,width)
    return cv.resize(img,dimensions,interpolation=cv.INTER_CUBIC)

#Transforming image into greyscale
def Grey(img):
    greyImg=cv.cvtColor(img,code=cv.COLOR_BGR2GRAY)
    return greyImg

#Edge cascading
def EdgeCascade(img,t1=50,t2=200):
    newImg=cv.Canny(img, t1, t2)
    return newImg

#Blurring 
def Blur(img):
    newImg=cv.medianBlur(img,3)
    return newImg

#Thresholding
def Threshold(img,threshold=180):
    thresh, newImg=cv.threshold(img,threshold,250,cv.THRESH_BINARY)
    return newImg

def adaptiveThreshold(img):
    newImg=cv.adaptiveThreshold(img,200,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,3)
    return newImg

#Sharpening
old_kernel_sharp=np.array([
                    [-1,-1,-1],
                    [-1, 9,-1],
                    [-1,-1,-1]])

kernel_sharp=np.array([[0,-1, 0],
                       [-1,5,-1],
                       [0,-1, 0]])

def Sharpen(img,kernel_sharp=kernel_sharp):
    return cv.filter2D(img,-1,kernel_sharp)