import Baseline.src.ImagePrintUtils as ImagePrint
import gc
import datetime as dt
import os
import time
import cv2
import Baseline.src.ClusteringUtils as ClusteringUtils
import numpy as np
import pywt
from scipy.spatial import distance

from Baseline.src.ImageOutProcessor import ImageOutProcessor
from Baseline.src.ObjectDescription import ObjectDescription


###############THRESHOLD_CONTROL########################
G2NN_THRESHOLD=0.5
MIN_CLUSTER_SIZE_CONTROL=4
MIN_NO_OF_MATCHES=16
MIN_DISTANCE=30
G2NN_TO_SHOW=30
IMG_SIZE = 480
########################################################

COLOR_IMAGE=[]
KP=[]
APPROX_IMAGE=[]
NOW=""
ObjProcessor=''
LOGS=''

def g2nn1(description):
    global LOGS
    global COLOR_IMAGE
    global KP
    global G2NN_THRESHOLD
    
    len_description=len(description)
    description_object=[]
    all_distance_keypoint=[]

    for i in range(0,len_description): #adding all the description or key points in an array
        description_object.append(ObjectDescription(description[i],KP[i],None))

    for i in range(0,len_description):#taking eculidean distance of all points
        distance_keypoint=[]
        for j in range(0,len_description):
            if i!=j:
               dist=distance.euclidean(description_object[i].description,description_object[j].description)#computing euclidean distance
               distance_keypoint.append(ObjectDescription(dist,description_object[i].key_coordinates,description_object[j].key_coordinates))
        distance_keypoint=ImagePrint.sort1(distance_keypoint)
        linear_all_distance_keypoint=[]
        dummy=[]
        for j in range(0,len(distance_keypoint)-1):#checking with threshold value
            if(distance_keypoint[j+1].description!=0):
                if(distance_keypoint[j].description/distance_keypoint[j+1].description>G2NN_THRESHOLD):
                    if(j==0):
                        break
                    for x in range(0,j+1):#adding data in all_distance keypoint
                        linear_all_distance_keypoint.append(distance_keypoint[x])
                    dummy=distance_keypoint[0:j+1]
                    all_distance_keypoint.append(dummy)
                    break
    small_kp=[]
    for x in range(0,len(all_distance_keypoint)):
        for y in range(0,len(all_distance_keypoint[x])):
            if all_distance_keypoint[x][y].key_coordinates not in small_kp:
                small_kp.append(all_distance_keypoint[x][y].key_coordinates)
            if all_distance_keypoint[x][y].reference_coordinates not in small_kp:
                small_kp.append(all_distance_keypoint[x][y].reference_coordinates)
    KP=small_kp
    LOGS=LOGS+"LEN of kp in g2nn="+str(len(KP))+"\n"
    return all_distance_keypoint

def performDywt2(input_image):
    a = pywt.swt2(input_image, 'haar', level=1,start_level=0)
    LL, (LH, HL, HH) = a[0]
    LL=LL/255
    return {
        'LL': LL,
        'LH': LH,
        'HL': HL,
        'HH': HH
        }

def performSift(approx_image):
    global ObjProcessor
    global APPROX_IMAGE
    global KP
    global COLOR_IMAGE
    global LOGS
    sift = cv2.xfeatures2d.SIFT_create()
    kp,desc = sift.detectAndCompute(approx_image, None)
    KP=kp
    APPROX_IMAGE=approx_image
    return desc

def preprocess_sift(new_img_LL):
    ready2sift=ImagePrint.convertArray(ImagePrint.clipp(new_img_LL),np.uint8)
    desc=performSift(ready2sift)
    return desc

def resizeImage(image, newsize = 480, color = True):
    if color:
        height, width, depth = image.shape
    else:
        height, width = image.shape
    imgScale = newsize / width
    newX, newY = image.shape[1] * imgScale, image.shape[0] * imgScale
    newImage = cv2.resize(image, (int(newX), int(newY)))
    return newImage

def input(name):
    global COLOR_IMAGE
    COLOR_IMAGE=cv2.imread(name,1)
    image = cv2.imread(name, 0)
    # image = resizeImage(image, newsize=IMG_SIZE, color=False)
    # COLOR_IMAGE = resizeImage(COLOR_IMAGE, newsize=IMG_SIZE, color=True)
    return image


def buildTable(descriptor,kp):
        global COLOR_IMAGE
        global APPROX_IMAGE
        (image_width,image_height)=APPROX_IMAGE.shape
        length_of_descriptor=len(descriptor)
        TABLE=np.zeros((image_width,image_height),np.int8)
        TABLE1=np.zeros((image_width,image_height),np.int8)
        TABLE2=np.zeros((image_width,image_height),np.int8)
        TABLE=TABLE.tolist()
        TABLE1=TABLE1.tolist()
        TABLE2=TABLE2.tolist()
        for i in range(0,length_of_descriptor):
            descriptor_i=len(descriptor[i])
            for j in range(0, descriptor_i):
                x_matched=int(descriptor[i][j].key_coordinates.pt[0])
                y_matched=int(descriptor[i][j].key_coordinates.pt[1])
                TABLE[x_matched][y_matched]=descriptor[i][j].key_coordinates #matched
                x_ref=int(descriptor[i][j].reference_coordinates.pt[0])
                y_ref=int(descriptor[i][j].reference_coordinates.pt[1])
                TABLE[x_ref][y_ref]=descriptor[i][j].reference_coordinates#reference

                x1=x_ref
                y1=y_matched
                x2=x_matched
                y2=y_ref

                if x2<x1:
                    temp=x1
                    x1=x2
                    x2=temp
                    temp=y1
                    y1=y2
                    y2=temp

                TABLE1[x1][y1]=True
                TABLE2[x2][y2]=True  
                
        return TABLE,TABLE1,TABLE2

def checkClusters(TABLE,TABLE1,TABLE2,cluster1,cluster2,matched_kp=None):
    count=0
    gc.collect()
    matched_kp=[]
    full_cluster1=[]
    full_cluster2=[]
    for i in  range(0,cluster1.size):
        for j in range(0,cluster2.size):
            x1=int(cluster1.contentx[i])
            y1=int(cluster2.contenty[j])
            x2=int(cluster2.contentx[j])
            y2=int(cluster1.contenty[i])
            
            if x2<x1:
                temp=x1
                x1=x2
                x2=temp
                temp=y1
                y1=y2
                y2=temp
            full_cluster1.append(TABLE[x1][y2])
            full_cluster2.append(TABLE[x2][y1])
            if TABLE1[x1][y1]==True:
                if TABLE2[x2][y2]==True:
                    if TABLE[x1][y2] not in matched_kp:
                        matched_kp.append(TABLE[x1][y2])
                    if TABLE[x2][y1] not in matched_kp:
                        matched_kp.append(TABLE[x2][y1])
                    count+=1
    
    return matched_kp,count,full_cluster1,full_cluster2
def convertIntoXnY_Array():
    global KP
    global APPROX_IMAGE
    image_width,image_height=APPROX_IMAGE.shape
    k=[]
    keypoint_map_tabl=np.zeros((image_width,image_height),np.int8)
    keypoint_map_table=keypoint_map_tabl.tolist()

    for i in range(0,len(KP)):
        x=int(KP[i].pt[0])
        y=int(KP[i].pt[1])
        keypoint_map_table[x][y]=KP[i]
        k.append(ClusteringUtils.XYAppend(x,y))
    return k     

def drawClusters(vector,table,table1,table2):
    global LOGS
    matched_kp=[]
    image_name=1  
    count=0
    s=""
    f1=[]
    f2=[]
    final_keypoint=[]
    cluster=[]
    for i in range(1,len(vector)-1):
        if vector[i].size>MIN_CLUSTER_SIZE_CONTROL:
            for j in range(i+1,len(vector)):
                    if vector[i].size>MIN_CLUSTER_SIZE_CONTROL:
                        matched,count,full_cluster1,full_cluster2= checkClusters(table,table1,table2,vector[i],vector[j],matched_kp)
                    if count>=MIN_NO_OF_MATCHES:
                        LOGS=LOGS+"cluster1.size="+str(vector[i].size)+"\n"
                        LOGS=LOGS+"cluster2.size="+str(vector[j].size)+"\n"
                        LOGS+="matches found in clusters="+str(count)+"\n"
                        cluster.extend(full_cluster1)
                        cluster.extend(full_cluster2)
                        final_keypoint.extend(full_cluster1)
                        final_keypoint.extend(full_cluster2)
                        f1.extend(full_cluster1)
                        f2.extend(full_cluster2)
                        s=s+str(image_name)+"="+str(count)+"\n"
                        image_name+=1
    img=cv2.drawKeypoints(COLOR_IMAGE, final_keypoint, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    ObjProcessor.addImage(img,"agglomerative")
    return f1,f2,img.copy()

def drawLinesOnImage(f1,f2,cluster_img):
    img = cluster_img
    for i in range(0, len(f1), 2):
        cv2.line(img, (int(f1[i].pt[0]), int(f1[i].pt[1])), (int(f2[i].pt[0]), int(f2[i].pt[1])), (0,255,0), 1, cv2.LINE_AA)
    ObjProcessor.addImage(img,"result")

def initialize():
    global COLOR_IMAGE
    global KP
    global LOGS
    global APPROX_IMAGE
    global NOW
    global ObjProcessor
    COLOR_IMAGE=[]
    KP=[]
    LOGS=""
    APPROX_IMAGE=[]
    NOW=""
    ObjProcessor=''


def main(file_name):
    initialize() # Initialise params
    global ObjProcessor
    global KP
    global LOGS
    global COLOR_IMAGE

    ObjProcessor = ImageOutProcessor(file_name[1]) # Initialise object out processor class
    np.set_printoptions(threshold=np.nan) # set printing options (nan)
    
    image=input(file_name[0]) # read first image from filenames list

    start=time.clock()
    print(dt.datetime.now())    

    print(file_name[1]+"-"+"start_sift")
    desc=performSift(image)
    print(file_name[1]+"-"+"end_sift")

    print(file_name[1]+"-"+"start_g2nn")
    g2nn_output=g2nn1(desc)
    print(file_name[1]+"-"+"end g2nn")
    
    print(file_name[1]+"-"+"start_buildTable")
    table,table1,table2=buildTable(g2nn_output,KP)
    print(file_name[1]+"-"+"end g2nn")

    print(file_name[1]+"-"+"before clustering")
    k=convertIntoXnY_Array()
    vector=ClusteringUtils.heirarchyCluster(k,MIN_DISTANCE)
    print(file_name[1]+"-"+"end clustering")
           
    print(file_name[1]+"-"+"start drawClusters(Matching)")
    drawClusters(vector=vector,table=table,table1=table1,table2=table2)
    print(file_name[1]+"-"+"end drawClusters(Matching)")
    
    end=time.clock()
    print("start=%s end=%s" %(start,end))
    print((end-start))

    ObjProcessor.addObject(LOGS,"logs")
    ObjProcessor.printFile()
    ObjProcessor.printImage()

if __name__ == "__main__":
    DIR=os.path.abspath(__file__)
    DIR=str(DIR)
    DIR=DIR.replace(str(os.path.basename(__file__)),"Input/")
    list_dir=os.listdir(DIR)

    for name in list_dir:
        file_name=[]
        file_name.append(DIR+name)
        file_name.append(name)
        LOGS=LOGS+"G2NN_THRESHOLD="+str(G2NN_THRESHOLD)+"\n"
        LOGS=LOGS+"MIN_CLUSTER_SIZE_CONTROL="+str(MIN_CLUSTER_SIZE_CONTROL)+"\n"
        LOGS=LOGS+"MIN_NO_OF_MATCHES="+str(MIN_NO_OF_MATCHES)+"\n"
        LOGS=LOGS+"MIN_DISTANCE="+str(MIN_DISTANCE)+"\n"
        LOGS=LOGS+"G2NN_TO_SHOW="+str(G2NN_TO_SHOW)+"\n"
        print(file_name)
        main(file_name)

