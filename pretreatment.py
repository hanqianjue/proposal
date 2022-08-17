#图像预处理
import cv2
import numpy as np
from skimage.measure import label

# Prewitt 算子作为边缘检测
def Prewitt(img):
    kernelup = np.array([[-1, -1, -1,-1, -1, -1], [0, 0, 0,0, 0, 0], [1,1,1,1,1,1,1]], dtype=int)
    kerneldown = np.array([[1,1,1,1,1,1], [0, 0, 0,0, 0, 0], [-1, -1, -1,-1, -1, -1]], dtype=int)

    up = cv2.filter2D(img, cv2.CV_16S, kernelup)         #图像卷积运算
    down = cv2.filter2D(img, cv2.CV_16S, kerneldown)
    # 转uint8
    absup = cv2.convertScaleAbs(up)      #缩放，计算绝对值，然后将结果转换为8位
    absdown = cv2.convertScaleAbs(down)
    Prewitt = cv2.addWeighted(absup, 0.5, absdown, 0.5, 0)  #将两张相同大小，相同类型的图片融合
    return Prewitt

#def RotationCorrection:
def largestConnectComponent(bw_img ):
    labeled_img, num = label(bw_img, background=0, return_num=True)

    # plt.figure(), plt.imshow(labeled_img, 'gray')

    max_label = 0
    max_num = 0
    max_num_2 = 0
    max_label_2 = 0
    for i in range(1, num + 1):  # 这里从1开始，防止将背景设置为最大连通域
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    for i in range(1, num + 1):
        if np.sum(labeled_img == i) > max_num_2:
            if np.sum(labeled_img == i)!= max_num:
                max_num_2 = np.sum(labeled_img == i)
                max_label_2 = i
    lcc = (labeled_img == max_label)
    lcc_2 = (labeled_img == max_label_2)
    return lcc|lcc_2

def find_peak(arr):
    peaks=[]
    step=1
    pos=1
    while(pos<300):   #实际上两个peak主要分布在100-200和450-550
        if (arr[pos]>=arr[pos+step])and(arr[pos]>arr[pos-step]):
            if ((arr[pos]>arr[pos+20])and(arr[pos]>arr[pos-20])):  #这个if语句可以滤除很多的噪音peak
                peaks.append(pos)
        pos=pos+step
    peak1=0
    peak2=320  #取的图片中间位置，但是有可能存在后面的关节腔灰度值比中间位置低的情况
    print(peaks)
    for i in peaks:   #这个循环从好多个peak中找到我们要的两个最大的peak
        if (i<340):
            if arr[i]>=arr[peak1] :
                peak1=i
        else:
             if arr[i]>=arr[peak2] :
                peak2=i
    #if(peak2-peak1)<200 :
        #print("EOI error in peak")
    return peak1,peak2





class pretreatmengt_ROI():
    def __init__(self):
        super(pretreatmengt_ROI, self).__init__()






