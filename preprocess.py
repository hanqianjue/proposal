# load all images in a directory
import os

import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt

#cropping variables
xs=16
xf=240
ys=16
yf=240
# xs=40
# xf=240
# ys=20
# yf=220
maxWidth = xf-xs#200
maxHeight = yf-ys#200

#dimensions of resultant images
finalWidth = 224
finalHeight = 224

#directory paths
fold = "fv_struct"
proc_dir = "fv_proc"
proc_train_dir = "fv_proc\\train"
proc_test_dir = "fv_proc\\test"



for root, dirs, files in os.walk(fold):
    for name in files:
        if name.endswith(".bmp"):
        	# load image as grayscale
            img_data_gs = cv2.imread(os.path.join(root, name), cv2.IMREAD_GRAYSCALE)

            # crop image
            img_data_crp = img_data_gs[ys:yf, xs:xf]

            # enhance contrast of image
            img_data_crp_enh = cv2.equalizeHist(img_data_crp)
            
            # blur the image
            img_data_blur = cv2.bilateralFilter(img_data_crp_enh, 11, 20, 20)
            
            # make a copy. we will apply the mask to this copy
            img_data_crp2 = img_data_blur.copy()
            
            # split image along horizontal axis 
            img_blur_top = img_data_blur[0:int(maxHeight/2), 0:maxWidth]
            img_blur_bot = img_data_blur[int(maxHeight/2):maxHeight, 0:maxWidth]
            
            #edge detection            
            kernel = np.ones((2,2), np.uint8)

            img_edge_bot = cv2.Canny(img_blur_bot, 50, 230)
            img_edge_bot = cv2.dilate(img_edge_bot, kernel, iterations=3)
            img_edge_bot = cv2.erode(img_edge_bot, kernel, iterations=1)

            # create preliminary mask for bottom image
            cv2.floodFill(img_edge_bot, None, (50,int(maxHeight/2)-1), 255)
            cv2.floodFill(img_edge_bot, None, (0,0), 0)
                        
            cnts = cv2.findContours(img_edge_bot, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
            c = None
            
            if len(cnts) > 0:
               for c in cnts:
                   cv2.fillPoly(img_edge_bot, pts =[c], color=(255,255,255))
            
            img_edge_top = cv2.Canny(img_blur_top, 50, 230)
            img_edge_top = cv2.dilate(img_edge_top, kernel, iterations=3)
            img_edge_top = cv2.erode(img_edge_top, kernel, iterations=1)
            
            # create preliminary mask for top image
            cv2.floodFill(img_edge_top, None, (0,0), 255)
            cv2.floodFill(img_edge_top, None, (50,int(maxHeight/2)-1), 0)
            
            cnts = cv2.findContours(img_edge_top, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
            
            if len(cnts) > 0:
                for c in cnts: 
                    cv2.fillPoly(img_edge_top, pts =[c], color=(255,255,255))
                    
            # join the masked images together        
            img_mask = cv2.vconcat([img_edge_top, img_edge_bot])
            img_mask = cv2.bitwise_not(img_mask)
            
            # attempt fill any gaps
            cv2.floodFill(img_mask, None,(110,0),0) 
            cv2.floodFill(img_mask, None,(110,maxHeight-1),0) 
            cv2.floodFill(img_mask, None,(110,int(maxHeight/2)-1),255) 
            
            # edge detection, once again
            img_data_crp[img_mask == 0] = 0
            
            # capture final contour
            img_roi = cv2.Canny(img_data_crp, 25, 200)
            img_roi = cv2.dilate(img_roi, kernel, iterations=3)
            img_roi = cv2.erode(img_roi, kernel, iterations=1)
            
            # attempt to clean it up
            cv2.floodFill(img_roi, None, (50,0), 0)
            cv2.floodFill(img_roi, None, (50,maxHeight-1), 0)
            cv2.floodFill(img_roi, None, (50,int(maxHeight/2)), 255)
            
            # create border along edges
            for i in range(maxHeight):
                img_roi[i,0] = 0
                img_roi[i,1] = 0
                img_roi[i,maxWidth-1] = 0
                img_roi[i,maxWidth-2] = 0
            
            for i in range(maxWidth):
                img_roi[0,i] = 0
                img_roi[maxHeight-1,i] = 0
            
            # capture contour and get bounding rectangle
            roi_cnts, hier = cv2.findContours(img_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            roi_c = max(roi_cnts, key = cv2.contourArea)
            cv2.fillPoly(img_roi, pts =[roi_c], color=(255,255,255))
            img_data_crp2[img_roi == 0] = 0
            x,y,w,h = cv2.boundingRect(roi_c)
            
            # get corner points of bounding rectangle
            box = np.array([
                    [x,y],
                    [x+w,y],
                    [x+w,y+h],
                    [0,y+h]], dtype="float32")
            
            dst = np.array([
                    [0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype = "float32")
            
            # stretch roi to fill 200x200 image
            M = cv2.getPerspectiveTransform(box, dst)
            img_final = cv2.warpPerspective(img_data_crp2, M, (maxWidth, maxHeight))
            img_final = cv2.resize(img_final, (finalWidth, finalHeight))
            
            #print('> loaded %s %s %s' % (name, root, img_data_crp.shape))
            
            # create directories
            dir_names = root.split("\\")
            if(not os.path.exists(proc_dir)):
                os.makedirs(proc_dir)
                os.makedirs(proc_train_dir)
                os.makedirs(proc_test_dir)
            
            # save images in respective directory
            if( "train" in root ):
                if(not os.path.exists("%s\\%s" % (proc_train_dir, dir_names[2]))):
                    os.makedirs("%s\\%s" % (proc_train_dir, dir_names[2]))
                file2save = os.path.join(proc_train_dir, dir_names[2], name)
                cv2.imwrite(file2save, img_final)
                
            if( "test" in root):
                if(not os.path.exists("%s\\%s" % (proc_test_dir, dir_names[2]))):
                    os.makedirs("%s\\%s" % (proc_test_dir, dir_names[2]))
                file2save = os.path.join(proc_test_dir, dir_names[2], name)
                cv2.imwrite(file2save, img_final)
            
            
    

            