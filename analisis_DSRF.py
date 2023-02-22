# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 13:38:31 2023

@author: rosalio
"""


# voy a tratar de analizar las imágenes de dsrf

# cargo la imagen de prubea
import glob
l = 3
count = 0
for filename in glob.glob(r"C:\Users\rosalio\Documents\2023\wing_asymmetry\analisis\hembras\controles\select\dsrf\*.tif"):
    if count==l:
        path = filename
        #print(path)
    count+=1
    
#path = C:\Users\rosalio\Documents\2023\wing_asymmetry\analisis\hembras\controles\select\dsrf\MAX_control-hembras-30nov2022.lif - Series006.tif
  
import cv2
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)


# cargo los datos
import numpy as np
elipse = np.load(r"C:\Users\rosalio\Documents\2023\wing_asymmetry\analisis\hembras\controles\select\dsrf\analisis_09FEB2023\datos_hembras_control_09FEB2023.npy", allow_pickle=True)

x, y, borde_x, borde_y, centroidx, centroidy, x0, y0, ap, bp, e, phi, thDA, threshDA, thVA, threshVA, thDP, threshDP, thVP, threshVP = elipse[l]



##############################################################
##############################################################
def img_cut_ellipse(img2analize, x, y, centroidx, centroidy, x0, y0, ap, bp, e, phi):
    """ corta img2analize en la elipse"""
    # el centroide de la elipse está en la posición
    # x = centroidx + x0
    # y = centroidy + y0
    cx = centroidx + x0
    cy = centroidy + y0
    # recorto la imagen
    
    rows_original, cols_original = np.shape(img2analize)
    
    # esto es por si la elipse a recortar se sale de los bordes de la imagen
    if(cy-bp>0):
        new_row0 = cy-bp
    else:
        new_row0 = 0
        
    if(cy+bp< rows_original):
        new_row1 = cy+bp
    else:
        new_row1 = rows_original    
        
    if(cx-ap>0):
        new_col0 = cx-ap
    else:
        new_col0 = 0
        
    if(cx+ap< cols_original):
        new_col1 = cx + ap
    else:
        new_col1 = cols_original
    

    img_roi = img2analize[int(round(new_row0)):int(round(new_row1)), int(round(new_col0)):int(round(new_col1))]


    img_roi_square = img_roi
    
    # elimino los puntos fuera de la elipse en ambas imágenes
    # creo una máscara con la ellipse calculada
    
    rows, cols = img_roi.shape
    mask_ellipse = np.zeros((rows,cols), np.uint8)

    #creo que el nuevo centro de la ellipse está en ap, bp
    center = (int(round(ap)),int(round(bp)))
    axesL = (int(round(ap)), int(round(bp)))
    angle = int(round(phi*180/np.pi))
    start_angle = 0
    end_angle = 360
    color = (255, 0, 0)
    thickness = -1

    mask_ellipse = cv2.ellipse(mask_ellipse, center, axesL, angle, start_angle, end_angle, color, thickness)
    img_roi_ellipse = cv2.bitwise_and(img_roi, mask_ellipse)
    #img_roi_ellipse = cv2.ellipse(img_roi_ellipse, center, axesL, angle, start_angle, end_angle, color, 3)
    return img_roi_ellipse, img_roi_square    
    
##############################################################
#############################################################



from skimage import morphology

footprint = morphology.disk(3)
res = morphology.white_tophat(img, footprint)
img = img - res
    
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,199, 3)


img_roi_ellipse, img_roi_square  = img_cut_ellipse(th3, x, y, centroidx, centroidy, x0, y0, 0.9*ap, 0.9*bp, e, phi)
th3 = img_roi_ellipse

#cv2.imshow("img cortada", th3)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


rows, cols  = th3.shape
mask = np.zeros((rows, cols), np.uint8)

contours, hierarchy = cv2.findContours(th3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

k = 0
for i in contours:
    a = cv2.contourArea(i)
    if(a>300):
        cv2.drawContours(mask, contours, k, (255, 0, 0), -1)
    k += 1
    
cv2.imshow("mask image", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
     
#inverted_img = cv2.bitwise_not(mask)
#cv2.imshow("inverted image", inverted_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
###### hola hola 
#from skimage.transform import downscale_local_mean, rescale
#img_rescaled = rescale(mask,7, anti_aliasing=False) 
#img_downscale = downscale_local_mean(img_rescaled, (1,20))
#img_rescaled2 = rescale(img_downscale, 0.3, anti_aliasing=False) 

#cv2.imshow("img transformada", img_rescaled2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
        
        
    