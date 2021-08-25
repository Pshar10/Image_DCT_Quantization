import numpy as np 
import cv2
from matplotlib import pyplot as plt

Selection = 11


pic = cv2.imread('i.jpg',0)
img=pic
plt.figure(figsize=(10,10))
plt.imshow(img,cmap='gray')
plt.show()




def Quant_matrix(Type_Q):

    Q10 = np.array([[80,60,50,80,120,200,255,255],  #fine steps for low freequency and large steps for high
                [55,60,70,95,130,255,255,255],
                [70,65,80,120,200,255,255,255],
                [70,85,110,145,255,255,255,255],
                [90,110,185,255,255,255,255,255],
                [120,175,255,255,255,255,255,255],
                [245,255,255,255,255,255,255,255],
                [255,255,255,255,255,255,255,255]])

    Q50 = np.array([[16,11,10,16,24,40,51,61],
                [12,12,14,19,26,58,60,55],
                [14,13,16,24,40,57,69,56],
                [14,17,22,29,51,87,80,62],
                [18,22,37,56,68,109,103,77],
                [24,35,55,64,81,104,113,92],
                [49,64,78,87,103,121,120,101],
                [72,92,95,98,112,100,130,99]])

    Q90 = np.array([[3,2,2,3,5,8,10,12],
                    [2,2,3,4,5,12,12,11],
                    [3,3,3,5,8,11,14,11],
                    [3,3,4,6,10,17,16,12],
                    [4,4,7,11,14,22,21,15],
                    [5,7,11,13,16,12,23,18],
                    [10,13,16,17,21,24,24,21],
                    [14,18,19,20,22,20,20,20]])
    if Type_Q == "Q10":
        return Q10
    elif Type_Q == "Q50":
        return Q50
    elif Type_Q == "Q90":
        return Q90
    else:
        return np.ones((8,8))

def Quantization(Q):

    pic = cv2.imread('i.jpg',0)

    h  = len(pic)
    w = len(pic[0])



    IMG = [] 
    block_size = 8

    Index_Y = 0 
    for i in range(block_size,h+1,block_size): 
        Index_X = 0 
        for j in range(block_size,w+1,block_size): 
            IMG.append(pic[Index_Y:i,Index_X:j]-np.ones((8,8))*128) # Normalizing the values to 0 and 1
            Index_X = j
        Index_Y = i


    
    img_float = [np.float32(img) for img in IMG]


    DCTmatrix = []
    for block in img_float:
        currDCT = cv2.dct(block) 
        DCTmatrix.append(currDCT)  #loading the DCT matrix with values
    print("The length of DCTmatrix list is image size times 8 : ",len(DCTmatrix))
    print("\n\n")



    r = 0
    r_c = []
    for j in range(int(w/block_size),len(DCTmatrix)+1,int(w/block_size)):
        r_c.append(np.hstack((DCTmatrix[r:j])))
        r = j
    reconstructed_image = np.vstack((r_c))
    img=reconstructed_image


    plt.figure(figsize=(10,10))
    plt.imshow(np.log10(abs(img)+0.0001),cmap='gray')  #showing the DCT image
    plt.show()



    MAT_S = Quant_matrix(Q)


    for k in DCTmatrix:
        for i in range(block_size):
            for j in range(block_size):
                k[i,j] = np.around(k[i,j]/MAT_S[i,j]) #element by element division




#Inverse Discrete Cosine Transform
    IDCTmatrix = []
    for i in DCTmatrix:
        curriDCT = cv2.idct(i)
        IDCTmatrix.append(curriDCT) #loading the DCT matrix with values
    # IDCTmatrix[0][0]

# Forming 8*8 blocks again

    r = 0
    r_c = []
    for j in range(int(w/block_size),len(IDCTmatrix)+1,int(w/block_size)):
        r_c.append(np.hstack((IDCTmatrix[r:j])))
        r = j
    reconstructed_image = np.vstack((r_c))

    H  = len(reconstructed_image) 
    H = str(H)
    W = len(reconstructed_image[0]) 
    W = str(W)

    img=reconstructed_image


    plt.figure(figsize=(10,10))
    plt.imshow(img,cmap='gray')
    plt.show()

    print("The resulting image is of same size as :" +H+"x"+W+"")
    print("\n\n")

while(Selection !='q'):
    print("\n\nWelcome to the Quantization segment:\n\nPress the following keys for the results\n\n")
    print("a  :  Q10 matrix")
    print("b  :  Q50 matrix")
    print("c  :  Q90 matrix")
    print("q  :  quit")
    Selection = input()


    if Selection == 'a': # more compression, less quality
        Quantization("Q10")
        pass

    elif Selection == 'b':  #best compression and best quality
        Quantization("Q50")
        pass

    elif Selection == 'c':
        Quantization("Q90") # more compression less quality
        pass






# Other way of DCT and Inverse DCT transform : As prescribed in lecture


# [r,c,d]=pic.shape
# frame=np.reshape(pic[:,:,1],(-1,8), order='C')
# X=sft.dct(frame/255.0,axis=1,norm='ortho')
# X=np.reshape(X,(-1,c), order='C')
# X=np.reshape(X.T,(-1,8), order='C')
# X=sft.dct(X,axis=1,norm='ortho')
# X=(np.reshape(X,(-1,r), order='C')).T

# #applying inverse DCT

# X=np.reshape(X,(-1,8), order='C')
# X=sft.idct(X,axis=1,norm='ortho')


# X=np.reshape(X,(-1,c), order='C')

# X=np.reshape(X.T,(-1,8), order='C')

# x=sft.idct(X,axis=1,norm='ortho')


# x=(np.reshape(x,(-1,r), order='C')).T 