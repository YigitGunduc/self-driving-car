from PIL import ImageGrab

def grabscreen():
    # Reading image
    img = np.array(ImageGrab.grab(bbox=(0,40,800,680)))
    #converting image to gray scale
    img = cv2.cvtColor(img,cv2.COLORBGR2GRAY)
    #resizeing image 
    img = cv2.resize(400,320)
    # saving height, width, and channels for later use
    height, width, channels = img.shape     
    
    return img


