from tkinter import *
from tkinter import ttk,filedialog
import cv2, math, pywt
import numpy as np
from operator import itemgetter
from skimage.transform import warp, AffineTransform
from matplotlib import pyplot as plt


window = Tk() 
window.title("Procesamiento de imagenes")
window.geometry('800x140')

#Variables
img_left_clr=None
win_size = 50
max_iters = 1000
epsilon = 100
verbose = True
CANTIDADMATCH = None


#Métodos de ejecución.

def ransac(pts1, pts2, img_l, img_r, max_iters=1000, epsilon=1):
    best_matches = []
    # Number of samples
    N = 4
    
    for i in range(max_iters):
        # Get 4 random samples from features
        idx = np.random.randint(0, len(pts1) - 1, N)
        src = pts1[idx]
        dst = pts2[idx]

        # Calculate the homography matrix H
        H = cv2.getPerspectiveTransform(src, dst)
        Hp = cv2.perspectiveTransform(pts1[None], H)[0]

        # Find the inliers by computing the SSD(p',Hp) and saving inliers (feature pairs) that are SSD(p',Hp) < epsilon
        inliers = []
        for i in range(len(pts1)):
            ssd = np.sum(np.square(pts2[i] - Hp[i]))
            if ssd < epsilon:
                inliers.append([pts1[i], pts2[i]])
        
        # Keep the largest set of inliers and the corresponding homography matrix
        if len(inliers) > len(best_matches):
            best_matches = inliers
    print("best_matches")
    print(len(best_matches))
    return best_matches

def mops(img, truth, win_size, h, w, r):
    H, W = img.shape
    offset = win_size // 2
    
    # Draw line for angle of gradient (for debugging)
    #length = 150
    #h2 =  int(h - length * math.cos(math.radians(r)))
    #w2 =  int(w - length * math.sin(math.radians(r)))
    #cv2.line(img, (w, h), (w2, h2), (0,255,0), 2)

    # Rotate image s.t. gradient angle of feature is origin
    M = cv2.getRotationMatrix2D((w, h), -1*r, 1)
    img_rot = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_NEAREST)

    # Get 40x40 window around feature
    win = img_rot[h-offset:h+offset, w-offset:w+offset]

    # Size (s,s) of each sample region
    s = win_size // 8
    # Prefiltering (N,N) -> (N//s, N//s)
    i = 0
    rows = []
    while i < win_size:
        j = 0
        cols = []
        while j < win_size:
            # Sample (s,s) region from window
            sample = win[i:i+s, j:j+s]
            # Downsample (s,s) region to a single value
            sample = np.sum(sample) / (s*s)
            cols.append(sample)
            j += s
        rows.append(cols)
        i += s
    
    feature = np.array(rows)
    # Normalize
    feature = (feature - np.mean(feature)) / np.std(feature)
    # Haar wave transform
    coeffs = pywt.dwt2(feature, 'haar')
    feature = pywt.idwt2(coeffs, 'haar')
    #plot([img, img_rot, truth, win, feature])
    return feature

def get_matches(ft1, ft2, left_match, right_match, img_left, img_right, max_matches, win_size):
    potential_matches = []
    offset = win_size // 2

    # Feature descriptor with brute-force matching
    for h1, w1, r1 in ft1:
        
        # Copy image as to not modify it by reference
        img_left_tmp = np.copy(img_left)

        # Copy coordinates incase changed by border
        h1_tmp = h1
        w1_tmp = w1

        # Get 40x40 window around feature
        win_left = img_left[h1-offset:h1+offset, w1-offset:w1+offset]

        # Handle border points
        if win_left.shape != (win_size, win_size):
            diff_h, diff_w = np.subtract(win_left.shape, (win_size, win_size))
            p_h = abs(diff_h // 2)
            p_w = abs(diff_w // 2)
            img_left_tmp = cv2.copyMakeBorder(img_left_tmp, p_h, p_h, p_w, p_w, cv2.BORDER_REFLECT)
            h1 += p_h
            w1 += p_w
            win_left = img_left_tmp[h1-offset:h1+offset, w1-offset:w1+offset]
        
        # Run multiscale oriented patches descriptor
        feature_left = mops(img_left_tmp, win_left, win_size, h1, w1, r1)

        lowest_dist = math.inf
        potential_match = ()
        for h2, w2, r2 in ft2:

            # Copy image as to not modify it by reference
            img_right_tmp = np.copy(img_right)

            # Copy coordinates incase changed by border
            h2_tmp = h2
            w2_tmp = w2

            # Get 40x40 window around feature
            win_right = img_right[h2-offset:h2+offset, w2-offset:w2+offset]

            # Handle border points
            if win_right.shape != (win_size, win_size):
                diff_h, diff_w = np.subtract(win_right.shape, (win_size, win_size))
                p_h = abs(diff_h // 2)
                p_w = abs(diff_w // 2)
                img_right_tmp = cv2.copyMakeBorder(img_right_tmp, p_h, p_h, p_w, p_w, cv2.BORDER_REFLECT)
                h2 += p_h
                w2 += p_w
                win_right = img_right_tmp[h2-offset:h2+offset, w2-offset:w2+offset]

            # Run multiscale oriented patches descriptor
            feature_right = mops(img_right_tmp, win_right, win_size, h2, w2, r2)
            
            # Check distance between features
            curr_dist = np.linalg.norm(feature_left - feature_right)
            if curr_dist < lowest_dist:
                lowest_dist = curr_dist
                potential_match = ([h1_tmp, w1_tmp, r1], [h2_tmp, w2_tmp, r2], curr_dist)
        
        potential_matches.append(potential_match)
        
    # Sort matches from smallest distance up
    matches = sorted(potential_matches, key=itemgetter(2))
    for match in matches:
        # Ensure no duplicates
        if match[0][0:2] not in left_match and match[1][0:2] not in right_match:
            # Add to matches
            left_match.append(match[0][0:2])
            right_match.append(match[1][0:2])
            # Remove from potential matches
            ft1.remove(tuple(match[0]))
            ft2.remove(tuple(match[1]))
    
    # Recursively keep going until every point has a match
    while(len(left_match) < max_matches and len(right_match) < max_matches):
        print('Otra coincidencia')
        get_matches(ft1, ft2, left_match, right_match, img_left, img_right, max_matches, win_size)

    return np.array(left_match, dtype=np.float32), np.array(right_match, dtype=np.float32)

def read_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("No se pudo acceder a la imagen con la ruta "+str(image_path))
    return img

def harris(img, sigma=1, threshold=0.01):
    height, width = img.shape
    shape = (height, width)
    # Calculate the dx,dy gradients of the image (np.gradient doesnt work)
    dx = cv2.Sobel(img, cv2.CV_64F,1,0,ksize=5)
    dy = cv2.Sobel(img, cv2.CV_64F,0,1,ksize=5)
    # Get angle for rotation
    _, ang = cv2.cartToPolar(dx, dy, angleInDegrees=True)
    # Square the derivatives (A,B,C of H) and apply apply gaussian filters to each
    sigma = (sigma, sigma)
    Ixx = cv2.GaussianBlur(dx * dx, sigma, 0)
    Ixy = cv2.GaussianBlur(dx * dy, sigma, 0)
    Iyy = cv2.GaussianBlur(dy * dy, sigma, 0)
    H = np.array([[Ixx, Ixy], [Ixy, Iyy]])
    # Find the determinate
    num = (H[0, 0] * H[1, 1]) - (H[0, 1] * H[1, 0])
    # Find the trace
    denom = H[0,0] + H[1,1]
    # Find the response using harmonic mean of the eigenvalues (Brown et. al. variation) 
    # print("num")
    # print(num)
    # print("denom")
    # print(denom)
    R = np.nan_to_num(num / denom)
    #np.seterr(divide='ignore', invalid='ignore')
    
    # Adaptive non-maximum suppression, keep the top 1% of values and remove non-maximum points in a 9x9 neighbourhood
    R_flat = R[:].flatten()
    # Get number of values in top threshold %
    N = int(len(R_flat) * threshold)
    # Get values in top threshold %
    top_k_percentile = np.partition(R_flat, -N)[-N:]
    # Find lowest value in top threshold %
    minimum = np.min(top_k_percentile)
    # Set all values less than this to 0
    R[R < minimum] = 0
    # Set non-maximum points in an SxS neighbourhood to 0
    s = 9
    for h in range(R.shape[0] - s):
        for w in range(R.shape[1] - s):
            maximum = np.max(R[h:h+s+1, w:w+s+1])
            for i in range(h, h+s+1):
                for j in range(w, w+s+1):
                    if R[i, j] != maximum:
                        R[i, j] = 0
                        
    # Return harris corners in [H, W, R] format
    features = list(np.where(R > 0))
    features.append(ang[np.where(R > 0)])
    corners = zip(*features)
    return list(corners)

def draw_corners(corners, img, name):
    for h, w, r in corners:
        cv2.circle(img, (w, h), 2, (0, 0, 255))

    cv2.imwrite(name + '.png', img)

def draw_matches(matches, img_left, img_right,mov, verbose=False):
    # Determine the max height
    height = max(img_left.shape[0], img_right.shape[0])
    # Width is the two images side-by-side
    width = img_left.shape[1] + img_right.shape[1]

    img_out = np.zeros((height, width, 3), dtype=np.uint8)
    # Place the images in the empty image 
    img_out[0:img_left.shape[0], 0:img_left.shape[1], :] = img_left
    img_out[0:img_right.shape[0], img_left.shape[1]:, :] = img_right

    # The right image coordinates are offset since the image is no longer at (0,0)
    ow = img_left.shape[1]
   
    #Draw a line between the matched pairs in green
    for p1,p2 in matches:
        p1o = (int(p1[1]), int(p1[0]))
        p2o = (int(p2[1] + ow), int(p2[0]))
        color = list(np.random.random(size=3) * 256)
        cv2.line(img_out, p1o, p2o, color, thickness=2)

    if verbose:
        print("Press enter to continue ... ")
        cv2.imshow(nameimg, img_out)
        cv2.waitKey(0)
    cv2.imshow("nameimg", img_out)
    cv2.imwrite(mov+".png", img_out)
    cv2.waitKey(0)

#Métodos de modificación de la imagen
def translate(img, point): #Mueve la imagen
    newimg = img
    rows,cols, _ = newimg.shape
    M = np.float32([[1,0,point[0]],[0,1,point[1]]])
    dst = cv2.warpAffine(newimg,M,(cols,rows))
    return dst

def scale(img, ratio):
    newimg = img
    width = int(newimg.shape[1] * ratio / 100)
    height = int(newimg.shape[0] * ratio / 100)
    dsize = (width, height)
    output = cv2.resize(newimg, dsize)
    return output

def rotate(img, angle):
    newimg = img
    rows,cols, _ = newimg.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    dst = cv2.warpAffine(newimg,M,(cols,rows))
    return dst

#Métdos iniciales
def openImage():
    global img_left_clr
    window.filename =  filedialog.askopenfilename(initialdir = "./",title = "Select file",filetypes = (("Photo files","*.png"),("all files","*.*")))
    imgURL= window.filename
    asd= imgURL.split("/")
    nameimg = asd[-1]
    print(nameimg)
    img_left_clr = cv2.imread(imgURL)
    print(window.filename)
    lbl = Label(window, text="La dirección de la imagen es "+imgURL)
    lbl.place(x=140, y=10)

def doMove():
    CANTIDADMATCH = int(matchmax.get())
    movx = int(moverx.get())
    movy = int(movery.get())
    mover = [movx , movy]
    img_right_clr = translate(img_left_clr,mover)
    img_left = cv2.cvtColor(img_left_clr, cv2.COLOR_BGR2GRAY)
    img_right = cv2.cvtColor(img_right_clr, cv2.COLOR_RGB2GRAY)
    mov = "mover_"+str(movx)+"_"+str(movy)
    doAlguno(img_right_clr,img_left_clr,img_right,img_left,CANTIDADMATCH,mov)
   
def doScalar():
    CANTIDADMATCH = int(matchmax.get())
    ratio = int(numescalar.get())
    img_right_clr = scale(img_left_clr,ratio)
    img_left = cv2.cvtColor(img_left_clr, cv2.COLOR_BGR2GRAY)
    img_right = cv2.cvtColor(img_right_clr, cv2.COLOR_RGB2GRAY)
    namesc = "escalar_"+str(ratio)
    doAlguno(img_right_clr,img_left_clr,img_right,img_left,CANTIDADMATCH,namesc)

def doRotate():
    CANTIDADMATCH = int(matchmax.get())
    angle2 = int(angle.get())
    img_right_clr = rotate(img_left_clr,angle2)
    img_left = cv2.cvtColor(img_left_clr, cv2.COLOR_BGR2GRAY)
    img_right = cv2.cvtColor(img_right_clr, cv2.COLOR_RGB2GRAY)
    rotname = "Rotar_"+str(angle2)
    doAlguno(img_right_clr,img_left_clr,img_right,img_left,CANTIDADMATCH,rotname)

def doAlguno(img_right_clr,img_left_clr,img_right,img_left,CANTIDADMATCH,imname):
    print("Getting the features from the Harris Detector")
    ftL = harris(img_left, sigma=3, threshold=0.01)
    draw_corners(ftL, img_left_clr, 'corners_left')
    ftR = harris(img_right, sigma=3, threshold=0.01)
    draw_corners(ftR, img_right_clr, 'corners_right')
    print("  -- Number of features (left): ", len(ftL))
    print("  -- Number of features (right): ", len(ftR))
    print("Finding the best matches between images")
    max_matches = min(len(ftL), len(ftR))
    print(max_matches)
    ptsL,ptsR = get_matches(ftL, ftR, [], [], img_left, img_right, max_matches, win_size=win_size)
    print(" -- Nmber of matches = ", len(ptsL))
    assert len(ptsL) == len(ptsR)
    print("Performing RANSAC")
    matches = ransac(ptsL,ptsR, img_left, img_right, max_iters, epsilon)
    lenmatches = len(matches)
    matches = matches[:CANTIDADMATCH] #Obtiene los 6 primeros
    print(" -- Number of pruned matches = ", len(matches))
    draw_matches(matches, img_left_clr, img_right_clr,imname)


#-------------------> Elementos de la vista
btn = Button(window, text="Selecciona una imagen",command=openImage)
btn.place(x=0, y=10)

btn2 = Button(window, text="Mover",command=doMove)
btn2.place(x=0, y=40)
moverx = ttk.Entry(window)
moverx.place(x=100, y=40)
movery = ttk.Entry(window)
movery.place(x=200, y=40)

btn3 = Button(window, text="Resize",command=doScalar)
btn3.place(x=0, y=70)
numescalar = ttk.Entry(window)
numescalar.place(x=100, y=70)



btn4 = Button(window, text="Rotar",command=doRotate)
btn4.place(x=0, y=100)
angle = ttk.Entry(window)
angle.place(x=100, y=100)

matchmax = ttk.Entry(window)
matchmax.place(x=600, y=100)
lbl2 = Label(window, text="Numero máximo de match")
lbl2.place(x=440, y=100)

if __name__ == "__main__":
    
    window.mainloop()