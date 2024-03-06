# Indicate color selected
# Primitives like square and circle (primitives)
# File types like MOV or capture using a specific video type
import cv2
import numpy as np
from imutils.video import VideoStream
import imutils
import sympy as sym
from sympy import sin as sin
from sympy import cos as cos
import pandas as pd
import glob

sym.init_printing(pretty_print=False)

# shape = sd.detect(c)

# # multiply the contour (x, y)-coordinates by the resize ratio,
# # then draw the contours and the name of the shape on the image
# c = c.astype("float")
# c *= ratio
# c = c.astype("int")

# cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
#             0.5, (255, 255, 255), 2)

def goToPointInVideo(vid, startTime, fps, vidEncode=False):
    # startTime in seconds
    if vidEncode: # For some video encoding types, vid.set doesn't work
        vid.set(cv2.CAP_PROP_POS_MSEC, startTime*1000)
    else:
        frameNumber = int(round(fps*startTime))
        vid.set(cv2.CAP_PROP_POS_FRAMES, frameNumber)

def showImg(frame, title="Image"):
    cv2.imshow(title, frame.copy())
    key = cv2.waitKey(0)
    if key == 27 or key == ord('q'):
        cv2.destroyAllWindows()

def displayVideo(title, imgWidth=1500):
    vs = cv2.VideoCapture(title)
    ret, frame = vs.read() # grab a frame
    while ret:
        frame = imutils.resize(frame, width=imgWidth)
        cv2.imshow("Display", frame.copy())
        key = cv2.waitKey(0)
        if key == 27 or key == ord("q"): # if "Esc" or "q" is pressed
            cv2.destroyAllWindows()
            break
        ret, frame = vs.read()
    print("End of video")

def getROIFromVideo(title, imgWidth=1500):
    vs = cv2.VideoCapture(title)
    
    ret, frame = vs.read()

    while ret:
        ret, frame = vs.read()
        frame = resize(frame)
        cv2.imshow("Video", frame)
        key = cv2.waitKey(0)
        if key == ord('q') or key == 27:
            cv2.destroyAllWindows()
            r = cv2.selectROI(frame)
            break
    cv2.destroyAllWindows()

    # If ROI invalid, return whole image
    if len(np.unique(r)) < 4:
        r = np.zeros(4).astype("int")
        r[3] = len(frame)
        r[2] = len(frame[0])
    
    return r

def matchColor(boundary, frame):
    lower, upper = np.array(boundary, dtype="uint8")
    mask = cv2.inRange(frame, lower, upper)
    output = cv2.bitwise_and(frame, frame, mask = mask)    
    return mask, output
            
def evaluateCircleMatch(edges, centreX, centreY, radius):
    mask = np.zeros(edges.shape, dtype=np.uint8)
    circle = cv2.circle(mask, (int(centreX), int(centreY)), int(radius), (255, 255, 255), 1) # Change remove "circle ="?
    ballEdge = edges & mask
    numCorr = len(np.nonzero(ballEdge)[0])
    return numCorr

def detectCircleMorph(frameSaturation, ballRadiusEst):
    '''
    ballRadiusEst: In pixels
    '''
    # Convert to binary image and close small "holes" in the ball
    # gray = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(frameSaturation,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # thresh = cv2.bitwise_not(thresh) # Ensure white object in black background
    element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(5, 5))
    morph_img = thresh.copy()
    cv2.morphologyEx(src=thresh, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img)
    # Haven't tried but probably inferior alternative: Canny

    # Find contours
    _,contours,_ = cv2.findContours(morph_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    sorted_areas = np.sort(areas)
    # cv2.drawContours(frame_color, contours, -1, (0,255,0), 3)
    cnt=contours[areas.index(sorted_areas[-1])] #the biggest contour

    # Bounding box (red)
    # r = cv2.boundingRect(cnt)
    # cv2.rectangle(frame_color,(r[0],r[1]),(r[0]+r[2],r[1]+r[3]),(0,0,255),1)

    # Min circle (green)
    # (x,y),radius = cv2.minEnclosingCircle(cnt)
    # center = (int(x),int(y))
    # radius = int(radius)
    # cv2.circle(frame_color,center,radius,(0,255,0),1)

    # Fit ellipse (blue)
    ellipse = cv2.fitEllipse(cnt)
    ellipseCenter, ellipseAxes, _ = ellipse
    if np.abs(ellipseAxes[0]/2-ballRadiusEst) < np.abs(ellipseAxes[1]/2-ballRadiusEst):
        circleRadius = int(np.rint(ellipseAxes[0]/2))
    else:
        circleRadius = int(np.rint(ellipseAxes[1]/2))    
    circleCenter = tuple(np.rint(ellipseCenter).astype(int))
    return ellipse, circleCenter, circleRadius, morph_img

def detectCircleRANSAC(maxIterations, edgesFrame):
    bestCenterX = 0
    bestCenterY = 0
    bestR = 0
    bestNumCorr = 0
    
    edge_coords = np.nonzero(edges)
    
    for i in range(maxIterations):
        
        numCorr = 0
        coord1 = 0
        coord2 = 0
        coord3 = 0

        while True:
            
            coord1 = np.random.randint(0, len(edge_coords[0]))
            coord2 = np.random.randint(0, len(edge_coords[0]))
            coord3 = np.random.randint(0, len(edge_coords[0]))

            if coord1 == coord2 or coord1 == coord3 or coord2 == coord3:
                continue

            x1 = edge_coords[1][coord1]
            y1 = edge_coords[0][coord1]
            x2 = edge_coords[1][coord2]
            y2 = edge_coords[0][coord2]
            x3 = edge_coords[1][coord3]
            y3 = edge_coords[0][coord3]

            if x1 == x2 or x2 == x3 or x1 == x3:
                continue
            if y1 == y2 or y2 == y3 or y1 == y3:
                continue
            if abs((y2-y1)/(x2-x1)) == abs((y3-y2)/(x3-x2)):
                continue

            break
        
        centerx, centery, radius = getCircleFromThreePoints(x1, y1, x2, y2, x3, y3) 

        mask = np.zeros(edgesFrame.shape, dtype=np.uint8)
        circle = cv2.circle(mask, (int(centerx), int(centery)), int(radius), (255, 255, 255), 1)

        ballEdge = edgesFrame & mask
        numCorr = len(np.nonzero(ballEdge)[0])

        if numCorr >= bestNumCorr:
            bestNumCorr = numCorr
            bestCenterX = centerx
            bestCenterY = centery
            bestR = radius
    return bestCenterX, bestCenterY, bestR

def rotateNoScaling(frame, angle=90):
    rows,cols,_ = frame.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    return cv2.warpAffine(frame,M,(cols,rows))

def getPositionVectors(r1, r2, theta1, theta2):
    # Position vector of point 1 in the frame of camera 1
    p1 = [np.sqrt(R**2 - r1**2), r1*np.sin(theta1), r1*np.sin(theta1)]
    # Position vector of point 2 in the frame of camera 2
    p2Prime = [np.sqrt(R**2 - r2**2), r2*np.sin(theta2), r2*np.sin(theta2)]
    # Position vector of point 2 in the frame of camera 1
    p2 = [-p2Prime[1], p2Prime[0], p2Prime[2]]
    return p1, p2

def solveAngles(p1, p1_dt, p2, p2_dt):
    e1 = rotMatrix*p1 - p1_dt
    e2 = rotMatrix*p2 - p2_dt
    sol = sym.solvers.solvers.nsolve((e1, e2), (alpha, beta, gamma), (1, 1, 1))

# Not yet fully implemented
def getTaitBryanAngles(r1, r2, theta1, theta2,
                       r1_dt, r2_dt, theta1_dt, theta2_dt):
    p1, p2= getPositionVectors(r1, r2, theta1, theta2)
    p1_dt, p2_dt = getPositionVectors(r1_dt, r2_dt, theta1_dt, theta2_dt)

class SelectionWindow():

    def __init__(self, title, frame):
        self.title = title
        self.frame = frame.copy()
        
        self.minPointsLeft = 0
        self.func = self.callbackFunc

        self.selectionPts = []
       
    def displayWindow(self):
        cv2.namedWindow(self.title)
        if self.func != None:
            cv2.setMouseCallback(self.title, self.func)
        cv2.imshow(self.title, self.frame)
        
        while True:
            key = cv2.waitKey(0)
            
            if self.minPointsLeft <= 0 and (key == ord('q') or key == ord('Q')):
                cv2.destroyWindow(self.title) 
                break

    def callbackFunc(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
            self.minPointsLeft -= 1
            self.selectionPts.append((x, y))
            cv2.circle(self.frame, (x,y), 2, (255,255,255), thickness=1)
            cv2.imshow(self.title, self.frame)

class CalibWindow(SelectionWindow):

    def __init__(self, title, frame):
        super().__init__(title, frame)
        self.calibPoints = []
        self.minPointsLeft = 2
        self.func = self.set_calib_length

    def set_calib_length(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
            self.minPointsLeft -= 1
            self.calibPoints.append((x, y))
            cv2.circle(self.frame, (x,y), 2, (255,255,255), thickness=1)
            cv2.imshow(self.title, self.frame)
    
    def getCalibScale(self, calibLength):
        p1 = self.calibPoints[-1]
        p2 = self.calibPoints[-2]
        return ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5/calibLength

class PickColorWindow(SelectionWindow):

    def __init__(self, title, frame):
        super().__init__(title, frame)
        self.pickedColors = []
        self.minPointsLeft = 1
        self.func = self.pick_color

    def pick_color(self, event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
            self.minPointsLeft -= 1
            self.pickedColors.append(self.frame[y,x])
            cv2.circle(self.frame, (x,y), 2, (255,255,255), thickness=1)
            cv2.imshow(self.title, self.frame)

    def getPickedColors(self):
        return np.mean(self.pickedColors, axis=1)

if __name__ == "__main__":

    # Exclude all track and .trk files
    allFiles = np.array(glob.glob("Test 1/*"))
    files = allFiles[np.flatnonzero(np.core.defchararray.find(allFiles,"track")==-1)]
    files = files[np.flatnonzero(np.core.defchararray.find(files,".trk")==-1)]

    data = []

    for title in files:
        
        print(title)
        img = cv2.imread(title)
        
        # Resize
        height, width, depth = img.shape
        img_resized = cv2.resize(img,(int(width/3),int(height/3)))

        # Truncate
        h = len(img_resized)
        w = len(img_resized[0])
        img_trunc = img_resized[int(9*h/18):int(h*30/48)][:,int(0*w/9):int(4*w/9)]

        # Convert to HSV and split
        hsv = cv2.cvtColor(img_trunc, cv2.COLOR_BGR2HSV)
        (h,s,v) = cv2.split(hsv)
        
    #     # Pick color
        def pick_color(event,x,y,flags,param):
            if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
                print(s[y,x])
                cv2.circle(s, (x,y), 2, (255,255,255), thickness=1)
                cv2.imshow("Color Selector", s)

        # cv2.namedWindow("Color Selector")
        # cv2.setMouseCallback("Color Selector", pick_color)
        # display("Color Selector", s)
        # break

        # Mask saturation for ball
        mask = cv2.inRange(s, 130, 255)
        # display("Color Selector", mask)
        # break

        # Detect edges
        edges = cv2.Canny(mask,100,200)
        # display("Edges", edges)
        # break

        if len(np.nonzero(edges)[0]) < 3:
            continue
        
        # Run RANSAC circle detection
        maxIterations = 2000
        print("Running", maxIterations, "iterations")
        bestCenterX, bestCenterY, bestR = detectCircleRANSAC(maxIterations, edges)

        # Manually detect funnel edge
        cv2.circle(img_trunc, (int(bestCenterX), int(bestCenterY)), int(bestR), (255, 255, 255), 1)
        img_trunc_copy = img_trunc.copy()
        
        def pickEdge(event,x,y,flags,param):
            global yfunnel
            if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
                cv2.circle(img_trunc_copy, (x,y), 2, (255,255,255), thickness=1)
                cv2.imshow("Edge Selector", img_trunc_copy)
                yfunnel = y
                
        cv2.namedWindow("Edge Selector")
        cv2.setMouseCallback("Edge Selector", pickEdge)
        cv2.imshow("Edge Selector", img_trunc_copy)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        
        if key == 27 or key == ord("q"): # if "Esc" or "q" is pressed
            continue
        
        calibScale = 0.02955/(bestR*2)
        ydisp = (bestCenterY - yfunnel)*calibScale # Note, y axis of cv2 is flipped. Origin at top left hand corner of image

        cv2.circle(img_trunc, (int(bestCenterX), int(yfunnel)), 2, (0, 255, 0), -1)
        cv2.imwrite("./"+title.split(".")[0]+"track "+str(ydisp)+".JPG", img_trunc)
        data.append([title.split("\\")[1][:-4], ydisp])

    dataCopy = np.array(data)

    pd.DataFrame(dataCopy)

    # mask, output = matchColor(boundary, s, "HSV")
    # coord = cv2.findNonZero(mask)
