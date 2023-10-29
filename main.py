import cv2
import imutils
import numpy as np

##########################
#   Generic Functions    #
##########################

def showImg(img, title="Image"):
    cv2.imshow(title, img)
    key = cv2.waitKey(0)
    
    while key != ord("q") and key != ord("Q"):
        key = cv2.waitKey(0)

    cv2.destroyAllWindows()

def resize(frame, maxHeight=800, maxWidth=1500):
    # If resize factor for height is greater than that of width
    if frame.shape[0]/maxHeight > frame.shape[1]/maxWidth:
        resizedFrame = imutils.resize(frame.copy(), height=maxHeight)
    else:
        resizedFrame = imutils.resize(frame.copy(), width=maxWidth)
    return resizedFrame, resizedFrame.shape[0]/frame.shape[0]

def getROIFromVideo(vidPath, fromCenter=False):
    vs = cv2.VideoCapture(vidPath)
    
    ret, frame = vs.read()

    # while ret:
    #     ret, frame = vs.read()
    #     dispFrame, _ = resize(frame)
    #     cv2.imshow("Video", dispFrame)
    #     key = cv2.waitKey(0)
    #     if key == ord('q') or key == 27:
    #         break

    return getROIFromFrame(frame, fromCenter)

def getROIFromFrame(frame, fromCenter=False):
    frame, scale = resize(frame)
    r = cv2.selectROI(frame, fromCenter=fromCenter)
    cv2.destroyAllWindows()

    # If ROI invalid, return whole image
    if len(np.unique(r)) < 4:
        r = np.zeros(4).astype("int")
        r[3] = len(frame)
        r[2] = len(frame[0])
    
    return np.round(np.array(r)/scale).astype("int")

def cropWithROI(frame, roi):
    return frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

def getTemplatesFromVideo(vidPath, pipeline, templateWidth, templateHeight):
    box = None
    templates = []

    def on_mouse(event, x, y, flags, userdata):
        nonlocal box
        # Draw box
        if event == cv2.EVENT_LBUTTONDOWN:
            p = (x,y)
            p1 = (int(p[0]-templateWidth/2), int(p[1]-templateHeight/2))
            p2 = (int(p[0]+templateWidth/2), int(p[1]+templateHeight/2))
            box = [p1[0], p1[1], templateWidth, templateHeight]

            frameDraw = frame.copy()
            cv2.rectangle(frameDraw, p1, p2, (255, 0, 0), 1)
            cv2.imshow('Frame', frameDraw)

    cap = cv2.VideoCapture(vidPath)
    cv2.namedWindow('Frame', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('Frame', on_mouse)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame = pipeline(frame)
        
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(0)
        if key == ord("q") or key == ord("Q"):
            break

        if key == ord("e") or key == ord("E"):
            if box is not None:
                templates.append(cropWithROI(frame, box))

    cap.release()
    cv2.destroyAllWindows()
    return templates

def getTemplateMatches(frame, template, confidenceThresh):
    res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= confidenceThresh)

    confidences = res[loc]

    boxes = []
    w, h = template.shape[-2::-1]
    for pt in zip(*loc[::-1]):
        boxes.append([*pt, pt[0] + w, pt[1] + h])

    return boxes, confidences

def getOutputVidFrameSize(vidPath, pipeline, outputHeight):
    cap = cv2.VideoCapture(vidPath)
    _, frame = cap.read()
    frame = pipeline(frame)
    frame = imutils.resize(frame, height=outputHeight)
    return frame.shape[-2::-1]

def getFrameFromVid(vidPath, time):
    cap = cv2.VideoCapture(vidPath)
    cap.set(cv2.CAP_PROP_POS_MSEC, time*1000)
    ret, frame = cap.read()
    return frame

###############################
#   Circle/Ellipse Fitting    #
###############################

def getCircleFromThreePoints(x1, y1, x2, y2, x3, y3):
    a = 2*(x1*(y2-y3) - y1*(x2-x3) + x2*y3 - x3*y2)
    centerx = (x1*x1+y1*y1)*(y2-y3) + (x2*x2+y2*y2)*(y3-y1) + (x3*x3+y3*y3)*(y1-y2)
    centerx /= a
    centery = (x1*x1 + y1*y1)*(x3-x2) + (x2*x2+y2*y2)*(x1-x3) + (x3*x3 + y3*y3)*(x2-x1)
    centery /= a
    radius = np.sqrt((centerx-x1)**2 + (centery-y1)**2)

    return centerx, centery, radius

def getEllipseFromPoints(pts):
    cnt = np.array(pts).reshape((-1,1,2)).astype(np.int32)
    ellipse = cv2.fitEllipse(cnt)
    return ellipse

def cropWithEllipse(img, ellipse):
    

class SelectionWindow():

    def __init__(self, title, frame):
        self.title = title
        self.frame = frame.copy()
        
        self.minPointsLeft = 0
        self.func = self.callback_func

        self.selectionPts = []
       
    def displayWindow(self):
        cv2.namedWindow(self.title)
        if self.func != None:
            cv2.setMouseCallback(self.title, self.func)
        cv2.imshow(self.title, self.frame)
        
        while True:
            key = cv2.waitKey(0)
            
            if self.minPointsLeft <= 0 and key == ord('q') or key == ord('Q'):
                cv2.destroyWindow(self.title) 
                break

    def callback_func(self,event,x,y,flags,param):
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
    
# Add exception for case when three points happen to line up nicely or two or more points have the same coordinates
class CircleCalibWindow(CalibWindow):
    
    def __init__(self, title, frame):
        super().__init__(title, frame)
        self.minPointsLeft = 3
        self.centerx, self.centery, self.radius = 0, 0, 0
        self.frameCopy = self.frame.copy()

    def set_calib_length(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
            self.minPointsLeft -= 1
            self.calibPoints.append((x, y))
            cv2.circle(self.frame, (x,y), 2, (255,0,0), thickness=1)
            
            if len(self.calibPoints) >= 3:
                x1, y1, x2, y2, x3, y3 = *self.calibPoints[-3], *self.calibPoints[-2], *self.calibPoints[-1]
                self.centerx, self.centery, self.radius = getCircleFromThreePoints(x1, y1, x2, y2, x3, y3)
                
                # New frame to "remove" draw circle in last frame
                self.frame = self.frame.copy()

                # Python 3.7.5: Some bug probably, round does not always convert to int
                # print(self.centerx, self.centery, self.radius)
                # print(round(self.centerx), round(self.centery), round(self.radius))
                cv2.circle(self.frame, (int(round(self.centerx)),int(round(self.centery))), int(round(self.radius)), (255,0,0), thickness=1)
            
            cv2.imshow(self.title, self.frame)

    def getCalibScale(self, calibRadius):
        return self.radius/calibRadius

    def getCircle(self):
        return (self.centerx, self.centery, self.radius)

class EllipseCalibWindow(CalibWindow):
    
    def __init__(self, title, frame):
        super().__init__(title, frame)
        self.minPointsLeft = 5
        self.frameCopy = self.frame.copy()

    def set_calib_length(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
            self.minPointsLeft -= 1
            self.calibPoints.append((x, y))
            cv2.circle(self.frame, (x,y), 2, (255,0,0), thickness=1)
            
            if len(self.calibPoints) >= 5:
                ellipse = getEllipseFromPoints(self.calibPoints)
                ellipse = tuple(map(lambda l: np.round(l, decimals=0).astype(int), ellipse))
                
                # New frame to "remove" draw circle in last frame
                self.frame = self.frame.copy()
                ((centx,centy), (width,height), angle) = ellipse
                cv2.ellipse(self.frame, (int(centx),int(centy)), (int(width/2),int(height/2)), angle, 0, 360, (0,0,255), 1)

                

            cv2.imshow(self.title, self.frame)