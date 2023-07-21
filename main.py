import cv2
import imutils
import numpy as np

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
        _, frame = cap.read()
        frame = pipeline(frame)
        
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(0)
        if key == ord("q") or key == ord("Q"):
            cap.release()
            cv2.destroyAllWindows()

        if key == ord("e") or key == ord("E"):
            if box is not None:
                templates.append(cropWithROI(frame, box))

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