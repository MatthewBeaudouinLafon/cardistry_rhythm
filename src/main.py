import cv2 as cv
import numpy as np

VIDEO_PATH = 'video/whiplash.mov'

def play_video(video_path):
    cap = cv.VideoCapture(video_path)

    while(cap.isOpened()):
        ret, frame = cap.read()
        
        if frame is None:
            print("end of video")
            break

        cv.imshow('frame', frame)  # TODO: Probably should check frame size on video end so it doesn't exit with an error
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()
    cap.release()

def lk_optical_flow(video_path):
    """
    Calculates Lucas-Kanade optical flow on video. Code taken from
        https://docs.opencv.org/3.4/d7/d8b/tutorial_py_lucas_kanade.html
    """
    cap = cv.VideoCapture(video_path)

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                    maxLevel = 2,
                    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    while(1):
        ret,frame = cap.read()

        if frame is None:
            print("end of video")
            break

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
        img = cv.add(frame,mask)
        cv.imshow('frame',img)
        k = cv.waitKey(30) & 0xff

        if k == 27:
            break
            
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
    cv.destroyAllWindows()
    cap.release()

def dense_optical_flow(video_path):
    """
    Calculates dense optical flow on video (ie does it on every pixel).
    Code taken from
        https://docs.opencv.org/3.4/d7/d8b/tutorial_py_lucas_kanade.html
    """
    cap = cv.VideoCapture(video_path)
    ret, frame1 = cap.read()
    prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    while(1):
        ret, frame2 = cap.read()

        if frame2 is None:
            print("end of video")
            break

        next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
        cv.imshow('frame2',bgr)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv.imwrite('opticalfb.png',frame2)
            cv.imwrite('opticalhsv.png',bgr)
        prvs = next
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    # lk_optical_flow(VIDEO_PATH)
    dense_optical_flow(VIDEO_PATH)