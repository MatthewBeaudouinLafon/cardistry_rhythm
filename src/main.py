import math
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

class RhythmDetector(object):
    """

    """

    def __init__(self, video_path):
        self.video_path = video_path

        plt.ion()
        self.fig, (self.ax_distribution, self.ax_metrics) = plt.subplots(nrows=2, ncols=1)
        self.metrics = {
            'average': [],
            'percentiles': []
        }

    def dense_optical_flow(self):
        """
        Calculates dense optical flow on video (ie does it on every pixel).
        Code taken from
            https://docs.opencv.org/3.4/d7/d8b/tutorial_py_lucas_kanade.html
        """
        cap = cv.VideoCapture(self.video_path)
        total_frame_number = cap.get(cv.CAP_PROP_FRAME_COUNT)
        ret, frame = cap.read()
        previous_bw = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame)
        hsv[...,1] = 255

        while(True):
            ret, frame = cap.read()
            frame_number = cap.get(cv.CAP_PROP_POS_FRAMES)

            if frame is None:
                print("end of video")
                break

            # Show frame in one window
            cv.imshow('frame', frame)

            # Calculate flow
            current_bw = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
            flow = cv.calcOpticalFlowFarneback(previous_bw, current_bw, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            self.motion_metric(flow, frame_number, total_frame_number, verbose=True)

            # Make color viz in another window
            mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
            bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
            cv.imshow('flow',bgr)

            # Interpret keyboard
            k = cv.waitKey(30) & 0xff
            if k == 27:
                print("User stopped")
                break
            elif k == ord('s'):
                cv.imwrite('opticalfb.png',frame)
                cv.imwrite('opticalhsv.png',bgr)

            previous_bw = current_bw

        cap.release()
        cv.destroyAllWindows()

    def update_plots(self, total_frame_number):
        # Display pixel motion distribution histogram
        self.ax_distribution.set_xlabel('Motion')
        self.ax_distribution.set_ylabel('Frequency')
        self.ax_distribution.set_xlim(0, 100)  # TODO: no magic numbers
        self.ax_distribution.set_ylim(0, 60000)
        self.ax_distribution.set_title('Distribution of pixel motion')

        self.ax_metrics.set_xlabel('time (frames)')
        self.ax_metrics.set_ylabel('metric')
        self.ax_metrics.set_xlim(0, total_frame_number)
        self.ax_metrics.set_ylim(0, 6)

        # plt.show(block=False)
        self.fig.canvas.draw()

    def motion_metric(self, flow, frame_number, total_frame_number, verbose=False):
        """
        Calculate the motion metric from the flow. Currently includes per frame
        - an average
        magnitude.
        TODO: Try doing a second derivative like thing by taking the difference
        between the current and the previous flow. This would show those sharp
        corners that function as beats.
        """
        mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
        flat_mag = mag.flatten()

        # Compute metrics
        self.ax_distribution.clear()
        n, bins, _ = self.ax_distribution.hist(x=flat_mag, bins=30)  # TODO: Don't be lazy and use a real histogram thing?
        
        average_motion = np.mean(flat_mag)
        percentiles = [np.percentile(flat_mag, p) for p in range(25, 75+1, 25)]  # 75+1 because range is exclusive 💅
        self.metrics['average'].append(average_motion)
        self.ax_metrics.clear()
        self.ax_metrics.plot(self.metrics['average'])

        if verbose:
            self.update_plots(total_frame_number)
            print("average motion: {}".format(average_motion))

        

    def play_video(self):
        cap = cv.VideoCapture(self.video_path)

        while(cap.isOpened()):
            ret, frame = cap.read()
            
            if frame is None:
                print("end of video")
                break

            cv.imshow('frame', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        cv.destroyAllWindows()
        cap.release()

    def lk_optical_flow(self):
        """
        Calculates Lucas-Kanade optical flow on video. Code taken from
            https://docs.opencv.org/3.4/d7/d8b/tutorial_py_lucas_kanade.html
        """
        cap = cv.VideoCapture(self.video_path)

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


if __name__ == "__main__":
    rhy_det = RhythmDetector(video_path='video/whiplash.mov')

    # lk_optical_flow(VIDEO_PATH)
    rhy_det.dense_optical_flow()
