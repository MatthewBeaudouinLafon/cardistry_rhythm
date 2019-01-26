import os
import math
import time
from enum import Enum
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

from MetricManager import MetricManager

# IDEAS: 
# - Analyze motion by subtracting average flow to each flow vector.
# Removes overall movement and makes unique movement pop out in viz.
# - Make flow viz a mask on the real vid. No motion = black pixel,
# most motion = pixel like in original vid. 
# TODO: Stop being lazy and move these ideas to an actual document or GH issues. 


class RhythmDetector(object):
    """
    This class handles doing optical flow analysis on a video. It also plots
    stuff. 

    Usage:
    >>> rhythm = RhythmDetector(video_name='sybil_cut.mov')

    >>> rhythm.analyze(method='dense_optical_flow', verbose=True, metrics=[average, percentiles, second_div])
    [=======================] 100%

    >>> rhythm.play()  # TODO: Implement
    """

    def __init__(self, video_name, video_folder_path='video/'):
        self.video_name = video_name
        self.video_path = os.path.join('video/', video_name)
        self.percentiles = [50, 75]

        self.metrics = None  # MetricManager instance

        self.total_frames = None
        self.current_frame = 0

    def analyze(self,
        method='dense_optical_flow',
        chosen_metrics = [ 
                            ['flow_mag_distribution'],
                            ['flow_ang_distribution'],
                            ['mean_flow_magnitude', 'flow_percentiles', 'net_flow_mag', 'net_flow_ang']
                          ],
        show_viz=True,  # TODO: Rethink what these options should be.
        save_all=True,
        verbose=True):
        """
        Analyze video using optical flow. 

        Inputs:
            method(string): 
                one of 'dense_optical_flow'
            
            chosen_metrics ([string]):
                Nested list of metrics. Inner list represents metrics in a subplot.
                A histogram takes a whole subplot.
                Note: You can only have *three* subplots.
            
            show_viz (bool):
                Show visualizations while running
            
            verbose (bool):
                Print metrics.
            
            save_all (bool):
                Save metrics, plots, and viz if enabled. 
        """
        print("Analyzing '{}'".format(self.video_name))
        # Choose method
        if method == 'dense_optical_flow':
            print("Analyzing using Optical Flow")
            calculate_flow = self.dense_optical_flow
        else:
            print('Method "{}" unknown.'.format(method))
            return

        self.metrics = MetricManager(desired_metrics=chosen_metrics)

        # Setup Open CV
        cap = cv.VideoCapture(self.video_path)
        _, frame = cap.read()
        previous_bw = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

        self.total_frame_number = cap.get(cv.CAP_PROP_FRAME_COUNT)

        vid_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        vid_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        self.metrics.setup(
            source_vid_length=self.total_frame_number,
            source_vid_width=vid_width,
            source_vid_height=vid_height,
            source_vid_fps=cap.get(cv.CAP_PROP_FPS),
            out_vid_path='result/',
            out_vid_name=self.video_name
        )

        while(True):  # frame goes None when video stops
            # Read Frame
            _, frame = cap.read()
            self.current_frame = cap.get(cv.CAP_PROP_POS_FRAMES)

            if frame is None:
                print("end of video")
                break

            # Do Flow
            current_bw = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
            flow = calculate_flow(previous_bw, current_bw)

            # Compute Metrics
            self.metrics.compute_metrics(flow)

            if show_viz:
                self.metrics.plot_metrics(flow, frame)

                # Show frame in one window
                cv.imshow('frame', frame)

            # Interpret keyboard
            k = cv.waitKey(30) & 0xff
            if k == 27:
                print("User stopped")
                break
            elif k == ord('s'):
                cv.imwrite('opticalfb.png', frame)
                # cv.imwrite('opticalhsv.png', viz_image)

            previous_bw = current_bw

        self.metrics.clean_up()

        cap.release()
        cv.destroyAllWindows()

    def play_video(self):
        cap = cv.VideoCapture(self.video_path)

        while(cap.isOpened()):
            _, frame = cap.read()
            
            if frame is None:
                print("end of video")
                break

            cv.imshow('frame', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        cv.destroyAllWindows()
        cap.release()

    def dense_optical_flow(self, previous_frame, current_frame):
        # TODO: Make static/refactor out?
        #                                                                       Literally magic numbers taken from tutorial
        return cv.calcOpticalFlowFarneback(previous_frame, current_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    def lk_optical_flow(self):
        """
        Calculates Lucas-Kanade optical flow on video. Code taken from
            https://docs.opencv.org/3.4/d7/d8b/tutorial_py_lucas_kanade.html
        NOTE: This doesn't do any of the fancy metric stuff.
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
        _, old_frame = cap.read()
        old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
        p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)

        while(1):
            _, frame = cap.read()

            if frame is None:
                print("end of video")
                break

            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # calculate optical flow
            p1, st, _ = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

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
    import sys

    rhy_det = RhythmDetector(video_name='AutoPortrait.mov')
    # rhy_det = RhythmDetector(video_name='whiplash.mov')
    # rhy_det = RhythmDetector(video_name='BasicMotion.mov')
    
    assert len(sys.argv) <= 2, Exception('Too many arguments.')

    if len(sys.argv) == 1:
        # Default for debugging purposes.
        video_name = 'AutoPortrait.mov'
        # video_name = 'whiplash.mov'
        # video_name = 'BasicMotion.mov'
    else:
        video_name = sys.argv[1]
    
    video_path = os.path.join('video/', video_name)

    if video_name == 'your_video_name.mov':
        assert os.path.isfile(video_path), Exception('"{}" not found. Did you replace "your_video_name.mov" with the file name of your video?')
    else:
        assert os.path.isfile(video_path), Exception('"{}" not found.')


    start = time.time()
    rhy_det = RhythmDetector(video_name=video_name)

    rhy_det.analyze()
    print("Took {}seconds".format(time.time() - start))
