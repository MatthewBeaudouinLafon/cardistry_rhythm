import math
import time
from enum import Enum
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

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
        self.video_path = video_folder_path + video_name  # TODO: use proper python path merging
        self.percentiles = [50, 75]

        self.total_frames = None
        self.current_frame = 0

        plt.ion()
        # TODO: Make figure size the same aspect ratio as the video?
        self.fig, (self.ax_mag_distribution, self.ax_ang_distribution, self.ax_metrics) = \
            plt.subplots(nrows=3, ncols=1, figsize=(5,5))
        self.plot_lines = []  # Used to store plot lines so that they can be cleared on redraw. 
        self.metrics = {}
        # when computed, depending on what's passed:
        # {
        #     'magnitude_average': [],
        #     'magnitude_percentiles': [[] for i in self.percentiles],
        #     ...
        # }

    def has_computed_metric(self, metric_name):
        """
        Returns whether a metric has been computed.
        TODO: Use for input safety's sake.
        """
        return self.metrics.get(metric_name, None) is not None

    def analyze(self,
        method='dense_optical_flow',
        chosen_metrics=['all'],
        show_viz=True,  # TODO: Rethink what these options should be.
        save_all=True,
        verbose=True):
        """
        Analyze video using optical flow. 

        Inputs:
            method(string): 
                one of 'dense_optical_flow'
            
            chosen_metrics ([string]):
                array of strings featuring 'magnitude_percentiles', 'magnitude_average'  # TODO: Update after plot class refactor
            
            show_viz (bool):
                Show visualizations while running
            
            verbose (bool):
                Print metrics.
            
            save_all (bool):
                Save metrics, plots, and viz if enabled. 
        """
        # Choose method
        if method == 'dense_optical_flow':
            print("Analyzing using Optical Flow")
            calculate_flow = self.dense_optical_flow
        else:
            print('Method "{}" unknown.'.format(method))
            return

        # Setup Open CV
        cap = cv.VideoCapture(self.video_path)
        _, frame = cap.read()
        previous_bw = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame)
        hsv[...,1] = 255
        self.total_frame_number = cap.get(cv.CAP_PROP_FRAME_COUNT)

        # Setup video writers
        if save_all:
            vid_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            vid_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            plot_height = vid_width  # TODO: Refactor for safety
            full_video_out = cv.VideoWriter(
                'result/' + self.video_name[:-4] +'_analysis.mov',  # TODO: Better path join
                cv.VideoWriter.fourcc('m','p','4','v'),
                cap.get(cv.CAP_PROP_FPS),
                (vid_width, vid_height + plot_height)
            )  # filename, fourcc, fps, frameSize

        # Setup plots
        if len(chosen_metrics) > 0: # TODO: Good?
            self.setup_plots()

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
            self.compute_metric(flow, chosen_metrics, verbose=verbose)

            if show_viz:
                # Plot metrics
                self.plot_metrics(chosen_metrics, flow=flow)

                # Make color viz in another window. Flow magnitude is hsv value, angle is hue.
                mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
                hsv[...,0] = ang*180/np.pi/2
                hsv[...,2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
                viz_image = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)

                if 'net_vector' in chosen_metrics or 'all' in chosen_metrics:
                    net_vector = self.metrics['net_vector'][-1]
                    net_vector_mag, net_vector_ang = cv.cartToPolar(float(net_vector[0]), float(net_vector[1]))

                    # Draw magnitude and angle vizualizer
                    diagram_center = (70,70)
                    arrow_length = 50
                    disp_angle = net_vector_ang[0]  # Why does cartToPolar spit out [value, 0, 0, 0]? Beats me.
                    arrow_head = (int(diagram_center[0] + arrow_length * math.cos(disp_angle)), int(diagram_center[1] + arrow_length * math.sin(disp_angle)))
                    cv.circle(img=viz_image, center=diagram_center, radius=int(100*net_vector_mag[0]),color=(0, 0, 255),thickness=-1)            
                    cv.arrowedLine(viz_image, diagram_center, arrow_head, thickness=2, color=(255,255,255))

                # Show frame in one window
                cv.imshow('frame', frame)
                cv.imshow('flow', viz_image)


                if save_all:
                    # Convert plot image to np array to save with opencv TODO: wtf is this seriously the best way to do this  
                    plot_image_array = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                    plot_image_array = plot_image_array.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

                    # Resize the raster image of the plot *gag* TODO: Do this in a gag-less way. 
                    plot_image_array = cv.resize(plot_image_array, (vid_width, vid_width))
                    plot_image_array = cv.cvtColor(plot_image_array, cv.COLOR_RGB2BGR)
                    
                    full_video_out.write(np.concatenate((viz_image, plot_image_array), axis=0))

            # Interpret keyboard
            k = cv.waitKey(30) & 0xff
            if k == 27:
                print("User stopped")
                break
            elif k == ord('s'):
                cv.imwrite('opticalfb.png', frame)
                cv.imwrite('opticalhsv.png', viz_image)

            previous_bw = current_bw

        if save_all:
            self.save_final_plots()  # TODO: Check if plots exist?

        cap.release()
        cv.destroyAllWindows()

    def compute_metric(self, flow, chosen_metrics, verbose=True):
        """
        Calculate the motion metric from the flow. 

        TODO: Replace lists with numpy arrays. Initialize to have number of
        frames length of zeros, index with current frame. 
        TODO: Try doing a second derivative like thing by taking the difference
        between the current and the previous flow. This would show those sharp
        corners that function as beats.
        """
        all_metrics_chosen = 'all' in chosen_metrics
        console_output = '\n'
        
        mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
        flat_mag = mag.flatten()

        if all_metrics_chosen or 'net_vector' in chosen_metrics:
            # Average the flow vectors
            net_vector = np.sum(np.sum(flow, axis=0), axis=0) / np.size(flow)

            if self.metrics.get('net_vector') is None:
                self.metrics['net_vector'] = np.array([net_vector])
            else:
                self.metrics['net_vector'] = np.vstack((self.metrics['net_vector'], net_vector))
            
            console_output += "\nnet_vector: {}".format(net_vector)

        # Average flow magnitude
        if all_metrics_chosen or 'magnitude_average' in chosen_metrics:
            average_motion = np.mean(flat_mag)
            self.metrics['magnitude_average'] = self.metrics.get('magnitude_average', []) + [average_motion]
            console_output += "\naverage motion: {:.4f}".format(average_motion)

        # Different percentiles of flow magnitude frequency
        if all_metrics_chosen or 'magnitude_percentiles' in chosen_metrics:
            if self.metrics.get('magnitude_percentiles') is None:
                self.metrics['magnitude_percentiles'] = [[] for i in self.percentiles]

            percentile_strings = []
            for index, percent in enumerate(self.percentiles):
                percentile_motion = np.percentile(flat_mag, percent)
                self.metrics['magnitude_percentiles'][index].append(percentile_motion)
                percentile_strings.append("{}th = {:.4f}".format(
                    percent, 
                    self.metrics['magnitude_percentiles'][index][-1]))

            console_output += "\npercentiles: " + ', '.join(percentile_strings)
            # eg. console_output = "\npercentiles: 50th = 0.1891, 75th = 0.4381"

        # TODO: Fit line to histogram
        # A, B = np.polyfit(x, numpy.log(y), 1, w=numpy.sqrt(y))

        if verbose:
            # Build the string to print once so the console doesn't go nuts
            print(console_output)
            
    def plot_metrics(self, chosen_metrics, flow=None):
        """
        Plot with updated metrics.

        # TODO: Refactor to make a Metric class.
        # TODO: Plot all computed metrics by looping through keys?
        """
        all_metrics_chosen = 'all' in chosen_metrics
        
        self.clear_plots()

        # Plot flow vector frequency distribution
        if flow is not None:
            mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
            _, _, patches_mag = self.ax_mag_distribution.hist(x=mag.flatten(), bins=30, log=True, color='C0')
            _, _, patches_ang = self.ax_ang_distribution.hist(x=ang.flatten(), bins=30, color='C0')
            self.plot_lines += patches_mag
            self.plot_lines += patches_ang

        # Plot net_vector angle and magnitude
        if all_metrics_chosen or 'net_vector' in chosen_metrics:
            if self.metrics.get('net_vector') is None or len(self.metrics.get('net_vector')) == 0:
                print('Metric "net_vector" not computed')
            else:
                net_vectors = self.metrics['net_vector']
                net_vector_mag, net_vector_ang = cv.cartToPolar(net_vectors[...,0], net_vectors[...,1])

                self.plot_lines += self.ax_metrics.plot(net_vector_mag, label='net_vector magnitude', color='C0')
                self.plot_lines += self.ax_metrics.plot(net_vector_ang, label='net_vector angle', color='C1')

        # Plot average flow vector magnitude
        if all_metrics_chosen or 'magnitude_average' in chosen_metrics:
            if self.metrics.get('magnitude_average') is None or self.metrics.get('magnitude_average') == []:
                print('Metric "average" not computed')
            else:
                self.plot_lines += self.ax_metrics.plot(self.metrics['magnitude_average'], label='magnitude_average', color='C2')

        # Plot average flow vector magnitude percentiles
        if all_metrics_chosen or 'magnitude_percentiles' in chosen_metrics:
            if self.metrics.get('magnitude_percentiles') is None or self.metrics.get('magnitude_percentiles') == []:
                print('Metric "percentiles" not computed')

            for index, percent in enumerate(self.percentiles):
                self.plot_lines += self.ax_metrics.plot(
                    self.metrics['magnitude_percentiles'][index],
                    label='{}th percentile'.format(percent), 
                    color='C{}'.format(4+index)  # Make all percentile lines a different color
                )

        print("average magnitude / Net vector magnitude = {}".format(self.metrics['magnitude_average'][-1] / net_vector_mag[-1]))

        # NOTE: If you add a new plot with a legend, make sure to update the mock plots in setup_plots

        self.fig.canvas.draw()

    def clear_plots(self):
        """
        Remove data from plots for redraw.
        """
        for line in self.plot_lines:
            line.remove()
        self.plot_lines.clear()

    def setup_plots(self):
        # Display the magnitude pixel motion in a histogram
        self.ax_mag_distribution.set_xlabel('Motion Magnitude (pixels)')
        self.ax_mag_distribution.set_ylabel('Frequency')
        self.ax_mag_distribution.set_xlim(0, 100)  # TODO: no magic numbers
        self.ax_mag_distribution.set_ylim(10, 60000)
        # self.ax_mag_distribution.set_title('Distribution of pixel motion magnitude')

        # Display the magnitude pixel motion in a histogram
        self.ax_ang_distribution.set_xlabel('Motion Angle (radians)')
        self.ax_ang_distribution.set_ylabel('Frequency')
        self.ax_ang_distribution.set_xlim(0, 2*np.pi)
        self.ax_ang_distribution.set_ylim(10, 60000)  # TODO: no magic numbers (Why does 60000 work consistently?)
        # self.ax_ang_distribution.set_title('Distribution of pixel motion angle')

        # Display motion metrics throughout the video
        self.ax_metrics.set_xlabel('time (frames)')
        self.ax_metrics.set_ylabel('metric')
        self.ax_metrics.set_xlim(0, self.total_frame_number)
        self.ax_metrics.set_ylim(0, 6)
        # self.ax_metrics.set_title('Other metrics')
        
        # Mock plots to build the legend.
        # TODO: Don't like this code duplication ...
        self.ax_metrics.plot([], label='net_vector magnitude', color='C0')
        self.ax_metrics.plot([], label='net_vector angle', color='C1')
        self.ax_metrics.plot([], label='magnitude_average', color='C2')
        for index, percent in enumerate(self.percentiles):
                self.ax_metrics.plot(
                    [],
                    label='{}th percentile'.format(percent), 
                    color='C{}'.format(4+index)  # Make all percentile lines a different color
                )
        self.fig.legend(loc='lower center', ncol=2)

        self.fig.tight_layout()
        self.fig.subplots_adjust(bottom=0.25)
        self.fig.canvas.draw()
        pass

    def save_final_plots(self):
        """
        Save metric graphs in result/ folder. Includes average, 25th, 50th
        and 75th percentiles.

        Note: Deprecated for now because plots are saved with video. Code still
        here just in case.
        """
        extent = self.ax_metrics.get_tightbbox(self.fig.canvas.renderer).transformed(self.fig.dpi_scale_trans.inverted())
        self.fig.savefig('result/' + self.video_name[:-4] + '_metrics.png', bbox_inches=extent)
        # TODO: Split on '.' to ditch extention

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

    def dense_optical_flow(self, previous_frame, current_frame):
        # TODO: Make static/refactor out?
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
    # TODO: Make result/ and metrics/ folder if they don't exist.

    start = time.time()
    # rhy_det = RhythmDetector(video_name='AutoPortrait.mov')
    # rhy_det = RhythmDetector(video_name='whiplash.mov')
    rhy_det = RhythmDetector(video_name='BasicMotion.mov')

    rhy_det.analyze()
    print("Took {}seconds".format(time.time() - start))
