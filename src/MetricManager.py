import math
from enum import IntEnum
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox


class Metric(IntEnum):
    """
    Metrics used by MetricManager.

    Note:
        - DFLOW stands for flow difference, or the numeric derivative of the flow. 
        - There is space between categories in case more metrics come up.
    """

    # Histograms
    FIRST_HISTOGRAM = 0             # Used to check if enum is histogram
    FLOW_MAG_DISTRIBUTION = 1       # Histogram distribution of flow vector magnitude in an image
    FLOW_ANG_DISTRIBUTION = 2       # Same but angle
    DFLOW_MAG_DISTRIBUTION = 3      # This time with derivative of flow
    DFLOW_ANG_DISTRIBUTION = 4

    LAST_HISTOGRAM = 9               # Used to check if enum is histogram
    
    # Vectors
    FIRST_VECTOR = 10
    NET_FLOW = 11
    
    LAST_VECTOR = 19
    
    # Numbers
    FIRST_NUMBER = 20
    MEAN_FLOW_MAGNITUDE = 21
    FLOW_PERCENTILES = 22
    NET_FLOW_MAG = 23
    NET_FLOW_ANG = 24
    
    LAST_NUMBER = 29

    # TODO: Make metric method s.t. you do metric.is_histogram()
    def is_histogram(self):
        """
        Ouput (bool):
            Whether given metric is a histogram
        """
        return Metric.FIRST_HISTOGRAM < self < Metric.LAST_HISTOGRAM

    def is_vector(self):
        """
        Ouput (bool):
            Whether given metric is a vector
        """
        return Metric.FIRST_VECTOR < self < Metric.LAST_VECTOR

    def is_number(self):
        """
        Ouput (bool):
            Whether given metric is a number
        """
        return Metric.FIRST_NUMBER < self < Metric.LAST_NUMBER


class MetricManager(object):
    """
    Metric Manager takes flow from CardistryRhythm, and computes various
    metrics about that flow. It also handles vizualization of those metrics.
    """

    string_to_metric = {
        # Histograms
        'flow_mag_distribution' : Metric.FLOW_MAG_DISTRIBUTION,
        'flow_ang_distribution' : Metric.FLOW_ANG_DISTRIBUTION,
        'flow_diff_mag_distribution' : Metric.DFLOW_MAG_DISTRIBUTION,
        'flow_diff_ang_distribution' : Metric.DFLOW_ANG_DISTRIBUTION,

        # Vectors
        'net_flow' : Metric.NET_FLOW,

        # Numbers
        'mean_flow_magnitude' : Metric.MEAN_FLOW_MAGNITUDE,
        'flow_percentiles' : Metric.FLOW_PERCENTILES,
        'net_flow_mag' : Metric.NET_FLOW_MAG,
        'net_flow_ang' : Metric.NET_FLOW_ANG
    }

    metric_to_string = dict([value, key] for key, value in string_to_metric.items())

    def __init__(self, 
        desired_metrics, 
        image_viz_type='color',
        percentiles=[50, 75],
        verbose=True):
        """
        Inputs:
            desired_metrics ([[histogram], [metric_in_plot1, metric_in_plot1], [metric_in_plot2]]):
                Nested list of metrics. Inner list represents metrics in a subplot.
                A histogram takes a whole subplot.
                Note: You can only have *three* subplots.

            image_viz (string):
                Choice of 'color' or 'mask'. TODO: Implement mask.

            percentiles ([float]):
                List of percentiles for the relevant metric. Entires should be between 0, 100.

            verbose (bool):
                Print metrics as they are computed. 
        """
        num_subplots = 3

        # User inputs
        self.image_viz_type = image_viz_type
        self.percentiles = percentiles
        self.verbose = verbose

        self.video_in_width = None
        self.video_in_height = None

        # Output variables
        self.full_video_writer = None  # OpenCV video writer 
        self.subplots = None

        # Temporary variables
        self.plot_lines = []  # Used to store plot lines so that they can be cleared on redraw.
        self.image_viz = None

        self.metric_structure = []  # represent the metric plot structure
        self.metric_data = {}
        # when computed, depending on desired_metrics:
        # {
        #     Metric.MEAN_FLOW: [],
        #     Metric.FLOW_PERCENTILES: [[] for i in self.percentiles],
        #     ...
        # }

        # Check desired_metrics looks good
        assert (len(desired_metrics) < (num_subplots + 1)), "len(desired_metrics) must be {} or less.".format(num_subplots)

        for plot_metrics in desired_metrics:
            sub_plot = []
            
            # Check sub-arrays are well formed
            if len(plot_metrics) == 0:
                raise Exception("Empty array in desired_metrics.")

            elif len(plot_metrics) == 1:
                metric = MetricManager.string_to_metric.get(plot_metrics[0])
                assert (metric is not None), Exception('"{}" is not a valid metric string.'.format(plot_metrics[0]))
                sub_plot = [metric]
                self.register_metric_data(metric)

            else:
                # Initialize self.metric_structure and self.metric_data
                for metric_string in plot_metrics:
                    metric = MetricManager.string_to_metric.get(metric_string)

                    # TODO: Find better error message
                    assert (not metric.is_histogram()), "Histogram must be alone in subplot."
                    assert (metric is not None), Exception('"{}" is not a valid metric string.'.format(plot_metrics[0]))

                    sub_plot.append(metric)
                    self.register_metric_data(metric)

            self.metric_structure.append(sub_plot)

    def register_metric_data(self, metric):
        """
        Initialize metric in self.metric_data.
        """
        # Initialize self.metric_data
        if not metric.is_histogram():
            if metric is Metric.FLOW_PERCENTILES:
                self.metric_data[metric] = [[] for i in self.percentiles]
            else:
                self.metric_data[metric] = []
                
    def has_metric(self, metric):
        """
        Check whether the manager has a desired metric.

        Inputs:
            metric (Metric enum):
                metric to check.
        """
        return metric in self.metric_data

    def setup(self, source_vid_length, source_vid_width, source_vid_height, source_vid_fps, out_vid_path, out_vid_name):
        """
        Initialize video writers and plots.

        Inputs:
            source_vid_length(int):
                length of video in frames.

            source_vid_size ((int, int)):
                (width, height) of source video.
            
            source_vid_fps (float):
                Framerate of source video.

            out_vid_path (string):
                Location of output video.

            out_vid_name (string):
                Name of output video.
        """
        # Setup video writers
        full_video_path = out_vid_path + out_vid_name[:-4] +'_analysis.mov'
        print("Writing video at: '{}'".format(full_video_path))
        self.video_in_width = int(source_vid_width)
        self.video_in_height = int(source_vid_height)
        plot_height = self.video_in_width  # TODO: Refactor for safety
        self.full_video_writer = cv.VideoWriter(
            full_video_path,
            cv.VideoWriter.fourcc('m','p','4','v'),
            source_vid_fps,
            (self.video_in_width, self.video_in_height + plot_height)
        )  # filename, fourcc, fps, frameSize

        self.image_viz = np.zeros((self.video_in_height, self.video_in_width, 3), dtype='uint8')
        self.image_viz[...,1] = 255

        self.setup_plots(source_vid_length)
            
    def setup_plots(self, source_vid_length):
        plt.ion()
        self.fig, self.subplots = \
            plt.subplots(nrows=len(self.metric_structure), ncols=1, figsize=(5,5))

        # TODO: Deal with duplicate metrics elegantly
        color_counter = 0  # used for legend coloring

        for index, metric_group in enumerate(self.metric_structure):
            axes = self.subplots[index]

            if metric_group[0].is_histogram():
                metric = metric_group[0]
                axes.set_ylabel('Frequency')
                axes.set_ylim(10, 60000)

                # TODO: Maybe generalize this into a metric_to_plot_descriptors dict
                if metric is Metric.FLOW_MAG_DISTRIBUTION:
                    axes.set_xlabel('Motion Magnitude (pixels)')
                    axes.set_xlim(0, 100)
                elif metric is Metric.DFLOW_MAG_DISTRIBUTION:
                    axes.set_xlabel('Motion Derivative Magnitude (pixels/frame)')
                    axes.set_xlim(0, 100)
                elif metric is Metric.FLOW_ANG_DISTRIBUTION:
                    axes.set_xlabel('Motion Angle (radians)')
                    axes.set_xlim(0, 2*np.pi)
                elif metric is Metric.DFLOW_ANG_DISTRIBUTION:
                    axes.set_xlabel('Motion Derivative Angle (radians/frame)')
                    axes.set_xlim(0, 2*np.pi)
                else:
                    raise Exception("Histogram plot labels not found")

            else:
                # metric plot
                # TODO: Make more descriptive ylabel if there's only one metric.
                axes.set_xlabel('time (frames)')
                axes.set_ylabel('metric')
                axes.set_xlim(0, source_vid_length)  # TODO: no magic numbers
                axes.set_ylim(0, 6.5)

                for metric in metric_group:
                    if metric is Metric.FLOW_PERCENTILES:
                        for index, percent in enumerate(self.percentiles):
                            axes.plot(
                                [],
                                label='{}th percentile'.format(percent), 
                                color='C{}'.format(color_counter)  # Make all percentile lines a different color
                            )
                            color_counter += 1
                    else:
                        axes.plot([], label=MetricManager.metric_to_string[metric], color='C{}'.format(color_counter))
                        color_counter += 1

        self.fig.legend(loc='lower center', ncol=2)
        self.fig.tight_layout()
        self.fig.subplots_adjust(bottom=0.25)
        self.fig.canvas.draw()
        
    def compute_metrics(self, flow, dflow=None ):
        """
        Calculate the motion metric from the flow. 

        TODO: Replace lists with numpy arrays. Initialize to have number of
        frames length of zeros, index with current frame. 
        TODO: Try doing a second derivative like thing by taking the difference
        between the current and the previous flow. This would show those sharp
        corners that function as beats.
        """
        console_output = '\n'
        
        mag, _ = cv.cartToPolar(flow[...,0], flow[...,1])
        flat_mag = mag.flatten()

        # Average the flow vectors
        if self.has_metric(Metric.NET_FLOW) or \
                self.has_metric(Metric.NET_FLOW_MAG) or \
                self.has_metric(Metric.NET_FLOW_ANG):
            net_vector = np.sum(np.sum(flow, axis=0), axis=0) / np.size(flow)

            if self.metric_data.get(Metric.NET_FLOW) is None:
                # Might not exist if it's magnitude/angle is requested, but not the vector
                self.metric_data[Metric.NET_FLOW] = np.array([net_vector])
            else:
                self.metric_data[Metric.NET_FLOW] = np.vstack((self.metric_data[Metric.NET_FLOW], net_vector))
            
            cv_mag, cv_ang = cv.cartToPolar(*net_vector)
            self.metric_data[Metric.NET_FLOW_MAG].append(cv_mag[0]) # cartToPolar spits out ([mag, 0, 0, 0], [ang, 0, 0, 0])
            self.metric_data[Metric.NET_FLOW_ANG].append(cv_ang[0])
            # NOTE: should make this a function when other vector metrics are added. 

            console_output += "\nnet_vector: {}".format(net_vector)

        # Average flow magnitude
        if self.has_metric(Metric.MEAN_FLOW_MAGNITUDE):
            average_motion = np.mean(flat_mag)

            self.metric_data[Metric.MEAN_FLOW_MAGNITUDE].append(average_motion)
            console_output += "\naverage motion: {:.4f}".format(average_motion)

        # Different percentiles of flow magnitude frequency
        if self.has_metric(Metric.FLOW_PERCENTILES):
            percentile_strings = []
            for index, percent in enumerate(self.percentiles):
                percentile_motion = np.percentile(flat_mag, percent)
                self.metric_data[Metric.FLOW_PERCENTILES][index].append(percentile_motion)
                percentile_strings.append("{}th = {:.4f}".format(
                    percent, 
                    self.metric_data[Metric.FLOW_PERCENTILES][index][-1]))

            console_output += "\npercentiles: " + ', '.join(percentile_strings)
            # eg. console_output = "\npercentiles: 50th = 0.1891, 75th = 0.4381"

        # TODO: Fit line to histogram
        # A, B = np.polyfit(x, numpy.log(y), 1, w=numpy.sqrt(y))

        if self.verbose:
            # Build the string to print once so the console doesn't go nuts
            print(console_output)

    def clear_plots(self):
        """
        Remove data from plots for redraw.
        """
        for line in self.plot_lines:
            line.remove()
        self.plot_lines.clear()

    def plot_metrics(self, flow, dflow=None):
        """
        Plot metrics.
        NOTE: For the colors to work out, the for-loop structure must be the same as
        that of setup_plots (so that color_counter increments in the same way)
        TODO: Maybe find a way to make coloring more consistent?
        """
        
        # TODO: Deal with duplicate metrics elegantly
        color_counter = 0  # used for legend coloring

        self.clear_plots()

        for index, metric_group in enumerate(self.metric_structure):
            axes = self.subplots[index]

            if metric_group[0].is_histogram():
                assert (flow is not None), Exception('Undefined flow for plotting.')
                metric = metric_group[0]
                if metric is Metric.FLOW_MAG_DISTRIBUTION:
                    mag, _ = cv.cartToPolar(flow[...,0], flow[...,1])
                    _, _, patches = axes.hist(x=mag.flatten(), bins=30, log=True, color='C0')
                    self.plot_lines += patches
                    
                elif metric is Metric.FLOW_ANG_DISTRIBUTION:
                    _, ang = cv.cartToPolar(flow[...,0], flow[...,1])
                    _, _, patches = axes.hist(x=ang.flatten(), bins=30, color='C0')
                    self.plot_lines += patches

                elif metric is Metric.DFLOW_MAG_DISTRIBUTION:
                    mag, _ = cv.cartToPolar(dflow[...,0], dflow[...,1])
                    _, _, patches = axes.hist(x=mag.flatten(), bins=30, log=True, color='C0')
                    self.plot_lines += patches
                    
                elif metric is Metric.DFLOW_ANG_DISTRIBUTION:
                    _, ang = cv.cartToPolar(dflow[...,0], dflow[...,1])
                    _, _, patches = axes.hist(x=ang.flatten(), bins=30, color='C0')
                    self.plot_lines += patches
                    
                else:
                    raise Exception("Plotting method for histogram not found.")

            else:
                # metric plot
                for metric in metric_group:
                    if metric is Metric.FLOW_PERCENTILES:
                        for index, percent in enumerate(self.percentiles):
                            self.plot_lines += axes.plot(
                                self.metric_data[Metric.FLOW_PERCENTILES][index],
                                label='{}th percentile'.format(percent), 
                                color='C{}'.format(color_counter)  # Make all percentile lines a different color
                            )
                            color_counter += 1
                    else:
                        self.plot_lines += axes.plot(
                            self.metric_data[metric],
                            label=MetricManager.metric_to_string[metric],
                            color='C{}'.format(color_counter)
                        )
                        color_counter += 1
        
        viz_output = self.make_image_viz(flow)
        self.write_full_frame(viz_output, self.fig)

    def make_image_viz(self, flow):
        """
        Generate and display flow visualization.

        Inputs:
            flow (np.array):
                optical flow array.
        """
        if self.image_viz_type is None:
            pass
        elif self.image_viz_type == 'color':
            #Flow magnitude is hsv value, angle is hue.
            mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
            self.image_viz[...,0] = ang*180/np.pi/2
            self.image_viz[...,2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
            viz_output = cv.cvtColor(self.image_viz, cv.COLOR_HSV2BGR)

            # TODO: Refactor when more vectors are added and there's a reason to refactor
            if Metric.NET_FLOW in self.metric_data:
                net_vector = self.metric_data[Metric.NET_FLOW][-1]
                net_vector_mag, net_vector_ang = cv.cartToPolar(float(net_vector[0]), float(net_vector[1]))

                # Draw magnitude and angle vizualizer
                diagram_center = (70,70)
                arrow_length = 50
                disp_angle = net_vector_ang[0]  # Why does cartToPolar spit out ([mag, 0, 0, 0], [ang, 0, 0, 0])? Beats me.
                arrow_head = (int(diagram_center[0] + arrow_length * math.cos(disp_angle)), int(diagram_center[1] + arrow_length * math.sin(disp_angle)))
                
                cv.circle(img=viz_output, center=diagram_center, radius=int(100*net_vector_mag[0]),color=(0, 0, 255),thickness=-1)            
                cv.arrowedLine(viz_output, diagram_center, arrow_head, thickness=2, color=(255,255,255))

        elif self.image_viz_type == 'mask':
            raise Exception("Mask image viz not implemented")  # TODO: Implement
        else:
            raise Exception('"{}" is not a valid image vizualization.'.format(self.image_viz_type))

        cv.imshow('flow visualization', viz_output)
        return viz_output

    def write_full_frame(self, video_viz, fig):
        """
        Write viz frame to video, with plots.
        """
        # Convert plot image to np array to save with opencv TODO: wtf is this seriously the best way to do this  
        plot_image_array = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        plot_image_array = plot_image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Resize the raster image of the plot *gag* TODO: Do this in a gag-less way. 
        plot_image_array = cv.resize(plot_image_array, (self.video_in_width, self.video_in_width))
        plot_image_array = cv.cvtColor(plot_image_array, cv.COLOR_RGB2BGR)
        
        full_frame = np.concatenate((video_viz, plot_image_array), axis=0)
        self.full_video_writer.write(full_frame)

    def clean_up(self):
        self.full_video_writer.release()

    # NOTE: Not refactoring until needed.
    # def save_plots(self):
    #     """
    #     Save metric graphs in result/ folder. Includes average, 25th, 50th
    #     and 75th percentiles.

    #     Note: Deprecated for now because plots are saved with video. Code still
    #     here just in case.
    #     """
    #     extent = self.ax_metrics.get_tightbbox(self.fig.canvas.renderer).transformed(self.fig.dpi_scale_trans.inverted())
    #     self.fig.savefig('result/' + self.video_name[:-4] + '_metrics.png', bbox_inches=extent)
    #     # TODO: Split on '.' to ditch extention

    def compute_metric_summary(self):
        # TODO: Maybe compute some meta metrics like average net_vector angle.
        pass

if __name__ == "__main__":
    desired_metrics = [ 
        ['flow_mag_distribution'],
        ['flow_ang_distribution'],
        ['mean_flow_magnitude', 'flow_percentiles', 'net_flow_mag', 'net_flow_ang']
    ]

    expected_metric_structure = [
        [Metric.FLOW_MAG_DISTRIBUTION],
        [Metric.FLOW_ANG_DISTRIBUTION],
        [Metric.MEAN_FLOW_MAGNITUDE, Metric.FLOW_PERCENTILES, Metric.NET_FLOW_MAG, Metric.NET_FLOW_ANG]
    ]
    
    expected_metric_data = {
        Metric.MEAN_FLOW_MAGNITUDE: np.array([]), 
        Metric.FLOW_PERCENTILES:    [[], []], 
        Metric.NET_FLOW_MAG:        np.array([]),
        Metric.NET_FLOW_ANG:        np.array([])
    }

    print("Testing MetricManager")
    metrics = MetricManager(desired_metrics = desired_metrics)

    # TODO: Fix numpy array being obnoxious about equality
    # print("\nChecking data is well formed...")
    # assert (metrics.metric_data == expected_metric_data), Exception("metric data malformed")
    # print("All good")

    print("\nChecking plot structure is well formed...")
    assert (metrics.metric_structure == expected_metric_structure), Exception("metric structure malformed")
    print("All good")