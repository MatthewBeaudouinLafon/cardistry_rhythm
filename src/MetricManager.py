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
        Inputs:
            metric (Metric enum)

        Ouput (bool):
            Whether given metric is a histogram
        """
        return Metric.FIRST_HISTOGRAM < self < Metric.LAST_HISTOGRAM

    def is_vector(self):
        """
        Inputs:
            metric (Metric enum)

        Ouput (bool):
            Whether given metric is a vector
        """
        return Metric.FIRST_VECTOR < self < Metric.LAST_VECTOR

    def is_number(self):
        """
        Inputs:
            metric (Metric enum)

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

    metric_to_string = dict([value, key] for key, value in string_to_metric.iteritems())

    def __init__(self, 
        desired_metrics = [ 
                            ['flow_mag_distribution'],
                            ['flow_ang_distribution'],
                            ['mean_flow_magnitude', 'flow_percentiles', 'net_flow_mag', 'net_flow_ang']
                          ], 
        image_viz='color',
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

        self.image_viz = image_viz
        self.percentiles = percentiles
        self.verbose = verbose

        self.full_video_writer = None  # OpenCV video writer 
        self.subplots = None
        self.plot_lines = []  # Used to store plot lines so that they can be cleared on redraw. 

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
                self.metric_data[metric] = np.array([])
                
    def has_metric(self, metric):
        """
        Check whether the manager has a desired metric.

        Inputs:
            metric (Metric enum):
                metric to check.
        """
        return metric in self.metric_data

    def setup(self, source_vid_length, source_vid_size, source_vid_fps, out_vid_path, out_vid_name):
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
        vid_width = int(source_vid_size[0])
        vid_height = int(source_vid_size[1])
        plot_height = vid_width  # TODO: Refactor for safety
        self.full_video_writer = cv.VideoWriter(
            out_vid_path,
            cv.VideoWriter.fourcc('m','p','4','v'),
            source_vid_fps,
            (vid_width, vid_height + plot_height)
        )  # filename, fourcc, fps, frameSize

        self.setup_plots(source_vid_length)
        
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
        
        mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
        flat_mag = mag.flatten()

        # Average the flow vectors
        if self.has_metric(Metric.NET_FLOW):
            net_vector = np.sum(np.sum(flow, axis=0), axis=0) / np.size(flow)

            if self.metric_data[Metric.NET_FLOW]:
                self.metric_data[Metric.NET_FLOW] = np.array([net_vector])
            else:
                self.metric_data[Metric.NET_FLOW] = np.vstack((self.metric_data[Metric.NET_FLOW], net_vector))
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
                axes.set_ylim(10, 60000)

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

    def clear_plots(self):
        pass

    def plot_metrics(self):
        pass

    def save_plots(self):
        # ???
        pass

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
        Metric.MEAN_FLOW_MAGNITUDE: [], 
        Metric.FLOW_PERCENTILES:    [[], []], 
        Metric.NET_FLOW_MAG:        [],
        Metric.NET_FLOW_ANG:        []
    }

    print("Testing MetricManager")
    metrics = MetricManager(desired_metrics = desired_metrics)

    print("\nChecking data is well formed...")
    assert (metrics.metric_data == expected_metric_data), Exception("metric data malformed")
    print("All good")

    print("\nChecking plot structure is well formed...")
    assert (metrics.metric_structure == expected_metric_structure), Exception("metric structure malformed")
    print("All good")