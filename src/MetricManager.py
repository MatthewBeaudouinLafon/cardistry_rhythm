from enum import Enum

class Metric(Enum):
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
    
    LAST_NUMBER = 29

    def is_histogram(self, metric):
        """
        Inputs:
            metric (Metric enum)

        Ouput (bool):
            Whether given metric is a histogram
        """
        return Metric.FIRST_HISTOGRAM < metric < Metric.LAST_HISTOGRAM

    def is_vector(self, metric):
        """
        Inputs:
            metric (Metric enum)

        Ouput (bool):
            Whether given metric is a vector
        """
        return Metric.FIRST_VECTOR < metric < Metric.LAST_VECTOR

    def is_number(self, metric):
        """
        Inputs:
            metric (Metric enum)

        Ouput (bool):
            Whether given metric is a number
        """
        return Metric.FIRST_NUMBER < metric < Metric.LAST_NUMBER

    

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
        'flow_percentiles' : Metric.FLOW_PERCENTILES
    }

    def __init__(self, desired_metrics, 
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

        # Check desired_metrics looks good
        assert (len(desired_metrics) > num_subplots), "len(desired_metrics) must be {} or less.".format(num_subplots)

        self.desired_metrics = []
        for plot_metrics in desired_metrics:
            sub_plot = []
            
            if len(plot_metrics) == 0:
                raise Exception("Empty array in desired_metrics.")

            elif len(plot_metrics) == 1:
                metric = string_to_metric.get(plot_metrics[0])
                if metric is None:
                    raise Exception('"{}" is not a valid metric string.'.format(plot_metrics[0]))
                
                # TODO: figure out how to do the plot stuff
            else:
                for metric_string in plot_metrics:
                    # TODO: Find better error
                    assert (not Metric.is_histogram(metric_string)), "Histogram must be alone in subplot."

                    metric = string_to_metric.get(metric_string)
                    if metric is None:
                        raise Exception('"{}" is not a valid metric string.'.format(plot_metrics[0]))

                    # TODO: figure out how to do the plot stuff

            self.desired_metrics.append(sub_plot)

        self.image_viz = image_viz
        self.percentiles = percentiles
        self.verbose = verbose