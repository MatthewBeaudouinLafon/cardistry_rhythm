# cardistry_rhythm
By Matt Beaudouin

A computational approach to understanding rhythm in Cardistry

# Usage
1. Clone this repository.

2. Make sure you have Python3 installed with opencv2 and matplotlib.

3. Put the video you want to analyze in the `video/` folder.

4. In a command line, make your way to this project's top level folder and run 
        `python3 src/main.py your_video_name.mov`

Three windows will pop up. One of with the source video, one with the 
"flow visualization", and one with metrics for the video. The real time
video is saved in the `result` folder.