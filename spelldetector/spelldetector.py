'''
Spell Detector
====================

Detect spells from an IR-illuminated Harry Potter wand.

Usage
-----
spelldetector.py [<video_source>]


Keys
----
Q - quit
'''

from threading import Timer

import cv2 as cv
import numpy as np
import sys

class SpellDetector:
    def __init__(self, video_src):
        self.video_src = video_src
        self.frame_index = 0
        self.prev_frame = None

        # Background subtraction properties
        self.background_subtractor = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)

        # Wand detection & tracking properties
        self.update_interval = 5
        self.feature_params = dict(
            maxCorners = 5,
            qualityLevel = 0.01,
            minDistance = 30)
        self.lk_params = dict(
            winSize  = (25, 25),
            maxLevel = 7,
            criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        self.wand_tracks = []

    def run(self, isRemoveBackgroundEnabled=False):
        """
        Run Spell Detector!
        """

        cap = cv.VideoCapture(self.video_src)
        
        while(True):
            ret, frame = cap.read()
            if frame is None:
                break

            frame = cv.flip(frame, 1)
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            if isRemoveBackgroundEnabled:
                frame_gray = self.remove_background(frame_gray, isPlotEnabled=True)

            # Update points every x frames
            if self.frame_index % self.update_interval == 0:
                self.points = self.find_points(frame_gray, isPlotEnabled=True)

            self.track_wand(frame_gray, isPlotEnabled=True)

            self.frame_index += 1
            self.prev_frame = frame_gray

            # Display original video stream
            cv.imshow('Video Stream (RGB)', frame)

            # Press 'Q' to quit
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv.destroyAllWindows()

    def remove_background(self, frame_gray, threshold_value=200, isPlotEnabled=False):
        """
        Use background subtraction to improve wand tip detection.
        """

        mask = self.background_subtractor.apply(frame_gray, learningRate=0.001)
        frame_no_bg = cv.bitwise_and(frame_gray, frame_gray, mask=mask)
        _, frame_thresholded = cv.threshold(frame_gray, threshold_value, 255, cv.THRESH_BINARY)

        if isPlotEnabled:
            cv.imshow('Background Removed (B/W)', frame_thresholded)

        return frame_thresholded

    def find_points(self, frame, isPlotEnabled=False):
        """
        Find points to track... like the wand tip.
        """

        points = cv.goodFeaturesToTrack(frame, mask=None, **self.feature_params)

        if isPlotEnabled:
            vis = frame.copy()

            if points is not None:
                for p in points:
                    x, y = p.ravel()
                    cv.circle(vis, (x, y), 5, (255, 255, 255), thickness=-1)

            cv.imshow('Points', vis)

        return points

    def track_wand(self, frame, isPlotEnabled=False):
        """
        Track the wand tip.
        """
        
        # Start tracking when there is a previous frame and there are points to track
        if self.points is None or self.prev_frame is None:
            # Clear any old wand tracks
            self.wand_tracks.clear()
            return

        # Calculate optical flow
        new_points, status, err = cv.calcOpticalFlowPyrLK(self.prev_frame, frame, self.points, None, **self.lk_params)

        # Select good points
        if new_points is not None:
            good_new = new_points[status==1]

            # Update points for optical flow calculation
            self.points = good_new.copy().reshape(-1, 1, 2)

            # Add points to tracks array
            for p in self.points:
                x, y = p.ravel()
                self.wand_tracks.append([x, y])

        if isPlotEnabled:
            vis = frame.copy()

            if len(self.wand_tracks) > 0:
                x0, y0 = self.wand_tracks[0]
                for track in self.wand_tracks:
                    x1, y1 = track
                    cv.line(vis, (x0, y0), (x1, y1), (255, 255, 255), thickness=10)

                    x0, y0 = track
            else:
                vis = np.zeros_like(frame)
            
            cv.imshow('Tracked Points', vis)


def main():
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    SpellDetector(video_src).run(isRemoveBackgroundEnabled=True)
    print('Done')


if __name__ == "__main__":
    print(__doc__)
    main()