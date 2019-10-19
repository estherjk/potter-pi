"""
Keys
----
S - Save wand pattern as an image (for data collection)
Q - Quit
"""

import cv2 as cv
import numpy as np

import os
import sys
import time

class SpellDetector:
    """
    From the video stream, track the wand tip, determine what spell has been cast, and make magic!
    """

    def __init__(self, video_src, spell_classifier, spellcaster):
        self.video_src = video_src
        self.spell_classifier = spell_classifier
        self.spellcaster = spellcaster

        # Frame properties
        self.frame_index = 0
        self.prev_frame = None

        # Background subtraction properties
        self.background_subtractor = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)

        # Detection & tracking properties
        self.update_interval = 5
        self.feature_params = dict(
            maxCorners = 5,
            qualityLevel = 0.01,
            minDistance = 30)
        self.lk_params = dict(
            winSize  = (25, 25),
            maxLevel = 7,
            criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

        # Wand properties
        self.wand_tracks = []
        self.wand_pattern = None
        self.spell = ''
        self.max_frames_since_no_new_points = 30
        self.frames_since_no_new_points = 0

    def run(self, is_remove_background_enabled=False):
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

            if is_remove_background_enabled:
                frame_gray = self.remove_background(frame_gray, threshold_value=215)

            # Update points every x frames
            if self.frame_index % self.update_interval == 0:
                self.points = self.find_points(frame_gray)

            self.track_wand(frame_gray)

            self.frame_index += 1
            self.prev_frame = frame_gray

            # Display original video stream
            cv.imshow('Video Stream (RGB)', frame)

            # Press 'Q' to quit
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            # Press 'S' to save wand pattern as an image
            elif key == ord('s'):
                if self.wand_pattern is not None:
                    self.save_wand_pattern(self.wand_pattern)

        cap.release()
        cv.destroyAllWindows()

    def remove_background(self, frame_gray, threshold_value=200):
        """
        Use background subtraction to improve wand tip detection.
        """

        mask = self.background_subtractor.apply(frame_gray, learningRate=0.001)
        frame_no_bg = cv.bitwise_and(frame_gray, frame_gray, mask=mask)
        _, frame_thresholded = cv.threshold(frame_gray, threshold_value, 255, cv.THRESH_BINARY)

        cv.imshow('Background Removed (B/W)', frame_thresholded)

        return frame_thresholded

    def find_points(self, frame):
        """
        Find points to track... like the wand tip.
        """

        points = cv.goodFeaturesToTrack(frame, mask=None, **self.feature_params)

        vis = frame.copy()

        if points is not None:
            for p in points:
                x, y = p.ravel()
                cv.circle(vis, (x, y), 5, (255, 255, 255), thickness=-1)

        cv.imshow('Points', vis)

        return points

    def track_wand(self, frame):
        """
        Track the wand tip.
        """
        
        if self.prev_frame is None:
            return

        vis = frame.copy()

        if self.points is None:
            if len(self.wand_tracks) > 0:
                self.frames_since_no_new_points += 1

                # Predict and cast spell when no new frames have been added after specified interval
                if self.frames_since_no_new_points > self.max_frames_since_no_new_points:
                    self.spell = self.predict_spell(self.wand_pattern)
                    self.spellcaster.cast_spell(self.spell)
                    
                    # Reset
                    self.wand_tracks.clear()
                    self.frames_since_no_new_points = 0
        else:
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

        if len(self.wand_tracks) > 0:
            x0, y0 = self.wand_tracks[0]
            for track in self.wand_tracks:
                x1, y1 = track
                cv.line(vis, (x0, y0), (x1, y1), (255, 255, 255), thickness=10)

                x0, y0 = track

            self.spell = ''
            self.grab_wand_pattern(vis)
        
        cv.putText(vis, 'spell: %s' % self.spell, (10,20), cv.FONT_HERSHEY_PLAIN, 1.0, (255,255,255))
        cv.imshow('Tracked Points', vis)

    def grab_wand_pattern(self, frame):
        """
        Grab the wand pattern.
        """

        contours, _ = cv.findContours(frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return

        # Assume wand pattern is the first contour...
        contour = contours[0]

        # Then, crop the pattern
        x, y, w, h = cv.boundingRect(contour)
        cropped_frame = frame[y-10:y+h+10, x-10:x+w+10]

        if np.sum(cropped_frame) > 0:
            cropped_frame = cv.resize(cropped_frame, (100, 100))
            self.wand_pattern = cropped_frame.copy()

            cv.imshow('Wand Pattern', cropped_frame)

    def save_wand_pattern(self, frame, path='./data/'):
        """
        Save the wand pattern as an image. Used for data collection & model training.
        """

        filename = "pattern_" + str(time.time()) + ".png"
        cv.imwrite(os.path.join(path, filename), frame)


    def predict_spell(self, wand_pattern):
        """
        Predict the spell!
        """

        # Reshape wand pattern frame to be (n_batch=1, height, widgth, n_channels=1)
        wand_pattern = wand_pattern[np.newaxis, :, :, np.newaxis]
        
        predictions = self.spell_classifier.model.predict(wand_pattern)
        predicted_label_index = np.argmax(predictions[0])
        predicted_spell = self.spell_classifier.classes[predicted_label_index]
        print(predicted_spell)
        
        return predicted_spell