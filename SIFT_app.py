#!/usr/bin/env python3

# Above line informs the shell that it should
# start the program listed on that line and pass to it the 
# rest of the file contents as a script

# Library imports
from symbol import while_stmt
from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi 
# loadUi allows us to convert a QtDesigner .ui file to a Python object

import cv2
import sys
import numpy as np

# Defines the My_App class
# Loads the ui file we created in the previous section
class My_App(QtWidgets.QMainWindow):

    def __init__(self):
        super(My_App, self).__init__()
        loadUi("./SIFT_app.ui", self)

        self._cam_id = 0
        self._cam_fps = 10
        self._is_cam_enabled = False
        self._is_template_loaded = False

        # Interfaces with push buttons on the ui interface
        self.browse_button.clicked.connect(self.SLOT_browse_button)
        # Registers the click event from the toggle camera
        self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

        # Creates a cv2 camera object
        # Resolution is 320x240, as a low resolution will be less
        # computationally expensive
        self._camera_device = cv2.VideoCapture(self._cam_id)
        self._camera_device.set(3, 320)
        self._camera_device.set(4, 240)

        # Timer used to trigger the camera
        # and to query camera for data
        # Emits a signal every time the set interval elapses
        self._timer = QtCore.QTimer(self)
        # Connects timer signal to SLOT_query_camera() fxn
        self._timer.timeout.connect(self.SLOT_query_camera) 
        self._timer.setInterval(1000 / self._cam_fps)

        self.sift = cv2.SIFT_create()

    def SLOT_browse_button(self):
        # dlg variable instanitates a File Dialog object
        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        # Runs File Dialog object
        if dlg.exec_():
            #Returns absolute path of the selected file
            self.template_path = dlg.selectedFiles()[0]

        # Define image, then load in grayscale
        # Sift only works on grayscale, as it only works on a single channel
        img = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)  # queryimage


        # Features
        # Gets keypoints and descriptors of Queryimage
        self.kp_image, self.desc_image = self.sift.detectAndCompute(img, None)

        pixmap = QtGui.QPixmap(self.template_path)
        self.template_label.setPixmap(pixmap)

        print("Loaded template image file: " + self.template_path)

    # Source: stackoverflow.com/questions/34232632/
    def convert_cv_to_pixmap(self, cv_img):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = cv_img.shape
        bytesPerLine = channel * width
        q_img = QtGui.QImage(cv_img.data, width, height, 
                    bytesPerLine, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(q_img)

    # Captures a frame from our camera every time the timer interval elapses
    def SLOT_query_camera(self):
        ret, frame = self._camera_device.read()

        # Feature matching
        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        # Loads the flann algorythm to find the matching features
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Now detect the features and descriptions of the frame from the webcam, 
        # then compare it to the query image
        grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # trainimage
        # Gets keypoints and descriptors of key frame image
        kp_grayframe, desc_grayframe = self.sift.detectAndCompute(grayframe, None)
        matches = flann.knnMatch(self.desc_image, desc_grayframe, k=2)
        good_points = []
        
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good_points.append(m)
        
        if len(good_points) > 4:
            # Obtains matrix
            query_pts = np.float32([self.kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
            train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
            matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)

            # Perspective transform using points and matrixs
            # Makes it so we can obtain the actual image without viewpoint constraints
            h, w = frame.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)

            homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
            # cv2.imshow("Homography", homography)
            pixmap = QtGui.QPixmap( self.convert_cv_to_pixmap(homography))
        else:
            pixmap = QtGui.QPixmap(self. convert_cv_to_pixmap(frame))

        self.live_image_label.setPixmap(pixmap)

    # Turns timer on and off
    def SLOT_toggle_camera(self):
        if self._is_cam_enabled:
            self._timer.stop()
            self._is_cam_enabled = False
            self.toggle_cam_button.setText("&Enable camera")
        else:
            self._timer.start()
            self._is_cam_enabled = True
            self.toggle_cam_button.setText("&Disable camera")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myApp = My_App()
    myApp.show()
    sys.exit(app.exec_())


    