from PyQt5.QtGui import QImage, QPixmap

from PyQt5.QtCore import pyqtSlot, QTimer, QDate, Qt
from PyQt5.QtWidgets import QLabel, QMainWindow, QPushButton, QApplication, QMessageBox
from PyQt5.uic import loadUi
import cv2
import face_recognition
import numpy as np
import datetime
import os
import sys
import csv
from PIL import ImageGrab


class UIWindow(QMainWindow):
    def __init__(self):
        super(UIWindow, self).__init__()
        loadUi("mainWindow.ui", self)

        self.Date = self.findChild(QLabel, "Date")
        self.Time = self.findChild(QLabel, "Time")
        self.Name = self.findChild(QLabel, "Name")
        self.Status = self.findChild(QLabel, "Status")
        self.TotalTime = self.findChild(QLabel, "TotalTime")
        self.Video = self.findChild(QLabel, "Video")
        self.Hours = self.findChild(QLabel, "Hours")
        self.Min = self.findChild(QLabel, "Min")

        self.Switch = self.findChild(QPushButton, "SwitchMode")
        self.ClockInButton = self.findChild(QPushButton, "ClockInButton")
        self.ClockOutButton = self.findChild(QPushButton, "ClockOutButton")

        now = QDate.currentDate()
        self.current_date = now.toString('ddd dd MMMM yyyy')

        self.Date.setText(self.current_date)
        self.Time.setText(datetime.datetime.now().strftime("%I:%M %p"))
        self.mode = "0"
        self.startVideo(self.mode)
        self.Switch.clicked.connect(self.changeMode)
        self.show()

    def changeMode(self):
        if self.mode == "0":
            self.mode = "2"
        elif self.mode == "2":
            self.mode = "0"

    def findEncodings(self, images):
        self.encodeList = []
        for img in self.images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.encode = face_recognition.face_encodings(img)[0]
            self.encodeList.append(self.encode)
        return self.encodeList

    def markAttendance(self, name):
        if self.ClockInButton.isChecked():
            self.ClockInButton.setEnabled(False)
            with open('Attendance.csv', 'a') as f:
                if (name != 'unknown'):
                    date_time_string = datetime.datetime.now().strftime("%y/%m/%d %H:%M:%S")
                    f.writelines(
                        f'\n{name},{date_time_string},Clock In')
                    self.ClockInButton.setChecked(False)

                    self.Name.setText(name)
                    self.Status.setText('Clocked In')
                    self.Hours.setText('Measuring')
                    self.Min.setText('')

                    # print('Yes clicked and detected')
                    self.Time1 = datetime.datetime.now()
                    self.ClockInButton.setEnabled(True)
        elif self.ClockOutButton.isChecked():
            self.ClockOutButton.setEnabled(False)
            with open('Attendance.csv', 'a') as f:
                date_time_string = datetime.datetime.now().strftime("%y/%m/%d %H:%M:%S")
                f.writelines(
                    f'\n{name},{date_time_string},Clock Out')
                self.ClockOutButton.setChecked(False)

                self.Name.setText(name)
                self.Status.setText('Clocked Out')
                self.Time2 = datetime.datetime.now()

                self.ElapseList(name)
                self.TimeList2.append(datetime.datetime.now())
                CheckInTime = self.TimeList1[-1]
                CheckOutTime = self.TimeList2[-1]
                self.ElapseHours = (CheckOutTime - CheckInTime)
                self.Min.setText("{:.0f}".format(
                    abs(self.ElapseHours.total_seconds() / 60) % 60) + 'm')
                self.Hours.setText("{:.0f}".format(
                    abs(self.ElapseHours.total_seconds() / 60**2)) + 'h')
                self.ClockOutButton.setEnabled(True)

    def ElapseList(self, name):
        with open('Attendance.csv', "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 2

            Time1 = datetime.datetime.now()
            Time2 = datetime.datetime.now()
            for row in csv_reader:
                for field in row:
                    if field in row:
                        if field == 'Clock In':
                            if row[0] == name:
                                Time1 = (datetime.datetime.strptime(
                                    row[1], '%y/%m/%d %H:%M:%S'))
                                self.TimeList1.append(Time1)
                        if field == 'Clock Out':
                            if row[0] == name:
                                Time2 = (datetime.datetime.strptime(
                                    row[1], '%y/%m/%d %H:%M:%S'))
                                self.TimeList2.append(Time2)

    # FOR CAPTURING SCREEN RATHER THAN WEBCAM

    def captureScreen(self, bbox=(300, 300, 690+300, 530+300)):
        self.capScr = np.array(ImageGrab.grab(bbox))
        self.capScr = cv2.cvtColor(self.capScr, cv2.COLOR_RGB2BGR)
        return self.capScr

    @pyqtSlot()
    def startVideo(self, camera_name):

        if len(camera_name) == 1:
            self.capture = cv2.VideoCapture(int(camera_name))
        else:
            self.capture = cv2.VideoCapture(camera_name)
        self.timer = QTimer(self)  # Create Timer
        path = 'imageTest'
        if not os.path.exists(path):
            os.mkdir(path)
        # known face encoding and known face name list
        images = []
        self.class_names = []
        self.encode_list = []
        self.TimeList1 = []
        self.TimeList2 = []
        attendance_list = os.listdir(path)

        # print(attendance_list)
        for cl in attendance_list:
            cur_img = cv2.imread(f'{path}/{cl}')
            images.append(cur_img)
            self.class_names.append(os.path.splitext(cl)[0])
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(img)
            encodes_cur_frame = face_recognition.face_encodings(img, boxes)[0]
            # encode = face_recognition.face_encodings(img)[0]
            self.encode_list.append(encodes_cur_frame)
        # Connect timeout to the output function
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(10)  # emit the timeout() signal at x=40ms

    def face_rec_(self, frame, encode_list_known, class_names):
        faces_cur_frame = face_recognition.face_locations(frame)
        encodes_cur_frame = face_recognition.face_encodings(
            frame, faces_cur_frame)
        # count = 0
        for encodeFace, faceLoc in zip(encodes_cur_frame, faces_cur_frame):
            match = face_recognition.compare_faces(
                encode_list_known, encodeFace, tolerance=0.50)
            face_dis = face_recognition.face_distance(
                encode_list_known, encodeFace)
            self.name = "unknown"
            best_match_index = np.argmin(face_dis)
            if match[best_match_index]:
                self.name = class_names[best_match_index].upper()
                y1, x2, y2, x1 = faceLoc
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y2 - 20), (x2, y2),
                              (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, self.name, (x1 + 6, y2 - 6),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                self.markAttendance(self.name)
                self.Name.setText(self.name)
                self.Time.setText(datetime.datetime.now().strftime("%I:%M %p"))
            elif (self.name == "unknown"):
                y1, x2, y2, x1 = faceLoc
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.rectangle(frame, (x1, y2 - 20), (x2, y2),
                              (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, self.name, (x1 + 6, y2 - 6),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                self.Name.setText("N/A")
                self.Time.setText(datetime.datetime.now().strftime("%I:%M %p"))
        return frame

    def update_frame(self):
        if self.mode == "0":
            ret, self.image = self.capture.read()
        elif self.mode == "2":
            self.image = self.captureScreen()
        self.displayImage(self.image, self.encode_list, self.class_names, 1)

    def displayImage(self, image, encode_list, class_names, window=1):

        image = cv2.resize(image, (640, 480))
        try:
            image = self.face_rec_(image, encode_list, class_names)
        except Exception as e:
            print(e)
        qformat = QImage.Format_Indexed8
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        outImage = QImage(
            image, image.shape[1], image.shape[0], image.strides[0], qformat)
        outImage = outImage.rgbSwapped()

        if window == 1:
            self.Video.setPixmap(QPixmap.fromImage(outImage))
            self.Video.setScaledContents(True)

        # cv2.imshow('Webcam', self.img)
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = UIWindow()
    sys.exit(app.exec_())
