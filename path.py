import sys
import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QSlider, QPushButton, QHBoxLayout, QFileDialog,QLineEdit
import numpy as np

class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.show_original=True
        self.show_grayscale=False
        self.show_edge=False
        self.show_facedetection=False

        self.setWindowTitle("Image Processing by Video")
        self.setGeometry(100, 100, 1200, 400)

        self.video_label = QLabel()
        self.video_label.setStyleSheet("border: 1px solid black;")
       

        self.threshold_label = QLabel("Threshold:")
        self.threshold_value_label = QLabel("100")

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(255)
        self.threshold_slider.setValue(100)
        self.threshold_slider.setTickInterval(10)
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider.valueChanged.connect(self.update_edges)

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.play_video)
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause_video)
        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(QApplication.instance().quit)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self.set_position)
          
        self.video_path_edit = QLineEdit()
        self.video_path_edit.setPlaceholderText("Enter video path")
        self.video_path_edit.returnPressed.connect(self.load_video)
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_video)
        
        self.save_button=QPushButton('Save Image')
        self.save_button.clicked.connect(self.save_frame)
        
        self.original_button=QPushButton("Show original Frame")
        self.original_button.clicked.connect(self.original_display)
        self.grayscale_button = QPushButton("Add Grayscale")
        self.grayscale_button.clicked.connect(self.add_grayscale)
        self.edge_button = QPushButton("Add Edge Detection")
        self.edge_button.clicked.connect(self.add_edge)
        self.facedetection_button = QPushButton("face Detection")
        self.facedetection_button.clicked.connect(self.add_facedetection)
        
        self.save=QHBoxLayout()
        self.save.addWidget(self.save_button)
        self.save.addWidget(self.exit_button)
        
        self.buttons_layout=QHBoxLayout()
        self.buttons_layout.addWidget(self.play_button)
        self.buttons_layout.addWidget(self.pause_button)
        
        self.path_button=QHBoxLayout()
        self.path_button.addWidget(self.video_path_edit)
        self.path_button.addWidget(self.browse_button)
        
        self.threshold=QHBoxLayout()
        self.threshold.addWidget(self.threshold_label)
        self.threshold.addWidget(self.threshold_value_label)
        self.threshold.addWidget(self.threshold_slider)
        
        self.processing_buttons=QHBoxLayout()
        self.processing_buttons.addWidget(self.original_button)
        self.processing_buttons.addWidget(self.grayscale_button)
        self.processing_buttons.addWidget(self.edge_button)
        self.processing_buttons.addWidget(self.facedetection_button)
        
        
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.video_label)
        self.layout.addLayout(self.processing_buttons)
        self.layout.addLayout(self.buttons_layout)
        self.layout.addWidget(self.slider)
        self.layout.addLayout(self.threshold)
        self.layout.addLayout(self.path_button)
        self.layout.addLayout(self.save)
        

        self.central_widget = QWidget()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.video_capture = None
        self.is_playing = False
        
        '''Get the full path to the haarcascade_frontalface_default.xml file or 
        cascade_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'haarcascade_frontalface_default.xml') (not working in my lap)'''
        cascade_file_path = "C:\Python312\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
        # Load the Haar cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(cascade_file_path)

    def play_video(self):
        if not self.video_capture:
            return
        if not self.is_playing:
            self.timer.start(30)
            self.is_playing = True

    def pause_video(self):
        if self.is_playing:
            self.is_playing = False

    def set_position(self, position):
        if self.video_capture:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, position)


    def update_frame(self):
        if self.video_capture is not None and self.is_playing:
            ret, frame = self.video_capture.read()
            if not ret:
                return
            #resizeing the frame based on my screen we can use desktop length  and width here
            frame = cv2.resize(frame, (1900,790))
            #copying the frame to show when the video is paused
            self.original_frame=frame.copy()
            
            if self.show_original:
            
                # Display original video
                # Convert the frame to RGB for displaying with PyQt
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert the frame to QImage
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                self.video_label.setPixmap(pixmap)
            
            if self.show_grayscale:
                # Apply grayscale
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Display grayscale frame
                gray_qt_image = QImage(gray_frame.data, gray_frame.shape[1], gray_frame.shape[0], gray_frame.strides[0],
                                        QImage.Format_Grayscale8)
                # Display the QImage in the QLabel
                gray_pixmap = QPixmap.fromImage(gray_qt_image)
                self.video_label.setPixmap(gray_pixmap)
             
            if self.show_edge:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Apply edge detection
                edges = cv2.Canny(gray_frame, self.threshold_slider.value(), self.threshold_slider.value() * 2)
                # Display edges frame
                edges_qt_image = QImage(edges.data, edges.shape[1], edges.shape[0], edges.strides[0],
                                        QImage.Format_Grayscale8)
                edges_pixmap = QPixmap.fromImage(edges_qt_image)
                self.video_label.setPixmap(edges_pixmap)
            
            if self.show_facedetection:
                 # Convert the frame to grayscale
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Detect faces in the frame
                faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.05, minNeighbors=4, minSize=(40, 40))
                # Draw rectangles around the detected faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                self.video_label.setPixmap(pixmap)
                
            # Update the slider position
            position = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
            duration = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.slider.setRange(0, duration)
            self.slider.setValue(position)
        else:
            self.is_playing = False
            frame = self.original_frame
            # Display last available frame when paused
            if self.show_original:
                # Display original video based on paused frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                self.video_label.setPixmap(pixmap)
                
            if self.show_grayscale:
                 # Apply grayscale to paused frame
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Display grayscale frame
                gray_qt_image = QImage(gray_frame.data, gray_frame.shape[1], gray_frame.shape[0], gray_frame.strides[0],
                                        QImage.Format_Grayscale8)
                gray_pixmap = QPixmap.fromImage(gray_qt_image)
                self.video_label.setPixmap(gray_pixmap)
             
            if self.show_edge:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Apply edge detection to paused frame
                edges = cv2.Canny(gray_frame, self.threshold_slider.value(), self.threshold_slider.value() * 2)
                # Display edges frame
                edges_qt_image = QImage(edges.data, edges.shape[1], edges.shape[0], edges.strides[0],
                                        QImage.Format_Grayscale8)
                edges_pixmap = QPixmap.fromImage(edges_qt_image)
                self.video_label.setPixmap(edges_pixmap)
                
            if self.show_facedetection:
                # Convert the frame to grayscale
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Detect faces in the frame
                faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.05, minNeighbors=4, minSize=(40, 40))
                # Draw rectangles around the detected faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                # Convert the frame to RGB for displaying with PyQt
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert the frame to QImage
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                # Display the QImage in the QLabel
                pixmap = QPixmap.fromImage(qt_image)
                self.video_label.setPixmap(pixmap)


    def load_video(self, video_path):
        self.video_capture = cv2.VideoCapture(video_path)
        if not self.video_capture.isOpened():
            self.video_capture = None
            return                             #if not opened returning empty
        #setting slider initial value and range based on loaded video
        self.slider.setValue(0)
        self.slider.setRange(0, int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.play_video()


    def update_edges(self):
        # Update edge detection when threshold slider value changes
        self.update_frame()
        self.threshold_value_label.setText(str(self.threshold_slider.value()))
    
    
    def browse_video(self):
        #browsing the video in our system using QFiledialog
        video_path, _ = QFileDialog.getOpenFileName(self, "Open Video File as", "image1", "Video files (*.mp4 *.avi *.mkv)")
        if video_path:
            self.video_path_edit.setText(video_path)
            self.load_video(video_path)


    def save_frame(self):
        #saving paused frames or running frames
        if self.video_capture is not None:
            if self.original_frame is not None:
                frame=self.original_frame
                if self.show_original or self.show_facedetection:
                    file_path, _ = QFileDialog.getSaveFileName(self, "Save Normal Frame as", "image2", "Images (*.png *.jpg)")
                    if file_path:
                        cv2.imwrite(file_path, frame)

                if self.show_grayscale:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    file_path, _ = QFileDialog.getSaveFileName(self, "Save Grayscale Frame as", "image3", "Images (*.png *.jpg)")
                    if file_path:
                        cv2.imwrite(file_path, gray_frame)

                if self.show_edge:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray_frame, 100, 200)
                    file_path, _ = QFileDialog.getSaveFileName(self, "Save Edge Frame", "", "Images (*.png *.jpg)")
                    if file_path:
                        cv2.imwrite(file_path, edges)
  
    '''when the grayscale button is clicked it changes the values such that
       video_label shows grayscale version of current paused or running frame'''
    
    def add_grayscale(self):
        self.show_original=False
        self.show_edge=False
        self.show_grayscale=True
        self.show_facedetection=False
      
    def original_display(self):
        self.show_original=True
        self.show_edge=False
        self.show_grayscale=False
        self.show_facedetection=False
    
    def add_edge(self):
        self.show_original=False
        self.show_edge=True
        self.show_grayscale=False
        self.show_facedetection=False
        
    def add_facedetection(self):
        self.show_original=False
        self.show_edge=False
        self.show_grayscale=False
        self.show_facedetection=True

    
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    # Load video file
    video_path, _ = QFileDialog.getOpenFileName(player, "Open Video File", "", "Video files (*.mp4 *.avi *.mkv)")
    if video_path:
        player.load_video(video_path)

    sys.exit(app.exec_())
