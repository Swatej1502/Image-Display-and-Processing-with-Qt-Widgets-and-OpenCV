import sys
import cv2
import os
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QSlider,QPushButton,QHBoxLayout,QFileDialog,QDesktopWidget


class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.show_original=True
        self.show_grayscale=False
        self.show_edge=False
        self.show_facedetection=False

        self.setWindowTitle("Image Processing Through Camera")
        self.setGeometry(100, 100, 1200, 450)
        
        # using QDesktopWidget to  get screen resolution
        desktop = QDesktopWidget()
        screen_rect = desktop.screenGeometry()
        self.screen_width, self.screen_height = screen_rect.width(), screen_rect.height()

        #creating a videolabel to display current frame
        self.video_label = QLabel()
        self.video_label.setStyleSheet("border: 1px solid black;")
        
        #created a threshold slider to adjest threshold values for  edge detection
        self.threshold_label = QLabel("Threshold for Edge Detection:")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(255)
        self.threshold_slider.setValue(100)
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider.setTickInterval(10)
        self.threshold_slider.valueChanged.connect(self.update_edges)
        self.threshold_value_label = QLabel(str(self.threshold_slider.value()))
        
        # created pause button continue button and exit button
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause_video)
        self.continue_button = QPushButton("Continue")
        self.continue_button.clicked.connect(self.continue_video)
        self.is_paused = False
        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(QApplication.instance().quit)
        
        # created buttons for image processing(grayscale,edge detection)
        self.original_button=QPushButton("Show original Frame")
        self.original_button.clicked.connect(self.original_display)
        self.grayscale_button = QPushButton("Add Grayscale")
        self.grayscale_button.clicked.connect(self.add_grayscale)
        self.edge_button = QPushButton("Add Edge Detection")
        self.edge_button.clicked.connect(self.add_edge)
        self.facedetection_button = QPushButton("face Detection")
        self.facedetection_button.clicked.connect(self.add_facedetection)
         
        #created a save button
        self.save_frame_button = QPushButton("Save Frame")
        self.save_frame_button.clicked.connect(self.save_frame)
        
        self.save_exit_buttons_layout = QHBoxLayout()
        self.save_exit_buttons_layout.addWidget(self.save_frame_button)
        self.save_exit_buttons_layout.addWidget(self.exit_button)
        
        self.button_layout= QHBoxLayout()
        self.button_layout.addWidget(self.pause_button)
        self.button_layout.addWidget(self.continue_button)
        
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
        self.layout.addLayout(self.threshold)
        self.layout.addLayout(self.button_layout)
        self.layout.addLayout(self.save_exit_buttons_layout)

        self.central_widget = QWidget()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)
      
        '''Created a times which runs updateframe for every 30 milliseconds using Qtimer'''
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.video_capture = cv2.VideoCapture(0)  # 0 for webcam
        
        '''Get the full path to the haarcascade_frontalface_default.xml file     or 
        cascade_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'haarcascade_frontalface_default.xml') (not working in my lap)'''
        cascade_file_path = "C:\Python312\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"

        # Load the Haar cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(cascade_file_path)

    def update_frame(self):
        if not self.is_paused:
            ret, frame = self.video_capture.read()
            if not ret:
                return
            frame = cv2.resize(frame, (1900,790))
            self.original_frame=frame.copy()
            
            if self.show_original:
            
                # Display original video
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

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
            
        else:
            frame = self.original_frame
            # Display last available frame when paused
            if self.show_original:
            
                # Display original video
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                # Convert the frame to RGB for displaying with PyQt
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert the frame to QImage
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                # Display the QImage in the QLabel
                pixmap = QPixmap.fromImage(qt_image)
                self.video_label.setPixmap(pixmap)

    def pause_video(self):
        self.is_paused = True

    def continue_video(self):
        self.is_paused = False
        
    def update_edges(self):
        # Update edge detection when threshold slider value changes
        self.update_frame()
        self.threshold_value_label.setText(str(self.threshold_slider.value()))
        
    def save_frame(self):
        '''function which  saves the current video frame to disk 
        grayscale frame or edge detection frame''' 
        if self.original_frame is not None:
            
            if self.show_original:
                frame = cv2.resize(self.original_frame, (self.screen_width, self.screen_height))
                file_path, _ = QFileDialog.getSaveFileName(self, "Save Frame as", "exp", "PNG (*.png)")
                if file_path:
                    cv2.imwrite(file_path, frame)

            if self.show_grayscale:
                frame = cv2.resize(self.original_frame, (self.screen_width, self.screen_height))
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                file_path, _ = QFileDialog.getSaveFileName(self, "Save Grayscale Frame as", "exp", "PNG (*.png)")
                if file_path:
                    cv2.imwrite(file_path, gray_frame)

            if self.show_edge:
                frame = cv2.resize(self.original_frame, (self.screen_width, self.screen_height))
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray_frame, self.threshold_slider.value(), self.threshold_slider.value() * 2)
                file_path, _ = QFileDialog.getSaveFileName(self, "Save Edges Frame as", "exp", "PNG (*.png)")
                if file_path:
                    cv2.imwrite(file_path, edges)
           
            if self.show_facedetection:
                frame = cv2.resize(self.original_frame, (self.screen_width, self.screen_height))
                file_path, _ = QFileDialog.getSaveFileName(self, "Save Frame as", "exp", "PNG (*.png)")
                if file_path:
                    cv2.imwrite(file_path, frame) 

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
    app = QApplication(sys.argv)  #Initializing a new instance of application
    player = VideoPlayer()        #calling the videplayer class
    player.show()                 #This line displays the video player widget on the screen. 
    sys.exit(app.exec_())
