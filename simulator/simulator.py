import sys
import time
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QPushButton,
    QMessageBox,
    QFileDialog,
    QProgressBar,
    QWidget,
    QHBoxLayout,
)
import os
from PyQt5.QtCore import QThread, pyqtSignal, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from model_simulator import detect_suspicious_activity

class DetectionThread(QThread):
    progress_updated = pyqtSignal(int)

    def __init__(self, video_path, output_video_path):
        super().__init__()
        self.video_path = video_path
        self.output_video_path = output_video_path

    def run(self):
        detect_suspicious_activity(self.video_path, self.output_video_path, self.update_progress)

    def update_progress(self, value):
        self.progress_updated.emit(value)

class SuspiciousActivityDetector(QMainWindow):
    selected_video_path = ""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Suspicious Activity Detector")
        self.initUI()

        self.mediaPlayer.durationChanged.connect(self.duration_changed)
        self.mediaPlayer.positionChanged.connect(self.position_changed)

        # Flag to prevent recursive positionChanged signals when seeking
        self.seeking = False

    def initUI(self):
        self.resize(1240, 600)
        layout = QVBoxLayout()

        self.video_widget = QVideoWidget()
        layout.addWidget(self.video_widget)

        self.mediaPlayer = QMediaPlayer()
        self.mediaPlayer.setVideoOutput(self.video_widget)

        controls_layout = QHBoxLayout()

        self.play_pause_button = QPushButton("⏸️")
        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        controls_layout.addWidget(self.play_pause_button)

        layout.addLayout(controls_layout)

        self.video_label = QLabel("")
        self.video_button = QPushButton("Select Video")
        self.video_button.clicked.connect(self.select_video)
        layout.addWidget(self.video_button)

        self.detect_button = QPushButton("Detect Suspicious Activity")
        self.detect_button.clicked.connect(self.run_detection)
        layout.addWidget(self.detect_button)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Enable drag and drop
        self.setAcceptDrops(True)

        # Status bar
        self.status_bar = self.statusBar()

    def select_video(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi)"
        )
        if filename:
            self.selected_video_path = filename
            self.status_bar.showMessage(
                "Selected Video: {}".format(self.selected_video_path)
            )
            self.video_label.setText("Video File Path : {}".format(filename))

            # Load and play the video
            media = QMediaContent(QUrl.fromLocalFile(self.selected_video_path))
            self.mediaPlayer.setMedia(media)
            self.mediaPlayer.play()

    def run_detection(self):
        video_path = self.selected_video_path
        if not video_path:
            self.show_message("Error", "Please select a video file.")
            return

        output_video_path = video_path.replace(".mp4", "_output.mp4")
        output_video_file_path = os.path.abspath("../test_videos/improved.mp4")

        self.status_bar.showMessage("Processing Video: {}".format(video_path))

        self.thread = DetectionThread(video_path, output_video_file_path)
        self.thread.progress_updated.connect(self.update_progress_bar)
        self.thread.finished.connect(lambda: self.play_output_video(output_video_file_path))
        self.thread.start()

    def play_output_video(self, output_video_path):
        time.sleep(1)  # Adding a delay to ensure the file is ready to play
        # Load and play the output video
        media = QMediaContent(QUrl.fromLocalFile(output_video_path))
        self.mediaPlayer.setMedia(media)
        self.mediaPlayer.play()

    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)

    def toggle_play_pause(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
            self.play_pause_button.setText("▶️")
        else:
            self.mediaPlayer.play()
            self.play_pause_button.setText("⏸️")

    def duration_changed(self, duration):
        if duration > 0:
            self.video_duration = duration

    def position_changed(self, position):
        if not self.seeking and hasattr(self, "video_duration") and position >= self.video_duration - 1000:
            # If the position is within the last second of the video, seek to the beginning
            self.seeking = True
            self.mediaPlayer.setPosition(0)
            self.seeking = False

    def show_message(self, title, message):
        QMessageBox.information(self, title, message)

    # Implement dragEnterEvent to accept dragged files
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    # Implement dropEvent to handle dropped files
    def dropEvent(self, event):
        for url in event.mimeData().urls():
            filename = url.toLocalFile()
            if filename.endswith((".mp4", ".avi")):
                self.selected_video_path = filename
                self.status_bar.showMessage(
                    "Selected Video: {}".format(self.selected_video_path)
                )
                # Load and play the video
                media = QMediaContent(QUrl.fromLocalFile(self.selected_video_path))
                self.mediaPlayer.setMedia(media)
                self.mediaPlayer.play()
                return
        self.show_message("Error", "Please drop a valid video file.")

def main():
    app = QApplication(sys.argv)
    window = SuspiciousActivityDetector()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    os.environ["QT_QPA_PLATFORM"] = "wayland"
    main()

