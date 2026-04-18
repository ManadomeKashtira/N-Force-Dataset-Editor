import os
from PIL import Image
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QPushButton, QLabel, QHBoxLayout, 
    QProgressBar, QMessageBox, QFrame, QSlider, QTextEdit, QCheckBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

class ConversionWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(int, int)

    def __init__(self, items, target_ext, delete_originals=False, quality=95):
        super().__init__()
        self.items = items
        self.target_ext = target_ext.lower()
        self.delete_originals = delete_originals
        self.quality = quality

    def run(self):
        success_count = 0
        total = len(self.items)
        for i, item in enumerate(self.items):
            try:
                img_path = item.image_path
                base = os.path.splitext(img_path)[0]
                new_path = base + "." + self.target_ext
                
                if img_path.lower().endswith("." + self.target_ext):
                    success_count += 1
                    continue

                with Image.open(img_path) as img:
                    if self.target_ext in ["jpg", "jpeg"] and img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")
                    
                    save_args = {}
                    if self.target_ext in ["jpg", "jpeg", "webp"]:
                        save_args["quality"] = self.quality
                    
                    img.save(new_path, **save_args)
                
                if self.delete_originals and os.path.exists(new_path) and img_path != new_path:
                    os.remove(img_path)
                
                success_count += 1
            except Exception as e:
                print(f"Error converting {item.image_path}: {e}")
            self.progress.emit(int((i + 1) / total * 100))
        self.finished.emit(success_count, total)

class ConvertDialog(QDialog):
    def __init__(self, parent, items):
        super().__init__(parent)
        self.items = items
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Convert Images")
        self.setFixedWidth(350)
        self.setStyleSheet("""
            QDialog { background-color: #252526; color: #d4d4d4; border: 1px solid #333; }
            QLabel { color: #cccccc; font-size: 14px; margin-bottom: 5px; }
            QPushButton { 
                background-color: #333333; color: #cccccc; border: 1px solid #444; 
                padding: 10px; font-size: 13px; border-radius: 2px;
            }
            QPushButton:hover { background-color: #444444; }
            QPushButton:pressed { background-color: #007acc; }
            QProgressBar { border: 1px solid #3c3c3c; border-radius: 3px; text-align: center; background-color: #1e1e1e; color: white; }
            QProgressBar::chunk { background-color: #007acc; }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        title = QLabel("Select Target Format:")
        title.setStyleSheet("font-weight: bold; font-size: 15px;")
        layout.addWidget(title)

        format_layout = QVBoxLayout()
        formats = ["PNG", "JPG", "WEBP", "AvIF"]
        for fmt in formats:
            btn = QPushButton(fmt)
            btn.clicked.connect(lambda checked, f=fmt: self.start_conversion(f))
            format_layout.addWidget(btn)
        layout.addLayout(format_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(15)
        layout.addWidget(self.progress_bar)

        # Settings
        layout.addSpacing(10)
        layout.addWidget(QLabel("Conversion Settings:"))
        
        self.quality_label = QLabel("Quality: 95%")
        layout.addWidget(self.quality_label)
        self.quality_slider = QSlider(Qt.Orientation.Horizontal)
        self.quality_slider.setRange(1, 100); self.quality_slider.setValue(95)
        self.quality_slider.valueChanged.connect(lambda v: self.quality_label.setText(f"Quality: {v}%"))
        layout.addWidget(self.quality_slider)

        self.delete_cb = QCheckBox("Delete original files after conversion")
        self.delete_cb.setStyleSheet("color: #a7dfff; font-size: 11px;")
        layout.addWidget(self.delete_cb)
        
        layout.addSpacing(10)
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("font-size: 11px; color: #888;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

    def start_conversion(self, fmt):
        if not self.items:
            QMessageBox.warning(self, "Warning", "No images to convert.")
            return

        fmt = "jpg" if fmt == "JPG" else fmt.lower()
        if fmt == "avif":
            try:
                Image.new("RGB", (1, 1)).save("test.avif")
                os.remove("test.avif")
            except Exception:
                QMessageBox.critical(self, "Error", "AVIF support is missing. Please install 'pillow-avif-plugin'.")
                return

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        delete_orig = False
        if self.delete_cb.isChecked():
            res = QMessageBox.warning(self, "Confirm Delete", 
                "Are you sure you want to PERMANENTLY delete the original files after conversion?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if res == QMessageBox.StandardButton.Yes:
                delete_orig = True
            else:
                self.delete_cb.setChecked(False)

        self.status_label.setText(f"Converting to {fmt.upper()}...")
        for btn in self.findChildren(QPushButton): btn.setEnabled(False)

        self.worker = ConversionWorker(self.items, fmt, delete_originals=delete_orig, quality=self.quality_slider.value())
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_finished(self, success, total):
        self.status_label.setText(f"Done: {success}/{total} successful")
        QMessageBox.information(self, "Finished", f"Successfully converted {success} of {total} images.")
        self.accept()

class ResizeWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(int, int)

    def __init__(self, items, scale_percent):
        super().__init__()
        self.items = items
        self.scale = scale_percent / 100.0

    def run(self):
        success_count = 0
        total = len(self.items)
        for i, item in enumerate(self.items):
            try:
                with Image.open(item.image_path) as img:
                    new_size = (int(img.width * self.scale), int(img.height * self.scale))
                    resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
                    resized_img.save(item.image_path)
                success_count += 1
            except Exception as e:
                print(f"Error resizing {item.image_path}: {e}")
            self.progress.emit(int((i + 1) / total * 100))
        self.finished.emit(success_count, total)

class ResizeDialog(QDialog):
    def __init__(self, parent, items):
        super().__init__(parent)
        self.items = items
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Scale Images by %")
        self.setFixedWidth(350)
        self.setStyleSheet("""
            QDialog { background-color: #252526; color: #d4d4d4; }
            QLabel { color: #cccccc; font-size: 13px; }
            QPushButton { 
                background-color: #333333; color: #cccccc; border: 1px solid #444; 
                padding: 10px; border-radius: 2px;
            }
            QPushButton:hover { background-color: #444444; }
            QSlider::handle:horizontal { background: #007acc; width: 18px; margin: -5px 0; border-radius: 9px; }
            QSlider::groove:horizontal { border: 1px solid #3c3c3c; height: 8px; background: #1e1e1e; border-radius: 4px; }
        """)

        layout = QVBoxLayout(self)
        self.info_label = QLabel("Scale: 100%")
        self.info_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self.info_label)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(10, 200); self.slider.setValue(100)
        self.slider.valueChanged.connect(self.update_label)
        layout.addWidget(self.slider)

        warning = QLabel("Warning: This will overwrite original images!")
        warning.setStyleSheet("color: #fc0303; font-size: 11px;")
        layout.addWidget(warning)

        self.btn_run = QPushButton("Start Scaling")
        self.btn_run.clicked.connect(self.start_scaling)
        layout.addWidget(self.btn_run)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

    def update_label(self, v): self.info_label.setText(f"Scale: {v}%")

    def start_scaling(self):
        self.slider.setEnabled(False); self.btn_run.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.worker = ResizeWorker(self.items, self.slider.value())
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_finished(self, success, total):
        QMessageBox.information(self, "Finished", f"Scaled {success} images.")
        self.accept()

class MetadataDialog(QDialog):
    def __init__(self, parent, image_path):
        super().__init__(parent)
        self.image_path = image_path
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Image Metadata")
        self.resize(500, 400)
        self.setStyleSheet("background-color: #252526; color: #d4d4d4;")
        layout = QVBoxLayout(self)
        self.text_area = QTextEdit(); self.text_area.setReadOnly(True)
        self.text_area.setStyleSheet("font-family: 'Consolas'; font-size: 12px; background-color: #1e1e1e; border: 1px solid #333;")
        layout.addWidget(self.text_area)
        btn_close = QPushButton("Close"); btn_close.clicked.connect(self.close)
        btn_close.setStyleSheet("background-color: #333; color: #ccc; padding: 5px;")
        layout.addWidget(btn_close)
        self.load_metadata()

    def load_metadata(self):
        try:
            with Image.open(self.image_path) as img:
                info = f"File: {os.path.basename(self.image_path)}\nFormat: {img.format}\nMode: {img.mode}\nSize: {img.width} x {img.height}\n"
                info += f"File Size: {os.path.getsize(self.image_path) / 1024:.2f} KB\n"
                info += "-" * 40 + "\nEXIF Data:\n"
                exif = img._getexif()
                if exif:
                    from PIL.ExifTags import TAGS
                    for tag_id, value in exif.items():
                        tag = TAGS.get(tag_id, tag_id)
                        info += f"{tag}: {value}\n"
                else: info += "No EXIF data found."
                self.text_area.setText(info)
                self.text_area.setText(info)
        except Exception as e: self.text_area.setText(f"Error: {e}")
