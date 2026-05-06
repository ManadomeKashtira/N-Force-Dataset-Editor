import os
import numpy as np
from PIL import Image
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QPushButton, QLabel, QHBoxLayout, 
    QProgressBar, QMessageBox, QComboBox, QCheckBox, QFrame, QSlider, QLineEdit, QFileDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from huggingface_hub import hf_hub_download

class UpscaleWorker(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal(int)

    def __init__(self, items, model_config, target_scale, all_images=True, use_gpu=False):
        super().__init__()
        self.items = items
        self.model_config = model_config
        self.target_scale = target_scale
        self.use_gpu = use_gpu
        self.all_images = all_images
        self._is_paused = False
        self._is_cancelled = False

    def pause(self): self._is_paused = True
    def resume(self): self._is_paused = False
    def cancel(self): self._is_cancelled = True

    def run(self):
        try:
            import onnxruntime as ort
        except ImportError:
            self.log.emit("Error: onnxruntime not installed.")
            self.finished.emit(0)
            return

        try:
            self.log.emit(f"Loading upscaler: {self.model_config['name']}...")
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.use_gpu else ['CPUExecutionProvider']
            
            if self.model_config['path'].lower().endswith('.onnx'):
                session = ort.InferenceSession(self.model_config['path'], providers=providers)
                input_name = session.get_inputs()[0].name
                
                items_to_process = self.items if self.all_images else [self.items[0]]
                success_count = 0
                total = len(items_to_process)

                for i, item in enumerate(items_to_process):
                    while self._is_paused:
                        if self._is_cancelled: break
                        self.msleep(100)
                    if self._is_cancelled: break

                    try:
                        orig_img = Image.open(item.image_path).convert("RGB")
                        w, h = orig_img.size
                        
                        img_np = np.array(orig_img).astype(np.float32) / 255.0
                        img_np = np.transpose(img_np, (2, 0, 1)) # HWC to CHW
                        img_np = np.expand_dims(img_np, axis=0) # CHW to NCHW

                        # Inference
                        output = session.run(None, {input_name: img_np})[0][0]
                        
                        # Post-process
                        output = np.transpose(output, (1, 2, 0)) # CHW to HWC
                        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
                        upscaled_img = Image.fromarray(output)
                        
                        # Final resize to target scale
                        target_w, target_h = int(w * self.target_scale), int(h * self.target_scale)
                        if upscaled_img.size != (target_w, target_h):
                            upscaled_img = upscaled_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                        
                        upscaled_img.save(item.image_path)
                        success_count += 1
                    except Exception as e:
                        self.log.emit(f"Error upscaling {item.filename}: {e}")
                    
                    self.progress.emit(int((i + 1) / total * 100))
            else:
# Safetensors,pth, loader
                try:
                    import torch
                    from spandrel import ModelLoader
                except ImportError:
                    self.log.emit("Error: 'torch' and 'spandrel' required for .pth/.safetensors")
                    self.finished.emit(0)
                    return

                self.log.emit("Loading PyTorch model via Spandrel...")
                device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
                model = ModelLoader().load_from_file(self.model_config['path'])
                model.to(device)
                model.eval()

                items_to_process = self.items if self.all_images else [self.items[0]]
                success_count = 0
                total = len(items_to_process)

                for i, item in enumerate(items_to_process):
                    if self._is_cancelled: break
                    try:
                        orig_img = Image.open(item.image_path).convert("RGB")
                        w, h = orig_img.size
                        
                        # Convert to tensor
                        img_tensor = torch.from_numpy(np.array(orig_img)).permute(2, 0, 1).float() / 255.0
                        img_tensor = img_tensor.unsqueeze(0).to(device)

                        with torch.no_grad():
                            output = model(img_tensor)
                        
                        # Convert back to image
                        output = output.squeeze(0).cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                        output = (output * 255.0).astype(np.uint8)
                        upscaled_img = Image.fromarray(output)

                        # Final resize to target scale
                        target_w, target_h = int(w * self.target_scale), int(h * self.target_scale)
                        if upscaled_img.size != (target_w, target_h):
                            upscaled_img = upscaled_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                        
                        upscaled_img.save(item.image_path)
                        success_count += 1
                    except Exception as e:
                        self.log.emit(f"Error upscaling {item.filename}: {e}")
                    self.progress.emit(int((i + 1) / total * 100))

            
            self.finished.emit(success_count)
        except Exception as e:
            self.log.emit(f"Critical error: {e}")
            self.finished.emit(0)

class UpscaleDialog(QDialog):
    def __init__(self, parent, items, current_index=0, use_gpu=False):
        super().__init__(parent)
        self.items = items
        self.current_index = current_index
        self.use_gpu = use_gpu
        self.models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Upscale Model")
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.available_models = {
            "Remacri 4x": {"repo": "hekmon/ComfyUI-Upscaler-Onnx", "file": "4x_foolhardy_Remacri.onnx"},
            "Real-ESRGAN x4plus": {"repo": "bukuroo/RealESRGAN-ONNX", "file": "real-esrgan-x4plus-128.onnx"},
            "Real-ESRGAN x4": {"repo": "hekmon/ComfyUI-Upscaler-Onnx", "file": "RealESRGAN_x4.onnx"}
        }
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("AI Upscaler (ESRGAN)")
        self.setFixedWidth(450)
        self.setStyleSheet("""
            QDialog { background-color: #1e1e1e; color: #d4d4d4; border: 1px solid #333; }
            QLabel { color: #cccccc; font-size: 13px; }
            QPushButton { 
                background-color: #333333; color: #cccccc; border: 1px solid #444; 
                padding: 10px; border-radius: 4px; font-weight: bold;
            }
            QPushButton:hover { background-color: #444444; }
            QPushButton#StartBtn { border: 2px solid #800000; color: #a7dfff; }
            QComboBox { background-color: #252526; color: #ccc; border: 1px solid #444; padding: 10px; border-radius: 20px; }
            QSlider::handle:horizontal { background: #800000; width: 14px; height: 14px; border-radius: 7px; margin: -5px 0; }
            QSlider::groove:horizontal { border: 1px solid #333; height: 4px; background: #111; }
            QCheckBox { spacing: 5px; color: #a7dfff; font-weight: bold; }
            QLineEdit { background-color: #252526; color: #ccc; border: 1px solid #444; padding: 8px; border-top-left-radius: 4px; border-bottom-left-radius: 4px; }
        """)


        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        title = QLabel("Upscale AI")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #a7dfff;")
        layout.addWidget(title)

        layout.addWidget(QLabel("Select Model"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(list(self.available_models.keys()))
        self.model_combo.addItem("Custom Model...")
        self.model_combo.currentIndexChanged.connect(self.toggle_custom)
        layout.addWidget(self.model_combo)

        # Custom Model Path UI
        self.custom_container = QFrame()
        self.custom_container.setVisible(False)
        self.custom_layout = QHBoxLayout(self.custom_container)
        self.custom_layout.setContentsMargins(0, 0, 0, 0)
        self.custom_layout.setSpacing(0)
        
        self.model_path_input = QLineEdit()
        self.model_path_input.setPlaceholderText("Select .onnx, .pth, or .safetensors")
        self.btn_browse = QPushButton("Browse")
        self.btn_browse.setFixedWidth(80)
        self.btn_browse.setStyleSheet("border-radius: 0; border-top-right-radius: 4px; border-bottom-right-radius: 4px;")
        self.btn_browse.clicked.connect(self.browse_model)
        
        self.custom_layout.addWidget(self.model_path_input)
        self.custom_layout.addWidget(self.btn_browse)
        layout.addWidget(self.custom_container)

        # Scale Slider
        layout.addWidget(QLabel("Upscale Ratio (Multiplier)"))
        self.scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.scale_slider.setRange(2, 8)
        self.scale_slider.setValue(4)
        self.scale_val_label = QLabel("4x")
        self.scale_slider.valueChanged.connect(lambda v: self.scale_val_label.setText(f"{v}x"))
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(self.scale_slider)
        scale_layout.addWidget(self.scale_val_label)
        layout.addLayout(scale_layout)

        self.all_images_cb = QCheckBox("Upscale All Images in Dataset")
        self.all_images_cb.setChecked(False)
        layout.addWidget(self.all_images_cb)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine); sep.setStyleSheet("background-color: #a7dfff"); sep.setFixedHeight(1)
        layout.addWidget(sep)

        self.start_btn = QPushButton("Start Upscaling")
        self.start_btn.setObjectName("StartBtn")
        self.start_btn.clicked.connect(self.start_process)
        layout.addWidget(self.start_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("QProgressBar { height: 10px; border-radius: 5px; background: #111; color: transparent; text-align: center; } QProgressBar::chunk { background: #800000; }")
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-size: 11px; color: #666;")
        layout.addWidget(self.status_label)

    def toggle_custom(self):
        is_custom = self.model_combo.currentText() == "Custom Model..."
        self.custom_container.setVisible(is_custom)

    def browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Upscale Model", "", 
            "All-in-one Models (*.onnx *.pth *.safetensors);;ONNX Models (*.onnx);;PyTorch Models (*.pth *.safetensors)"
        )
        if file_path:
            self.model_path_input.setText(file_path)

    def start_process(self):
        model_name = self.model_combo.currentText()
        local_path = ""
        
        if model_name == "Custom Model...":
            local_path = self.model_path_input.text()
            if not local_path or not os.path.exists(local_path):
                QMessageBox.warning(self, "Missing Model", "Please select a valid local model file.")
                return
        else:
            config = self.available_models[model_name]
            model_name_clean = model_name.replace(" ", "_").replace(".", "_")
            model_folder = os.path.join(self.models_dir, model_name_clean)
            os.makedirs(model_folder, exist_ok=True)
            local_path = os.path.join(model_folder, config['file'])
            
            if not os.path.exists(local_path):
                self.status_label.setText(f"Checking/Downloading {model_name}...")
                self.start_btn.setEnabled(False)
                try:
                    local_path = hf_hub_download(
                        repo_id=config['repo'], 
                        filename=config['file'],
                        local_dir=model_folder,
                        local_dir_use_symlinks=False
                    )
                except Exception as e:
                    QMessageBox.critical(self, "Download Error", f"Failed to download model: {e}")
                    self.start_btn.setEnabled(True)
                    return


        items_to_pass = self.items
        if not self.all_images_cb.isChecked():
            items_to_pass = [self.items[self.current_index]]

        self.start_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        
        self.worker = UpscaleWorker(
            items_to_pass, 
            {"name": model_name, "path": local_path}, 
            self.scale_slider.value(),
            self.all_images_cb.isChecked(),
            use_gpu=self.use_gpu
        )
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.log.connect(self.status_label.setText)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_finished(self, count):
        QMessageBox.information(self, "Finished", f"Upscaled {count} images successfully.")
        self.accept()

