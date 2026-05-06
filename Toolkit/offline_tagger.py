import os
import csv
import json
from PIL import Image

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QPushButton, QLabel, QHBoxLayout, 
    QProgressBar, QMessageBox, QComboBox, QSlider, QCheckBox, QFrame,
    QFileDialog, QLineEdit, QWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from huggingface_hub import hf_hub_download

class TaggerWorker(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal(int)

    def __init__(self, items, model_config, threshold, char_threshold, append_mode, use_gpu=False):
        super().__init__()
        self.items = items
        self.model_config = model_config
        self.use_gpu = use_gpu
        self.threshold = threshold / 100.0
        self.char_threshold = char_threshold / 100.0
        self.append_mode = append_mode
        self._is_paused = False
        self._is_cancelled = False

    def pause(self): self._is_paused = True
    def resume(self): self._is_paused = False
    def cancel(self): self._is_cancelled = True

    def run(self):
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError:
            self.log.emit("Error: numpy or onnxruntime not installed. Please run 'pip install numpy onnxruntime'")
            self.finished.emit(0)
            return

        import os 

        # 1. Load Model
        try:
            self.log.emit(f"Loading model: {self.model_config['name']}...")
            is_llm = self.model_config.get('is_llm', False)
            prompt = self.model_config.get('prompt', 'Describe this image.')
            model_path = self.model_config['model_path']
            
            # ─── LLM / Safetensors Mode (Transformers) ───
            if is_llm:
                try:
                    import torch
                    from transformers import AutoProcessor, AutoModelForVision2Seq
                except ImportError:
                    self.log.emit("Error: 'transformers' or 'torch' not installed. Please run: pip install transformers torch accelerate")
                    self.finished.emit(0)
                    return

                device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
                self.log.emit(f"Initializing LLM on {device}...")
                
                try:
                    # Detect if it's a multimodal model (has processor)
                    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                    model = AutoModelForVision2Seq.from_pretrained(
                        model_path, 
                        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                        trust_remote_code=True
                    ).to(device)
                    model.eval()
                except Exception as e:
                    self.log.emit(f"Failed to load HF model: {e}")
                    self.finished.emit(0)
                    return

                success_count = 0
                total = len(self.items)

                for i, item in enumerate(self.items):
                    while self._is_paused:
                        if self._is_cancelled: break
                        self.msleep(100)
                    if self._is_cancelled: break

                    try:
                        self.log.emit(f"[{i+1}/{total}] Prompting: {item.filename}")
                        image = Image.open(item.image_path).convert("RGB")
                        
                        # Process For Multimodal LLM
                        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch.float16 if device == "cuda" else torch.float32)
                        
                        generated_ids = model.generate(**inputs, max_new_tokens=100)
                        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                        
                        final_tags = generated_text.replace(prompt, "").strip()
                        
                        # Save
                        current_tags = ""
                        if self.append_mode:
                            current_tags = item.load_caption()
                            if current_tags and not current_tags.endswith(", "): current_tags += ", "
                        item.save_caption(current_tags + final_tags)
                        success_count += 1
                    except Exception as e:
                        self.log.emit(f"Error {item.filename}: {e}")
                    
                    self.progress.emit(int((i + 1) / total * 100))
                
                self.finished.emit(success_count)
                return

# ─── ONNX Classification Mode (Standard Tagger) ───
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.use_gpu else ['CPUExecutionProvider']
            session = ort.InferenceSession(model_path, providers=providers)
            input_name = session.get_inputs()[0].name
            
            # 2. Load Tags (if not LLM)
            tags = []
            if self.model_config['tags_path'] and os.path.exists(self.model_config['tags_path']):
                if self.model_config['tags_path'].endswith('.json'):
                    with open(self.model_config['tags_path'], 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, dict) and 'tags' in data:
                            tags = data['tags']
                        elif isinstance(data, list):
                            tags = data
                        else:
                            self.log.emit("Error: Unexpected JSON structure for tags.")
                            self.finished.emit(0)
                            return
                else:
                    with open(self.model_config['tags_path'], 'r', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        try: next(reader) 
                        except: pass
                        for row in reader:
                            if len(row) >= 2: tags.append(row[1]) 
            
            success_count = 0
            total = len(self.items)

            for i, item in enumerate(self.items):
                while self._is_paused:
                    if self._is_cancelled: break
                    self.msleep(100)
                if self._is_cancelled: break

                try:
                    # 3. Process Image
                    img = Image.open(item.image_path).convert("RGB")
                    img = self.prepare_image(img, 448) # Default tagger size
                    import numpy as np
                    img_arr = np.array(img).astype(np.float32)

                    img_arr = img_arr[:, :, ::-1] # RGB to BGR
                    img_arr = np.expand_dims(img_arr, axis=0)

                    # 4. Inference
                    outputs = session.run(None, {input_name: img_arr})
                    
                    # 5. Extract Classification Tags
                    probs = outputs[0][0]
                    res = []
                    for j, prob in enumerate(probs):
                        if j < len(tags) and prob >= self.threshold:
                            res.append(tags[j])
                    final_tags = ", ".join(res)
                    
                    # 6. Save Tags
                    current_tags = ""
                    if self.append_mode:
                        current_tags = item.load_caption()
                        if current_tags and not current_tags.endswith(", "): current_tags += ", "
                    
                    item.save_caption(current_tags + final_tags)
                    success_count += 1
                except Exception as e:
                    self.log.emit(f"Error processing {item.filename}: {e}")
                
                self.progress.emit(int((i + 1) / total * 100))
            
            self.finished.emit(success_count)
        except Exception as e:
            self.log.emit(f"Critical error: {e}")
            self.finished.emit(0)

    def prepare_image(self, image, target_size):
        # padding and resizing
        width, height = image.size
        max_dim = max(width, height)
        new_img = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        new_img.paste(image, ((max_dim - width) // 2, (max_dim - height) // 2))
        return new_img.resize((target_size, target_size), Image.Resampling.LANCZOS)


class CustomModelDialog(QDialog):
    """Dialog for selecting a custom .onnx model and its tags file (CSV or JSON)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Load Custom Model")
        self.setFixedWidth(500)
        self.model_path = ""
        self.tags_path = ""
        self.setStyleSheet("""
            QDialog { background-color: #1e1e1e; color: #d4d4d4; border: 1px solid #333; }
            QLabel { color: #cccccc; font-size: 13px; }
            QLabel#SectionLabel { color: #007acc; font-size: 14px; font-weight: bold; }
            QLineEdit { background-color: #252526; color: #ccc; border: 1px solid #444; padding: 8px; border-radius: 4px; }
            QPushButton { 
                background-color: #333333; color: #cccccc; border: 5px solid #444; 
                padding: 8px 16px; border-radius: 4px; font-weight: bold;
            }
            QPushButton:hover { background-color: #444444; }
            QPushButton#ConfirmBtn { border: 2px solid #007acc; color: #007acc; }
            QPushButton#ConfirmBtn:hover { background-color: #007acc; }
        """)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        title = QLabel("Load Custom ONNX Model")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #007acc;")
        layout.addWidget(title)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: #007acc;")
        sep.setFixedHeight(1)
        layout.addWidget(sep)

# Load saved paths
        from PyQt6.QtCore import QSettings
        settings = QSettings("N-Force", "DatasetEditor")
        saved_model = settings.value("custom_model_path", "")
        saved_tags = settings.value("custom_tags_path", "")
        saved_is_llm = settings.value("custom_is_llm", False, type=bool)
        saved_prompt = settings.value("custom_prompt", "Describe this image.")

        self.model_path = saved_model if saved_model and os.path.exists(saved_model) else ""
        self.tags_path = saved_tags if saved_tags and os.path.exists(saved_tags) else ""

# LLM Toggle
        self.llm_cb = QCheckBox("Custom LLM Only (e.g. Gemma, Llama)")
        self.llm_cb.setChecked(saved_is_llm)
        self.llm_cb.toggled.connect(self.toggle_llm_mode)
        layout.addWidget(self.llm_cb)

# Prompt Section
        self.prompt_label = QLabel("System Prompt")
        self.prompt_label.setObjectName("SectionLabel")
        layout.addWidget(self.prompt_label)
        self.prompt_edit = QLineEdit()
        self.prompt_edit.setPlaceholderText("Enter prompt (e.g. Describe this image in detail)...")
        self.prompt_edit.setText(saved_prompt)
        layout.addWidget(self.prompt_edit)

        layout.addSpacing(10)

# Model File Section
        model_label = QLabel("Model File (.onnx)")
        model_label.setObjectName("SectionLabel")
        layout.addWidget(model_label)

        model_row = QHBoxLayout()
        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("Select .onnx model file...")
        self.model_edit.setText(self.model_path)
        self.model_edit.setReadOnly(True)
        model_browse = QPushButton("Browse")
        model_browse.clicked.connect(self.browse_model)
        model_row.addWidget(self.model_edit)
        model_row.addWidget(model_browse)
        layout.addLayout(model_row)

        layout.addSpacing(5)

# Tags File Section
        self.tags_label = QLabel("Tags File (.csv or .json)")
        self.tags_label.setObjectName("SectionLabel")
        layout.addWidget(self.tags_label)

        self.tags_row_widget = QWidget()
        tags_row = QHBoxLayout(self.tags_row_widget)
        tags_row.setContentsMargins(0,0,0,0)
        self.tags_edit = QLineEdit()
        self.tags_edit.setPlaceholderText("Select tags file (CSV or JSON)...")
        self.tags_edit.setText(self.tags_path)
        self.tags_edit.setReadOnly(True)
        self.tags_browse = QPushButton("Browse")
        self.tags_browse.clicked.connect(self.browse_tags)
        tags_row.addWidget(self.tags_edit)
        tags_row.addWidget(self.tags_browse)
        layout.addWidget(self.tags_row_widget)

        layout.addSpacing(15)

        self.toggle_llm_mode(saved_is_llm)

        
        btn_row = QHBoxLayout()
        confirm_btn = QPushButton("Confirm")
        confirm_btn.setObjectName("ConfirmBtn")
        confirm_btn.clicked.connect(self.validate_and_accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(confirm_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

    def toggle_llm_mode(self, checked):
        """Enable/Disable tags file selection and prompt based on LLM mode."""
        self.tags_label.setVisible(not checked)
        self.tags_row_widget.setVisible(not checked)
        self.prompt_label.setVisible(checked)
        self.prompt_edit.setVisible(checked)

    def browse_model(self):
        is_llm = self.llm_cb.isChecked()
        if is_llm:
            path = QFileDialog.getExistingDirectory(self, "Select HF Model Directory (contains safetensors)")
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select ONNX Model", "", "ONNX Model (*.onnx);;All Files (*)"
            )
        
        if path:
            self.model_path = path
            self.model_edit.setText(path)

    def browse_tags(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Tags File", "", "Tags Files (*.csv *.json);;CSV Files (*.csv);;JSON Files (*.json);;All Files (*)"
        )
        if path:
            self.tags_path = path
            self.tags_edit.setText(path)

    def validate_and_accept(self):
        if not self.model_path:
            QMessageBox.warning(self, "Missing Model", "Please select an .onnx model file.")
            return
        
        is_llm = self.llm_cb.isChecked()
        if not is_llm and not self.tags_path:
            QMessageBox.warning(self, "Missing Tags", "Please select a tags file (CSV or JSON).")
            return
            
        if not os.path.exists(self.model_path):
            QMessageBox.warning(self, "File Not Found", f"Model file not found:\n{self.model_path}")
            return
            
        if not is_llm and not os.path.exists(self.tags_path):
            QMessageBox.warning(self, "File Not Found", f"Tags file not found:\n{self.tags_path}")
            return
        
# Save custom paths
        from PyQt6.QtCore import QSettings
        settings = QSettings("N-Force", "DatasetEditor")
        settings.setValue("custom_model_path", self.model_path)
        settings.setValue("custom_tags_path", self.tags_path)
        settings.setValue("custom_is_llm", is_llm)
        settings.setValue("custom_prompt", self.prompt_edit.text())
        
        self.accept()


class OfflineTaggerDialog(QDialog):
    def __init__(self, parent, items, use_gpu=False):
        super().__init__(parent)
        self.items = items
        self.use_gpu = use_gpu
        self.models = {
            "WD EVA02 Large v3": {"repo": "SmilingWolf/wd-eva02-large-tagger-v3", "model": "model.onnx", "tags": "selected_tags.csv"},
            "PixAI v0.9": {"repo": "deepghs/pixai-tagger-v0.9-onnx", "model": "model.onnx", "tags": "selected_tags.csv"},
            "Camie v2": {"repo": "Camais03/camie-tagger-v2", "model": "camie-tagger-v2.onnx", "tags": "camie-tagger-v2-metadata.json"}
        }
        self.custom_model_config = None  
        self.load_settings()
        self.init_ui()

    def load_settings(self):
        from PyQt6.QtCore import QSettings
        settings = QSettings("N-Force", "DatasetEditor")
        self.last_model = settings.value("offline_tagger_model", "WD EVA02 Large v3")
        self.last_append = settings.value("offline_tagger_append", False, type=bool)
        self.last_tag_threshold = settings.value("offline_tagger_threshold", 35, type=int)
        self.last_char_threshold = settings.value("offline_tagger_char_threshold", 75, type=int)
        
        custom_model = settings.value("custom_model_path", "")
        custom_tags = settings.value("custom_tags_path", "")
        custom_is_llm = settings.value("custom_is_llm", False, type=bool)
        custom_prompt = settings.value("custom_prompt", "Describe this image.")

        if custom_model and os.path.exists(custom_model):
            if custom_is_llm or (custom_tags and os.path.exists(custom_tags)):
                self.custom_model_config = {
                    "name": "Custom Model",
                    "model_path": custom_model,
                    "tags_path": custom_tags,
                    "is_llm": custom_is_llm,
                    "prompt": custom_prompt
                }

    def init_ui(self):
        self.setWindowTitle("Offline Auto Tagger")
        self.setFixedWidth(450)
        self.setStyleSheet("""
            QDialog { background-color: #1e1e1e; color: #d4d4d4; border: 3px solid #333; }
            QLabel { color: #cccccc; font-size: 13px; }
            QPushButton { 
                background-color: #333333; color: #cccccc; border: 1px solid #444; 
                padding: 10px; border-radius: 4px; font-weight: bold;
            }
            QPushButton:hover { background-color: #34AAFB; }
            QPushButton#StartBtn { border: 2px solid #007aff; color: #34AAFB; }
            QPushButton#PauseBtn { border: 2px solid #555; }
            QPushButton#CustomBtn { border: 2px solid #665500; color: #ccaa44; }
            QPushButton#CustomBtn:hover { background-color: #3a3520; }
            QComboBox { background-color: #252526; color: #ccc; border: 1px solid #444; padding: 10px; border-radius: 20px; }
            QSlider::handle:horizontal { background: #007aff; width: 14px; height: 14px; border-radius: 7px; margin: -5px 0; }
            QSlider::groove:horizontal { border: 1px solid #333; height: 4px; background: #111; }
            QCheckBox { spacing: 5px; color: #007acc; font-weight: bold; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Header
        header = QHBoxLayout()
        title = QLabel("AUTO Tags Model")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #007acc;")
        self.append_cb = QCheckBox("Append Tags?")
        self.append_cb.setChecked(self.last_append)
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.append_cb)
        layout.addLayout(header)

        # Model Selector
        model_row = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(list(self.models.keys()))
        self.model_combo.addItem("✦ Custom Model")
        
        # Add a settings button for custom model
        self.custom_settings_btn = QPushButton("⚙")
        self.custom_settings_btn.setFixedWidth(40)
        self.custom_settings_btn.setToolTip("Configure Custom Model")
        self.custom_settings_btn.clicked.connect(self.configure_custom_model)
        
        # Apply last model
        idx = self.model_combo.findText(self.last_model)
        if idx >= 0: 
            self.model_combo.setCurrentIndex(idx)
            self.custom_settings_btn.setVisible(self.last_model == "✦ Custom Model")
        else:
            self.custom_settings_btn.setVisible(False)

        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        model_row.addWidget(self.model_combo, 1)
        model_row.addWidget(self.custom_settings_btn)
        layout.addLayout(model_row)

        # Custom Model Info Label (hidden)
        self.custom_info_label = QLabel("")
        if self.custom_model_config:
            model_name = os.path.basename(self.custom_model_config["model_path"])
            tags_name = os.path.basename(self.custom_model_config["tags_path"])
            self.custom_info_label.setText(f"Model: {model_name}\nTags: {tags_name}")
            self.custom_info_label.setVisible(self.model_combo.currentText() == "✦ Custom Model")
        else:
            self.custom_info_label.setVisible(False)
        self.custom_info_label.setStyleSheet("color: #34AAFB; font-size: 11px; padding: 4px;")
        self.custom_info_label.setWordWrap(True)
        layout.addWidget(self.custom_info_label)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine); sep.setStyleSheet("background-color: #007aff;"); sep.setFixedHeight(1)
        layout.addWidget(sep)

        # Thresholds
        layout.addWidget(QLabel("Tags Treshold"))
        self.tag_slider = QSlider(Qt.Orientation.Horizontal)
        self.tag_slider.setRange(0, 100); self.tag_slider.setValue(self.last_tag_threshold)
        self.tag_val_label = QLabel(f"{self.last_tag_threshold/100:.2f}")
        self.tag_slider.valueChanged.connect(lambda v: self.tag_val_label.setText(f"{v/100:.2f}"))
        row1 = QHBoxLayout(); row1.addWidget(self.tag_slider); row1.addWidget(self.tag_val_label)
        layout.addLayout(row1)

        layout.addWidget(QLabel("Character Treshold"))
        self.char_slider = QSlider(Qt.Orientation.Horizontal)
        self.char_slider.setRange(0, 100); self.char_slider.setValue(self.last_char_threshold)
        self.char_val_label = QLabel(f"{self.last_char_threshold/100:.2f}")
        self.char_slider.valueChanged.connect(lambda v: self.char_val_label.setText(f"{v/100:.2f}"))
        row2 = QHBoxLayout(); row2.addWidget(self.char_slider); row2.addWidget(self.char_val_label)
        layout.addLayout(row2)

        # Buttons
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start"); self.start_btn.setObjectName("StartBtn")
        self.pause_btn = QPushButton("Pause"); self.pause_btn.setObjectName("PauseBtn"); self.pause_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_process)
        self.pause_btn.clicked.connect(self.toggle_pause)
        btn_layout.addWidget(self.start_btn); btn_layout.addWidget(self.pause_btn)
        layout.addLayout(btn_layout)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("QProgressBar { height: 15px; border-radius: 5px; background: #111; color: transparent; text-align: center; } QProgressBar::chunk { background: #800000; }")
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #34AAFB; font-size: 7px;")
        layout.addWidget(self.status_label)

    def on_model_changed(self, text):
        """Handle model selection changes, show custom model dialog when needed."""
        is_custom = (text == "✦ Custom Model")
        self.custom_settings_btn.setVisible(is_custom)
        
        if is_custom:
            if not self.custom_model_config:
                self.configure_custom_model()
            else:
                self.update_custom_info_label()
        else:
            self.custom_model_config = None
            self.custom_info_label.setVisible(False)

    def configure_custom_model(self):
        """Open the custom model dialog and update config."""
        dialog = CustomModelDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.custom_model_config = {
                "name": "Custom Model",
                "model_path": dialog.model_path,
                "tags_path": dialog.tags_path,
                "is_llm": dialog.llm_cb.isChecked(),
                "prompt": dialog.prompt_edit.text()
            }
            self.update_custom_info_label()
        elif not self.custom_model_config:
            self.model_combo.blockSignals(True)
            self.model_combo.setCurrentIndex(0)
            self.model_combo.blockSignals(False)
            self.custom_settings_btn.setVisible(False)
            self.custom_info_label.setVisible(False)

    def update_custom_info_label(self):
        """Update the info label with current custom model details."""
        if not self.custom_model_config:
            self.custom_info_label.setVisible(False)
            return
            
        model_name = os.path.basename(self.custom_model_config["model_path"])
        if self.custom_model_config.get("is_llm"):
            prompt_preview = self.custom_model_config.get("prompt", "")[:50]
            self.custom_info_label.setText(f"Model (LLM): {model_name}\nPrompt: {prompt_preview}...")
        else:
            tags_name = os.path.basename(self.custom_model_config.get("tags_path", ""))
            self.custom_info_label.setText(f"Model: {model_name}\nTags: {tags_name}")
        self.custom_info_label.setVisible(True)

    def start_process(self):
        from PyQt6.QtCore import QSettings
        settings = QSettings("N-Force", "DatasetEditor")
        model_name = self.model_combo.currentText()
        settings.setValue("offline_tagger_model", model_name)
        settings.setValue("offline_tagger_append", self.append_cb.isChecked())
        settings.setValue("offline_tagger_threshold", self.tag_slider.value())
        settings.setValue("offline_tagger_char_threshold", self.char_slider.value())

        # Custom Model
        if model_name == "✦ Custom Model":
            if not self.custom_model_config:
                QMessageBox.warning(self, "No Custom Model", "Please select a custom model first.")
                return
            worker_config = self.custom_model_config
            self.status_label.setText("Loading custom model...")
            self.start_btn.setEnabled(False)

            self.worker = TaggerWorker(
                self.items,
                worker_config,
                self.tag_slider.value(),
                self.char_slider.value(),
                self.append_cb.isChecked(),
                use_gpu=self.use_gpu
            )
            self.worker.progress.connect(self.progress_bar.setValue)
            self.worker.log.connect(self.status_label.setText)
            self.worker.finished.connect(self.on_finished)
            self.worker.start()
            self.pause_btn.setEnabled(True)
            return

        # Checking / Downloading model
        config = self.models[model_name]
        
        self.status_label.setText("Checking/Downloading model...")
        self.start_btn.setEnabled(False)
        
        try:
            import sys
        
            if getattr(sys, 'frozen', False):
                
                base_dir = os.path.dirname(sys.executable)
            else:
                base_dir = os.path.dirname(os.path.abspath(__file__))
            
            base_model_dir = os.path.join(base_dir, "Model")
            model_folder = os.path.join(base_model_dir, model_name.replace(" ", "_"))
            
            if not os.path.exists(model_folder):
                os.makedirs(model_folder, exist_ok=True)

            model_path = os.path.join(model_folder, config['model'])
            tags_path = os.path.join(model_folder, config['tags'])

            if not os.path.exists(model_path) or not os.path.exists(tags_path):
                
                QMessageBox.information(self, "Download Model", f"The model '{model_name}' will now be downloaded from Hugging Face.\n\nThis may take a few minutes depending on your connection.")
                self.status_label.setText("Downloading model from Hugging Face...")
                
                model_path = hf_hub_download(
                    repo_id=config['repo'], 
                    filename=config['model'], 
                    local_dir=model_folder,
                    local_dir_use_symlinks=False
                )
                tags_path = hf_hub_download(
                    repo_id=config['repo'], 
                    filename=config['tags'], 
                    local_dir=model_folder,
                    local_dir_use_symlinks=False
                )
            
            worker_config = {
                "name": model_name,
                "model_path": model_path,
                "tags_path": tags_path
            }
            
            self.worker = TaggerWorker(
                self.items, 
                worker_config, 
                self.tag_slider.value(), 
                self.char_slider.value(),
                self.append_cb.isChecked(),
                use_gpu=self.use_gpu
            )
            self.worker.progress.connect(self.progress_bar.setValue)
            self.worker.log.connect(self.status_label.setText)
            self.worker.finished.connect(self.on_finished)
            self.worker.start()
            
            self.pause_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Download Error", f"Failed to download model: {str(e)}")
            self.status_label.setText(f"Error: {str(e)}")
            self.start_btn.setEnabled(True)

    def toggle_pause(self):
        if self.worker.isRunning():
            if self.pause_btn.text() == "Pause":
                self.worker.pause()
                self.pause_btn.setText("Resume")
            else:
                self.worker.resume()
                self.pause_btn.setText("Pause")

    def on_finished(self, count):
        QMessageBox.information(self, "Finished", f"Successfully tagged {count} images.")
        self.accept()
