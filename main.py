import sys
import os
import re
import shutil
from collections import Counter
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QListView, QTextEdit, QLabel, QPushButton, QFileDialog, 
    QLineEdit, QToolBar, QStatusBar, QSplitter, QStyle, QInputDialog, QMessageBox, QDialog,
    QSlider, QSpinBox, QScrollArea, QFrame, QCompleter, QAbstractItemView,
    QCheckBox, QComboBox, QProgressBar, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QLayout, QSizePolicy, QColorDialog, QMenu
)
from PyQt6.QtCore import Qt, QAbstractListModel, QModelIndex, QSize, pyqtSignal, QThread, QTimer, QRegularExpression, QStringListModel, QRect, QPoint, QPointF
from PyQt6.QtGui import QPixmap, QIcon, QAction, QSyntaxHighlighter, QTextCharFormat, QColor, QFont, QTextCursor, QBrush, QImage, QPainter
from offline_tagger import OfflineTaggerDialog
from image_upscaler import UpscaleDialog
from online_services import OnlineWorker
from clean_up import CleanUpDialog
from image_converter import ConvertDialog, ResizeDialog, MetadataDialog
import json


class FlowLayout(QLayout):
    """A flow layout that arranges widgets in a wrapping row."""
    def __init__(self, parent=None, margin=4, spacing=4):
        super().__init__(parent)
        self._items = []
        self._margin = margin
        self._spacing = spacing
        if parent is not None:
            self.setContentsMargins(margin, margin, margin, margin)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self._margin * 2
        size += QSize(m, m)
        return size

    def _do_layout(self, rect, test_only):
        x = rect.x()
        y = rect.y()
        line_height = 0
        for item in self._items:
            wid = item.widget()
            space_x = self._spacing
            space_y = self._spacing
            next_x = x + item.sizeHint().width() + space_x
            if next_x - space_x > rect.right() and line_height > 0:
                x = rect.x()
                y = y + line_height + space_y
                next_x = x + item.sizeHint().width() + space_x
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))
            x = next_x
            line_height = max(line_height, item.sizeHint().height())
        return y + line_height - rect.y()

    def clear_widgets(self):
        while self.count():
            item = self.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()



class ZoomableGraphicsView(QGraphicsView):
    colorPicked = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor("#1a1a1a")))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self._zoom = 0

    def setPixmap(self, pixmap):
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom = 0
#Zoom Logic
    def wheelEvent(self, event):
        
        if event.angleDelta().y() > 0:
            factor = 1.25
            self._zoom += 1
        else:
            factor = 0.8
            self._zoom -= 1
        
        if self._zoom > 0:
            self.scale(factor, factor)
        elif self._zoom <= 0:
# Reset to fit view
            self._zoom = 0
            self.resetTransform()
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        
    def mouseDoubleClickEvent(self, event):
# Double click to reset zoom
        self._zoom = 0
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
# Map click to image coordinates
            scene_pos = self.mapToScene(event.pos())
            pixmap = self._pixmap_item.pixmap()
            if not pixmap.isNull() and self._pixmap_item.contains(scene_pos):
# Get color from pixel
                local_pos = self._pixmap_item.mapFromScene(scene_pos)
                img = pixmap.toImage()
                x, y = int(local_pos.x()), int(local_pos.y())
                if 0 <= x < img.width() and 0 <= y < img.height():
                    color = QColor(img.pixelColor(x, y))
                    hex_color = color.name().lower()
                    
                    menu = QMenu(self)
                    menu.setStyleSheet("QMenu { background-color: #252526; color: white; border: 1px solid #444; }")
                    
# Color swatch
                    swatch = QPixmap(16, 16)
                    swatch.fill(color)
                    icon = QIcon(swatch)
                    
                    pick_act = QAction(icon, f"Add color Tag: {hex_color}", self)
                    pick_act.triggered.connect(lambda: self.colorPicked.emit(hex_color))
                    menu.addAction(pick_act)
                    
 # Window Color Picker
                    open_picker_act = QAction("Open Color Picker...", self)
                    open_picker_act.triggered.connect(self.open_full_picker)
                    menu.addAction(open_picker_act)
                    
                    menu.exec(event.globalPosition().toPoint())
                    return
        super().mousePressEvent(event)

    def open_full_picker(self):
        color = QColorDialog.getColor(Qt.GlobalColor.white, self, "Select Color")
        if color.isValid():
            self.colorPicked.emit(color.name().lower())

    def resizeEvent(self, event):
        if self._zoom == 0:
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        super().resizeEvent(event)

class TagHighlighter(QSyntaxHighlighter):
    def __init__(self, parent):
        super().__init__(parent)
        self.highlight_rules = []
        self.format = QTextCharFormat()
        self.format.setBackground(QColor("#323c4d"))
        self.format.setForeground(QColor("#ffffff"))
        self.format.setFontWeight(700)

    def set_tags(self, tags_string):
        self.highlight_rules = []
        if tags_string:
            tags = [t.strip() for t in tags_string.split(",") if t.strip()]
            for tag in tags:
                pattern = QRegularExpression(re.escape(tag), QRegularExpression.PatternOption.CaseInsensitiveOption)
                self.highlight_rules.append(pattern)
        self.rehighlight()

    def highlightBlock(self, text):
        for pattern in self.highlight_rules:
            match_iterator = pattern.globalMatch(text)
            while match_iterator.hasNext():
                match = match_iterator.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), self.format)

class CompleterLoader(QThread):
    finished = pyqtSignal(list)
    def run(self):
        tags = []
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "auto-complete.txt")
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    tags = [line.strip() for line in f if line.strip()]
            except Exception as e:
                print(f"Error loading completer: {e}")
        self.finished.emit(tags)

class TagTextEdit(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._completer = None
        self.tags_mode = True

    def setCompleter(self, completer):
        if self._completer:
            self._completer.activated.disconnect()
        self._completer = completer
        if not self._completer:
            return
        self._completer.setWidget(self)
        self._completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self._completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._completer.activated.connect(self.insertCompletion)

    def completer(self):
        return self._completer

    def insertCompletion(self, completion):
        if not self._completer or self._completer.widget() != self:
            return
        tc = self.textCursor()
        extra = len(completion) - len(self._completer.completionPrefix())
        tc.movePosition(QTextCursor.MoveOperation.Left)
        tc.movePosition(QTextCursor.MoveOperation.EndOfWord)
        tc.insertText(completion[-extra:])
        self.setTextCursor(tc)

    def textUnderCursor(self):
        tc = self.textCursor()
        tc.select(QTextCursor.SelectionType.WordUnderCursor)
        return tc.selectedText()

    def focusInEvent(self, e):
        if self._completer:
            self._completer.setWidget(self)
        super().focusInEvent(e)

    def keyPressEvent(self, e):
        if not self.tags_mode or not self._completer:
            super().keyPressEvent(e)
            return

        if self._completer.popup() and self._completer.popup().isVisible():
            if e.key() in (Qt.Key.Key_Enter, Qt.Key.Key_Return, Qt.Key.Key_Escape, Qt.Key.Key_Tab, Qt.Key.Key_Backtab):
                e.ignore()
                return

        isShortcut = e.modifiers() & Qt.KeyboardModifier.ControlModifier and e.key() == Qt.Key.Key_E
        if not isShortcut:
            super().keyPressEvent(e)

        ctrlOrShift = e.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
        if not isShortcut and (not e.text() or len(e.text()) == 0 or ctrlOrShift):
            if self._completer.popup(): self._completer.popup().hide()
            return

        completionPrefix = self.textUnderCursor()
        if not isShortcut and (len(completionPrefix) < 2):
            if self._completer.popup(): self._completer.popup().hide()
            return

        if completionPrefix != self._completer.completionPrefix():
            self._completer.setCompletionPrefix(completionPrefix)
            if self._completer.popup():
                self._completer.popup().setCurrentIndex(self._completer.completionModel().index(0, 0))

        cr = self.cursorRect()
        cr.translate(0, 25)
        
        if self._completer.popup():
            cr.setWidth(self._completer.popup().sizeHintForColumn(0) + self._completer.popup().verticalScrollBar().sizeHint().width())
        self._completer.complete(cr)

class DatasetItem:
    def __init__(self, image_path, text_path):
        self.image_path = image_path
        self.text_path = text_path
        self.filename = os.path.basename(image_path)
        self.caption = ""
        self._loaded_caption = False
        self.icon = None

    def load_caption(self):
        if not self._loaded_caption:
            if os.path.exists(self.text_path):
                try:
                    with open(self.text_path, 'r', encoding='utf-8') as f:
                        self.caption = f.read()
                except Exception: self.caption = ""
            self._loaded_caption = True
        return self.caption

    def save_caption(self, text):
        self.caption = text
        try:
            with open(self.text_path, 'w', encoding='utf-8') as f:
                f.write(text)
        except Exception: pass

class LoaderThread(QThread):
    finished = pyqtSignal(list)
    def __init__(self, directory):
        super().__init__()
        self.directory = directory
    def run(self):
        exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
        items = []
        try:
            with os.scandir(self.directory) as it:
                for entry in it:
                    if entry.is_file() and entry.name.lower().endswith(exts):
                        img_path = entry.path
                        txt_path = os.path.splitext(img_path)[0] + ".txt"
                        if not os.path.exists(txt_path): open(txt_path, 'w', encoding='utf-8').close()
                        items.append(DatasetItem(img_path, txt_path))
        except Exception: pass
        items.sort(key=lambda x: x.filename.lower())
        self.finished.emit(items)

class DatasetModel(QAbstractListModel):
    def __init__(self, all_items=None):
        super().__init__()
        self.all_items = all_items or []
        self.filtered_items = self.all_items
        self.icon_mode = False

    def rowCount(self, parent=QModelIndex()): return len(self.filtered_items)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or index.row() >= len(self.filtered_items): return None
        item = self.filtered_items[index.row()]
        if role == Qt.ItemDataRole.DisplayRole: return "" if self.icon_mode else item.filename
        if role == Qt.ItemDataRole.DecorationRole and self.icon_mode:
            if item.icon is None:
                pm = QPixmap(item.image_path)
                if not pm.isNull(): item.icon = QIcon(pm.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation))
                else: item.icon = QIcon()
            return item.icon
        return None

    def update_items(self, new_items):
        self.beginResetModel(); self.all_items = new_items; self.filtered_items = new_items; self.endResetModel()

    def filter_items(self, search_text):
        self.beginResetModel()
        if not search_text: self.filtered_items = self.all_items
        else:
            tags = [t.strip().lower() for t in search_text.split(",") if t.strip()]
            self.filtered_items = [item for item in self.all_items if all(tag in item.filename.lower() or tag in item.load_caption().lower() for tag in tags)]
        self.endResetModel()
        
    def remove_item(self, item):
        if item in self.all_items: self.all_items.remove(item)
        if item in self.filtered_items:
            idx = self.filtered_items.index(item)
            self.beginRemoveRows(QModelIndex(), idx, idx); self.filtered_items.remove(item); self.endRemoveRows()

class DatasetEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("N-Force Dataset Editor")
        self.resize(1650, 950)
        self.current_directory = ""
        self.model = DatasetModel()
        self.filter_timer = QTimer(); self.filter_timer.setSingleShot(True); self.filter_timer.timeout.connect(self.apply_filter)
        self.completer = QCompleter()
        self.completer_loader = CompleterLoader()
        self.completer_loader.finished.connect(self.on_completer_loaded)
        self.completer_loader.start()
        self.use_gpu = False
        self.online_config = {
            "gemini_key": "",
            "gemini_model": "gemini-2.0-flash",
            "caption_prompt": "Describe this image in detail.",
            "tags_prompt": "Provide a list of comma-separated tags for this image."
        }
        self.load_online_config()
        self.init_ui()
        self.apply_styles()

    def init_ui(self):
        central_widget = QWidget(); self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        toolbar = QToolBar(); self.addToolBar(toolbar)
        toolbar.addAction(QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon), "Open", self, triggered=self.open_folder))
        toolbar.addSeparator()
        self.grid_action = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogContentsView), "Grid", self, checkable=True)
        self.grid_action.triggered.connect(self.toggle_grid_view)
        toolbar.addAction(self.grid_action)
        self.grid_size_slider = QSlider(Qt.Orientation.Horizontal); self.grid_size_slider.setRange(100, 600); self.grid_size_slider.setValue(250); self.grid_size_slider.setFixedWidth(80)
        self.grid_size_slider.valueChanged.connect(self.update_grid_density); self.grid_size_slider.setEnabled(False)
        toolbar.addWidget(self.grid_size_slider)
        toolbar.addSeparator()
        toolbar.addAction(QAction("Prefix", self, triggered=self.bulk_prefix))
        toolbar.addAction(QAction("Replace", self, triggered=self.bulk_replace_dialog))
        toolbar.addAction(QAction("Delete Tag (Global)", self, triggered=self.delete_tag_global))
        toolbar.addAction(QAction("Rename", self, triggered=self.rename_all_images))
        toolbar.addAction(QAction("Tags Current IMG", self, triggered=self.tag_current_image))
        toolbar.addSeparator()
        toolbar.addAction(QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon), "Del Item", self, triggered=self.delete_current_item))
        toolbar.addSeparator()
        self.gpu_btn = QPushButton("CPU Mode")
        self.gpu_btn.setObjectName("GPU_BTN")
        self.gpu_btn.setCheckable(True)
        self.gpu_btn.setFixedWidth(100)
        self.gpu_btn.clicked.connect(self.toggle_gpu)
        toolbar.addWidget(self.gpu_btn)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        list_container = QWidget(); list_layout = QVBoxLayout(list_container)
        self.search_bar = QLineEdit(); self.search_bar.setPlaceholderText("Filter name/tags...")
        self.search_bar.textChanged.connect(self.restart_filter_timer)
        self.list_view = QListView(); self.list_view.setModel(self.model); self.list_view.clicked.connect(self.item_selected)
        self.list_view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        nav_layout = QHBoxLayout(); self.index_label = QLabel("0 / 0"); self.nav_slider = QSlider(Qt.Orientation.Horizontal)
        self.nav_slider.valueChanged.connect(self.on_slider_nav)
        nav_layout.addWidget(self.index_label); nav_layout.addWidget(self.nav_slider)
        list_layout.addWidget(self.search_bar); list_layout.addWidget(self.list_view); list_layout.addLayout(nav_layout)
        
        editor_container = QSplitter(Qt.Orientation.Vertical)
# Image Viewer
        self.image_container = QWidget(); img_layout = QVBoxLayout(self.image_container)
        self.image_view = ZoomableGraphicsView()
        self.image_view.colorPicked.connect(self.add_color_tag)
        img_layout.addWidget(self.image_view); self.image_container.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333;")
        
        text_edit_container = QWidget(); text_layout = QVBoxLayout(text_edit_container); tools_layout = QHBoxLayout()
        tools_layout.addWidget(QPushButton("Aa", clicked=self.transform_case))
        tools_layout.addWidget(QPushButton("Remove _", clicked=self.underscore_to_space))
        tools_layout.addWidget(QPushButton("Fix ,", clicked=self.fix_commas))
        self.mode_btn = QPushButton("Tags Mode"); self.mode_btn.setCheckable(True); self.mode_btn.setChecked(True); self.mode_btn.setFixedWidth(120); self.mode_btn.clicked.connect(self.toggle_editor_mode)
        tools_layout.addWidget(self.mode_btn)
        self.highlight_bar = QLineEdit(); self.highlight_bar.setPlaceholderText("Highlight tags..."); self.highlight_bar.textChanged.connect(self.update_highlight)
        tools_layout.addWidget(self.highlight_bar)
        self.text_edit = TagTextEdit(); self.text_edit.textChanged.connect(self.text_modified); self.text_edit.setCompleter(self.completer); self.highlighter = TagHighlighter(self.text_edit.document())
        text_layout.addLayout(tools_layout); text_layout.addWidget(self.text_edit)
        editor_container.addWidget(self.image_container); editor_container.addWidget(text_edit_container)
        
        self.right_panel = QFrame(); self.right_panel.setFixedWidth(320); self.right_panel.setObjectName("RightPanel")
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(8, 8, 8, 8)

        # ─── Top 10 Tags Panel ───
        top10_header = QHBoxLayout()
        top10_label = QLabel("<b>Top 10 Tags</b>")
        top10_label.setStyleSheet("color: #0cc;")
        self.clear_filter_btn = QPushButton("✕ Clear")
        self.clear_filter_btn.setFixedHeight(22)
        self.clear_filter_btn.setStyleSheet("""
            QPushButton { background: #333333; color: #ff6b6b; border: 1px solid #ff6b6b; 
            padding: 2px 8px; border-radius: 3px; font-size: 11px; font-weight: bold; }
            QPushButton:hover { background: #ff6b6b; color: #fff; }
        """)
        self.clear_filter_btn.clicked.connect(self.clear_tag_filter)
        self.clear_filter_btn.setVisible(False)
        top10_header.addWidget(top10_label)
        top10_header.addStretch()
        top10_header.addWidget(self.clear_filter_btn)
        right_layout.addLayout(top10_header)

        self.top10_container = QWidget()
        self.top10_flow = FlowLayout(self.top10_container, margin=2, spacing=4)
        right_layout.addWidget(self.top10_container)

        # ─── All Tags Panel ───
        sep_tags = QFrame()
        sep_tags.setFrameShape(QFrame.Shape.HLine)
        sep_tags.setStyleSheet("background-color: #444;")
        sep_tags.setFixedHeight(1)
        right_layout.addWidget(sep_tags)

        all_tags_header = QHBoxLayout()
        all_tags_label = QLabel("<b>All Tags</b>")
        all_tags_label.setStyleSheet("color: #888;")
        self.all_tags_count_label = QLabel("(0)")
        self.all_tags_count_label.setStyleSheet("color: #666; font-size: 11px;")
        all_tags_header.addWidget(all_tags_label)
        all_tags_header.addWidget(self.all_tags_count_label)
        all_tags_header.addStretch()
        right_layout.addLayout(all_tags_header)

        self.all_tags_scroll = QScrollArea()
        self.all_tags_scroll.setWidgetResizable(True)
        self.all_tags_scroll.setFixedHeight(250)
        self.all_tags_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.all_tags_scroll.setStyleSheet("""
            QScrollArea { border: 1px solid #333; background: #1a1a1a; border-radius: 4px; }
            QScrollBar:vertical { background: #1a1a1a; width: 8px; border: none; }
            QScrollBar::handle:vertical { background: #555; border-radius: 4px; min-height: 20px; }
            QScrollBar::handle:vertical:hover { background: #777; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        """)
        self.all_tags_container = QWidget()
        self.all_tags_container.setStyleSheet("background: #1a1a1a;")
        self.all_tags_flow = FlowLayout(self.all_tags_container, margin=4, spacing=3)
        self.all_tags_scroll.setWidget(self.all_tags_container)
        right_layout.addWidget(self.all_tags_scroll)

        # ─── Batch Operations Container ───
        self.batch_container = QWidget()
        self.batch_container.setObjectName("BatchContainer")
        self.batch_container.setStyleSheet("#BatchContainer { background-color: #252526; }")
        batch_v_layout = QVBoxLayout(self.batch_container)
        batch_v_layout.setContentsMargins(0, 10, 0, 0)
        batch_v_layout.setSpacing(8)

        sep_batch = QFrame()
        sep_batch.setFrameShape(QFrame.Shape.HLine)
        sep_batch.setStyleSheet("background-color: #34AAFB;")
        sep_batch.setFixedHeight(1)
        batch_v_layout.addWidget(sep_batch)

        batch_label = QLabel("<b>Batch Operations</b>")
        batch_label.setStyleSheet("color: #34AAFB; font-size: 14px; padding: 4px 0px; background: transparent;")
        batch_v_layout.addWidget(batch_label)
        
        batch_grid = QVBoxLayout()
        row1 = QHBoxLayout(); row1.setSpacing(4)
        row2 = QHBoxLayout(); row2.setSpacing(4)
        row3 = QHBoxLayout(); row3.setSpacing(4)

        self.btn_batch_tags = QPushButton("Tags all")
        self.btn_convert_img = QPushButton("Convert all IMG")
        self.btn_scale_img = QPushButton("Scale Img by %")
        self.btn_img_metadata = QPushButton("IMG Metadata")
        self.btn_upscale_ai = QPushButton("Upscale AI")
        self.btn_clean_up = QPushButton("Clean Up")

        row1.addWidget(self.btn_batch_tags)
        row1.addWidget(self.btn_convert_img)
        row2.addWidget(self.btn_scale_img)
        row2.addWidget(self.btn_img_metadata)
        row3.addWidget(self.btn_upscale_ai)
        row3.addWidget(self.btn_clean_up)
        
        batch_v_layout.addLayout(row1)
        batch_v_layout.addLayout(row2)
        batch_v_layout.addLayout(row3)
        
        right_layout.addWidget(self.batch_container)

        self.btn_batch_tags.clicked.connect(self.show_offline_tagger_dialog)
        self.btn_convert_img.clicked.connect(self.show_convert_dialog)
        self.btn_scale_img.clicked.connect(self.show_resize_dialog)
        self.btn_img_metadata.clicked.connect(self.show_metadata_dialog)
        self.btn_upscale_ai.clicked.connect(self.show_upscale_dialog)
        self.btn_clean_up.clicked.connect(self.show_cleanup_dialog)

        # Online Mode Section
        right_layout.addSpacing(20)
        online_header = QHBoxLayout()
        online_title = QLabel("<b>Online Mode</b>")
        online_title.setStyleSheet("color: #007acc;")
        self.btn_online_settings = QPushButton("Setting")
        self.btn_online_settings.setFlat(True)
        self.btn_online_settings.setStyleSheet("color: #007acc; text-decoration: underline;")
        self.btn_online_settings.clicked.connect(self.show_online_settings)
        online_header.addWidget(online_title)
        online_header.addStretch()
        online_header.addWidget(self.btn_online_settings)
        right_layout.addLayout(online_header)

        # Online Selection Mode
        selection_layout = QHBoxLayout()
        self.online_all_cb = QCheckBox("Apply to All")
        selection_layout.addWidget(self.online_all_cb)
        right_layout.addLayout(selection_layout)

        # Service Buttons Grid
        service_grid = QVBoxLayout()
        row1 = QHBoxLayout()
        self.btn_gemini_tags = QPushButton("Gemini Tags")
        self.btn_wd_tagger = QPushButton("WD Tagger")
        row1.addWidget(self.btn_gemini_tags)
        row1.addWidget(self.btn_wd_tagger)
        
        row2 = QHBoxLayout()
        self.btn_gemini_caption = QPushButton("Gemini Caption")
        self.btn_joy_tag = QPushButton("Joy Tag")
        row2.addWidget(self.btn_gemini_caption)
        row2.addWidget(self.btn_joy_tag)

        self.btn_joy_caption = QPushButton("Joy Caption")
        
        service_grid.addLayout(row1)
        service_grid.addLayout(row2)
        service_grid.addWidget(self.btn_joy_caption)
        right_layout.addLayout(service_grid)

        # Connect Online Buttons
        self.btn_gemini_tags.clicked.connect(lambda: self.run_online_service("gemini_tags"))
        self.btn_gemini_caption.clicked.connect(lambda: self.run_online_service("gemini_caption"))
        self.btn_wd_tagger.clicked.connect(lambda: self.run_online_service("wd_tagger"))
        self.btn_joy_tag.clicked.connect(lambda: self.run_online_service("joy_tag"))
        self.btn_joy_caption.clicked.connect(lambda: self.run_online_service("joy_caption"))

        # Progress / Log Area
        right_layout.addStretch()
        self.online_log = QLabel("Ready")
        self.online_log.setStyleSheet("font-size: 10px; color: #888;")
        self.online_progress = QProgressBar()
        self.online_progress.setFixedHeight(8)
        self.online_progress.setStyleSheet("QProgressBar { background: #111; border: none; border-radius: 4px; } QProgressBar::chunk { background: #007acc; }")
        self.online_progress.setValue(0)
        
        right_layout.addWidget(self.online_log)
        right_layout.addWidget(self.online_progress)

        self.main_splitter.addWidget(list_container); self.main_splitter.addWidget(editor_container); self.main_splitter.addWidget(self.right_panel)
        self.main_splitter.setStretchFactor(0, 1); self.main_splitter.setStretchFactor(1, 4)
        main_layout.addWidget(self.main_splitter)
        self.setStatusBar(QStatusBar())

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; color: #d4d4d4; }
            #RightPanel { background-color: #252526; border-left: 1px solid #333; }
            QListView { background-color: #252526; border: none; color: #cccccc; font-size: 13px; outline: none; }
            QListView::item:selected { background-color: #37373d; color: #ffffff; border-left: 3px solid #007acc; }
            QTextEdit { background-color: #1e1e1e; color: #d4d4d4; border: 1px solid #3c3c3c; font-family: 'Segoe UI'; font-size: 15px; padding: 12px; }
            QLineEdit { background-color: #1a1a1a; color: #cccccc; border: 1px solid #3c3c3c; padding: 5px; }
            QPushButton { background-color: #333333; color: #cccccc; border: 1px solid #444; padding: 6px; }
            QPushButton:hover { background-color: #444444; }
            QPushButton:checked { background-color: #007acc; color: white; border: 1px solid #005fa3; }
            QPushButton#GPU_BTN:checked { background-color: #007acc; color: white; border: 1px solid #005fa3; font-weight: bold; }
            QToolBar { background-color: #333333; border-bottom: 1px solid #252526; }
            QStatusBar { background-color: #007acc; color: white; }
        """)

    def on_completer_loaded(self, tags):
        if tags:
            model = QStringListModel(tags)
            self.completer.setModel(model)
            self.statusBar().showMessage(f"Auto-complete loaded: {len(tags)} tags", 3000)

    def toggle_editor_mode(self, checked):
        if checked:
            self.mode_btn.setText("Tags Mode"); self.text_edit.tags_mode = True
            self.highlight_bar.setStyleSheet("background-color: #1a1a1a; border: 1px solid #007acc; padding: 4px;")
            self.statusBar().showMessage("Tags Mode: Auto-complete enabled.", 2000)
        else:
            self.mode_btn.setText("Caption Mode"); self.text_edit.tags_mode = False
            self.highlight_bar.setStyleSheet("background-color: #1a1a1a; border: 1px solid #3c3c3c; padding: 4px;")
            self.statusBar().showMessage("Caption Mode: Auto-complete disabled.", 2000)

    def open_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if path:
            self.current_directory = path
            self.loader = LoaderThread(path); self.loader.finished.connect(self.on_load_finished); self.loader.start()

    def on_load_finished(self, items):
        self.model.update_items(items); self.update_nav_slider()
        if items: self.list_view.setCurrentIndex(self.model.index(0,0)); self.item_selected(self.model.index(0,0)); QTimer.singleShot(500, self.refresh_stats)

    def refresh_stats(self):
        if not self.model.all_items: return
        all_tags = []
        for item in self.model.all_items:
            tags = [t.strip().lower() for t in item.load_caption().split(",") if t.strip()]
            all_tags.extend(tags)
        all_counts = Counter(all_tags).most_common()
        top10 = all_counts[:10]

        # ─── Populate Top 10 Flow ───
        self.top10_flow.clear_widgets()
        for tag, count in top10:
            btn = QPushButton(f"{tag} ({count})")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet("""
                QPushButton { background: #007acc; color: #ffffff; border: 1px solid #005fa3; 
                padding: 4px 10px; border-radius: 12px; font-size: 11px; font-weight: bold; }
                QPushButton:hover { background: #34AAFB; color: #ffffff; border-color: #34AAFB; }
            """)
            btn.clicked.connect(lambda checked, t=tag: self.filter_by_tag(t))
            self.top10_flow.addWidget(btn)

        # ─── Populate All Tags Flow ───
        self.all_tags_flow.clear_widgets()
        if len(self.model.all_items) > 150:
            self.all_tags_count_label.setText("(Disabled)")
            disabled_lbl = QLabel("You have To many images and tags to display")
            disabled_lbl.setStyleSheet("color: #888; font-style: italic; padding: 10px;")
            self.all_tags_flow.addWidget(disabled_lbl)
        else:
            self.all_tags_count_label.setText(f"({len(all_counts)})")
            for tag, count in all_counts:
                btn = QPushButton(f"{tag} ({count})")
                btn.setCursor(Qt.CursorShape.PointingHandCursor)
                btn.setStyleSheet("""
                    QPushButton { background: #2a2a2a; color: #aaa; border: 1px solid #444; 
                    padding: 1px 3px; border-radius: 5px; font-size: 10px; }
                    QPushButton:hover { background: #3a3a3a; color: #ffffff; border-color: #888; }
                """)
                btn.clicked.connect(lambda checked, t=tag: self.filter_by_tag(t))
                self.all_tags_flow.addWidget(btn)

    def filter_by_tag(self, tag):
        """Filter the image list to show only images with the given tag."""
        self.search_bar.setText(tag)
        self.clear_filter_btn.setVisible(True)
        self.statusBar().showMessage(f"Filtering by tag: '{tag}'", 3000)

    def clear_tag_filter(self):
        """Clear the tag filter and show all images."""
        self.search_bar.clear()
        self.clear_filter_btn.setVisible(False)
        self.statusBar().showMessage("Filter cleared.", 2000)

    def add_color_tag(self, color_hex):
        """Add a color hex code as a tag to the current image's text."""
        idx = self.list_view.currentIndex()
        if not idx.isValid(): return
        
        current_text = self.text_edit.toPlainText().strip()
        if color_hex in current_text:
            self.statusBar().showMessage(f"Tag '{color_hex}' already exists.", 2000)
            return
            
        if current_text:
            if not current_text.endswith(","):
                current_text += ", "
            else:
                current_text += " "
        
        new_text = current_text + color_hex
        self.text_edit.setPlainText(new_text)
        self.statusBar().showMessage(f"Added color tag: {color_hex}", 3000)

    def toggle_grid_view(self, checked):
        self.grid_size_slider.setEnabled(checked)
        if checked:
            self.list_view.setViewMode(QListView.ViewMode.IconMode); self.list_view.setResizeMode(QListView.ResizeMode.Adjust)
            self.model.icon_mode = True; avail_width = self.list_view.width() - 30; self.grid_size_slider.setValue(avail_width // 4 - 20); self.update_grid_density(self.grid_size_slider.value())
        else:
            self.list_view.setViewMode(QListView.ViewMode.ListMode); self.model.icon_mode = False
        self.model.layoutChanged.emit()

    def update_grid_density(self, v):
        if self.model.icon_mode: self.list_view.setIconSize(QSize(v, v)); self.list_view.setGridSize(QSize(v + 10, v + 20))

    def on_slider_nav(self, v):
        if self.model.rowCount() > v: idx = self.model.index(v, 0); self.list_view.setCurrentIndex(idx); self.item_selected(idx); self.list_view.scrollTo(idx)

    def update_nav_slider(self):
        c = self.model.rowCount(); self.nav_slider.setRange(0, max(0, c - 1)); self.index_label.setText(f"0 / {c}")

    def item_selected(self, index):
        if not index.isValid(): return
        item = self.model.filtered_items[index.row()]
        self.nav_slider.blockSignals(True); self.nav_slider.setValue(index.row()); self.nav_slider.blockSignals(False); self.index_label.setText(f"{index.row() + 1} / {self.model.rowCount()}")
        pm = QPixmap(item.image_path)
        if not pm.isNull():
            self.image_view.setPixmap(pm)
        
        self.text_edit.blockSignals(True); self.text_edit.setPlainText(item.load_caption()); self.text_edit.blockSignals(False)

    def text_modified(self):
        idx = self.list_view.currentIndex()
        if idx.isValid(): 
            self.model.filtered_items[idx.row()].save_caption(self.text_edit.toPlainText())
            if not hasattr(self, '_stat_timer'): self._stat_timer = QTimer(); self._stat_timer.setSingleShot(True); self._stat_timer.timeout.connect(self.refresh_stats)
            self._stat_timer.start(5000)

    def restart_filter_timer(self): self.filter_timer.start(300)
    def apply_filter(self): self.model.filter_items(self.search_bar.text()); self.update_nav_slider()
    def update_highlight(self): self.highlighter.set_tags(self.highlight_bar.text())
    def transform_case(self): t = self.text_edit.toPlainText(); self.text_edit.setPlainText(t.lower() if t.isupper() else t.upper())
    def underscore_to_space(self): self.text_edit.setPlainText(self.text_edit.toPlainText().replace("_", " "))
    def fix_commas(self): self.text_edit.setPlainText(re.sub(r'\s*,\s*', ', ', self.text_edit.toPlainText()).strip())
    
    def bulk_replace_dialog(self):
        f, ok = QInputDialog.getText(self, "Replace", "Find:")
        if ok and f:
            r, ok = QInputDialog.getText(self, "Replace", f"Replace '{f}' with:")
            if ok:
                for item in self.model.all_items:
                    c = item.load_caption()
                    if f in c: item.save_caption(c.replace(f, r))
                self.refresh_stats()

    def delete_tag_global(self):
        if not self.model.all_items: return
        tag, ok = QInputDialog.getText(self, "Delete Tag (Global)", "Delete Tags From All IMGS!!:")
        if ok and tag:
            tag = tag.strip()
            count = 0
            for item in self.model.all_items:
                content = item.load_caption()
                tags = [t.strip() for t in content.split(",") if t.strip()]
                if tag in tags:
                    tags.remove(tag)
                    item.save_caption(", ".join(tags))
                    count += 1
            self.statusBar().showMessage(f"Removed '{tag}' from {count} images.", 3000)
            self.model.layoutChanged.emit()
            idx = self.list_view.currentIndex()
            if idx.isValid(): self.item_selected(idx)
            self.refresh_stats()

    def targeted_delete_text(self):
        d, ok = QInputDialog.getText(self, "Del", "Tag to remove:")
        if ok and d:
            if QMessageBox.question(self, 'Confirm', f"Del '{d}'?") == QMessageBox.StandardButton.Yes:
                for item in self.model.all_items:
                    c = item.load_caption()
                    if d in c: item.save_caption(c.replace(d, "").replace("  ", " ").strip())
                self.refresh_stats()

    def bulk_prefix(self):
        p, ok = QInputDialog.getText(self, "Prefix", "Add to all:")
        if ok and p:
            for item in self.model.all_items: item.save_caption(p + item.load_caption())
            self.refresh_stats()

    def rename_all_images(self):
        if not self.model.all_items: return
        n, ok = QInputDialog.getText(self, "Rename", "Name:")
        if ok and n:
            if QMessageBox.question(self, 'Confirm', "Rename ALL?") == QMessageBox.StandardButton.Yes:
                for i, item in enumerate(self.model.all_items, 1):
                    ni, nt = os.path.join(self.current_directory, f"{n}_{i}{os.path.splitext(item.image_path)[1]}"), os.path.join(self.current_directory, f"{n}_{i}.txt")
                    try: os.rename(item.image_path, ni); os.rename(item.text_path, nt); item.image_path, item.text_path, item.filename = ni, nt, os.path.basename(ni)
                    except Exception: pass
                self.model.layoutChanged.emit()

    def toggle_gpu(self, checked):
        self.use_gpu = checked
        self.gpu_btn.setText("GPU Mode" if checked else "CPU Mode")
        self.statusBar().showMessage(f"AI Provider switched to {'GPU' if checked else 'CPU'}", 2000)

    def delete_current_item(self):
        idx = self.list_view.currentIndex()
        if not idx.isValid(): return
        item = self.model.filtered_items[idx.row()]
        if QMessageBox.question(self, 'Delete', f"Delete {item.filename}?") == QMessageBox.StandardButton.Yes:
            try: os.remove(item.image_path); os.remove(item.text_path); self.model.remove_item(item); self.update_nav_slider(); self.refresh_stats()
            except Exception: pass

    def show_offline_tagger_dialog(self):
        if not self.model.all_items:
            QMessageBox.warning(self, "No Items", "No items to tag.")
            return
        dialog = OfflineTaggerDialog(self, self.model.all_items, use_gpu=self.use_gpu)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            idx = self.list_view.currentIndex()
            if idx.isValid(): self.item_selected(idx)
            self.refresh_stats()

    def tag_current_image(self):
        idx = self.list_view.currentIndex()
        if not idx.isValid():
            QMessageBox.warning(self, "No Selection", "Please select an image first.")
            return
        item = self.model.filtered_items[idx.row()]
        dialog = OfflineTaggerDialog(self, [item], use_gpu=self.use_gpu)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.item_selected(idx)
            self.refresh_stats()

    def show_convert_dialog(self):

        if not self.model.all_items:
            QMessageBox.warning(self, "No Items", "No items in the list to convert.")
            return
        dialog = ConvertDialog(self, self.model.all_items)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            if self.current_directory:
                self.loader = LoaderThread(self.current_directory)
                self.loader.finished.connect(self.on_load_finished)
                self.loader.start()

    def show_resize_dialog(self):
        if not self.model.all_items:
            QMessageBox.warning(self, "No Items", "No items to scale.")
            return
        dialog = ResizeDialog(self, self.model.all_items)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            idx = self.list_view.currentIndex()
            if idx.isValid(): self.item_selected(idx)

    def show_metadata_dialog(self):
        idx = self.list_view.currentIndex()
        if not idx.isValid():
            QMessageBox.warning(self, "No Selection", "Please select an image first.")
            return
        item = self.model.filtered_items[idx.row()]
        dialog = MetadataDialog(self, item.image_path)
        dialog.exec()

    def show_upscale_dialog(self):
        if not self.model.all_items:
            QMessageBox.warning(self, "No Items", "No items to upscale.")
            return
        idx = self.list_view.currentIndex().row() if self.list_view.currentIndex().isValid() else 0
        dialog = UpscaleDialog(self, self.model.all_items, idx, use_gpu=self.use_gpu)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            current_idx = self.list_view.currentIndex()
            if current_idx.isValid(): self.item_selected(current_idx)

    def show_cleanup_dialog(self):
        if not self.model.all_items:
            QMessageBox.warning(self, "No Items", "No items to clean up.")
            return
        dialog = CleanUpDialog(self, self.model.all_items, self.current_directory)
        dialog.exec()

    def load_online_config(self):
        if os.path.exists("online_config.json"):
            try:
                with open("online_config.json", "r") as f:
                    self.online_config.update(json.load(f))
            except Exception: pass

    def save_online_config(self):
        try:
            with open("online_config.json", "w") as f:
                json.dump(self.online_config, f)
        except Exception: pass

    def show_online_settings(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Online Settings")
        dialog.setFixedWidth(400)
        layout = QVBoxLayout(dialog)
        
        layout.addWidget(QLabel("Gemini API Key:"))
        key_edit = QLineEdit(self.online_config["gemini_key"])
        layout.addWidget(key_edit)

        layout.addWidget(QLabel("Gemini Model:"))
        model_combo = QComboBox()
        gemini_models = [
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-2.5-flash-preview-04-17",
            "gemini-2.5-pro-preview-03-25",
        ]
        model_combo.addItems(gemini_models)
        current_model = self.online_config.get("gemini_model", "gemini-2.0-flash")
        idx = model_combo.findText(current_model)
        if idx >= 0: model_combo.setCurrentIndex(idx)
        layout.addWidget(model_combo)
        
        layout.addWidget(QLabel("Caption Prompt:"))
        cap_edit = QTextEdit(self.online_config["caption_prompt"])
        cap_edit.setFixedHeight(100)
        layout.addWidget(cap_edit)
        
        layout.addWidget(QLabel("Tags Prompt:"))
        tag_edit = QTextEdit(self.online_config["tags_prompt"])
        tag_edit.setFixedHeight(100)
        layout.addWidget(tag_edit)
        
        save_btn = QPushButton("Save Settings", clicked=lambda: self.update_online_config(
            dialog, key_edit.text(), model_combo.currentText(),
            cap_edit.toPlainText(), tag_edit.toPlainText()
        ))
        layout.addWidget(save_btn)
        dialog.exec()

    def update_online_config(self, dialog, key, model, cap, tag):
        self.online_config.update({
            "gemini_key": key, "gemini_model": model,
            "caption_prompt": cap, "tags_prompt": tag
        })
        self.save_online_config()
        dialog.accept()

    def run_online_service(self, service_type):
        if not self.model.all_items: return
        
        items = self.model.all_items if self.online_all_cb.isChecked() else []
        if not items:
            idx = self.list_view.currentIndex()
            if idx.isValid(): items = [self.model.filtered_items[idx.row()]]
            else: return

        self.online_progress.setValue(0)
        self.online_worker = OnlineWorker(items, service_type, self.online_config)
        self.online_worker.progress.connect(self.online_progress.setValue)
        self.online_worker.log.connect(self.online_log.setText)
        self.online_worker.finished.connect(self.on_online_finished)
        self.online_worker.start()
        
        self.set_online_btns_enabled(False)

    def on_online_finished(self, count):
        self.online_log.setText(f"Finished: {count} images processed.")
        self.set_online_btns_enabled(True)
        idx = self.list_view.currentIndex()
        if idx.isValid(): self.item_selected(idx)
        self.refresh_stats()

    def set_online_btns_enabled(self, enabled):
        self.btn_gemini_tags.setEnabled(enabled)
        self.btn_gemini_caption.setEnabled(enabled)
        self.btn_wd_tagger.setEnabled(enabled)
        self.btn_joy_tag.setEnabled(enabled)
        self.btn_joy_caption.setEnabled(enabled)

if __name__ == "__main__":



    app = QApplication(sys.argv)
    window = DatasetEditor()
    window.show()
    sys.exit(app.exec())
