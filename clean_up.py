import os
import csv
from collections import Counter
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QPushButton, QMessageBox, QLabel, 
    QCheckBox, QFileDialog, QHBoxLayout, QScrollArea, QWidget, QSpinBox
)
from PyQt6.QtCore import Qt
try:
    from PIL import Image
except ImportError:
    pass

class CleanUpDialog(QDialog):
    def __init__(self, parent, dataset_items, current_directory):
        super().__init__(parent)
        self.setWindowTitle("Clean Up Dataset")
        self.setFixedWidth(400)
        self.dataset_items = dataset_items
        self.current_directory = current_directory
        self.parent_editor = parent

        layout = QVBoxLayout(self)
        
        desc = QLabel("<b>Batch Clean Up & Export</b>")
        desc.setStyleSheet("color: #34AAFB; font-size: 14px; margin-bottom: 10px;")
        layout.addWidget(desc)

        # Basic cleanup buttons
        self.btn_remove_duplicates = QPushButton("1. Remove Duplicate Words")
        self.btn_sort_tags = QPushButton("2. Transform Word (Sort A-Z)")
        self.btn_title_case = QPushButton("3. Capitalize Each Word (Title Case)")
        self.btn_delete_error_img = QPushButton("4. Delete Error/Corrupted Images")
        self.btn_strip_spaces = QPushButton("5. Clean Spacing & Empty Tags")
        
        # New advanced buttons
        self.btn_purge_rare = QPushButton("6. Low Frequency Purge (Rare Tags)")
        self.btn_export_csv = QPushButton("7. Convert to LLM (Export CSV)")

        layout.addWidget(self.btn_remove_duplicates)
        layout.addWidget(self.btn_sort_tags)
        layout.addWidget(self.btn_title_case)
        layout.addWidget(self.btn_delete_error_img)
        layout.addWidget(self.btn_strip_spaces)
        layout.addWidget(self.btn_purge_rare)
        layout.addWidget(self.btn_export_csv)

        self.btn_remove_duplicates.clicked.connect(self.remove_duplicates)
        self.btn_sort_tags.clicked.connect(self.sort_tags)
        self.btn_title_case.clicked.connect(self.title_case)
        self.btn_delete_error_img.clicked.connect(self.delete_error_img)
        self.btn_strip_spaces.clicked.connect(self.strip_spaces)
        self.btn_purge_rare.clicked.connect(self.low_frequency_purge)
        self.btn_export_csv.clicked.connect(self.export_to_csv)

        self.apply_styles()

    def apply_styles(self):
        self.setStyleSheet("""
            QDialog { background-color: #1e1e1e; color: #d4d4d4; }
            QPushButton { background-color: #333333; color: #cccccc; border: 1px solid #444; padding: 8px; border-radius: 4px; text-align: left; }
            QPushButton:hover { background-color: #444444; }
        """)

    def remove_duplicates(self):
        count = 0
        for item in self.dataset_items:
            content = item.load_caption()
            if not content: continue
            tags = [t.strip() for t in content.split(",") if t.strip()]
            seen = set()
            new_tags = []
            for t in tags:
                lower_t = t.lower()
                if lower_t not in seen:
                    seen.add(lower_t)
                    new_tags.append(t)
            new_content = ", ".join(new_tags)
            if new_content != content:
                item.save_caption(new_content)
                count += 1
        QMessageBox.information(self, "Success", f"Removed duplicates in {count} files.")
        if self.parent_editor: self.parent_editor.refresh_stats()

    def sort_tags(self):
        count = 0
        for item in self.dataset_items:
            content = item.load_caption()
            if not content: continue
            tags = [t.strip() for t in content.split(",") if t.strip()]
            sorted_tags = sorted(tags, key=lambda x: x.lower())
            new_content = ", ".join(sorted_tags)
            if new_content != content:
                item.save_caption(new_content)
                count += 1
        QMessageBox.information(self, "Success", f"Sorted tags in {count} files.")
        if self.parent_editor: self.parent_editor.refresh_stats()

    def title_case(self):
        count = 0
        for item in self.dataset_items:
            content = item.load_caption()
            if not content: continue
            tags = [t.strip() for t in content.split(",") if t.strip()]
            new_tags = [" ".join(word.capitalize() for word in t.split()) for t in tags]
            new_content = ", ".join(new_tags)
            if new_content != content:
                item.save_caption(new_content)
                count += 1
        QMessageBox.information(self, "Success", f"Capitalized words in {count} files.")
        if self.parent_editor: self.parent_editor.refresh_stats()

    def delete_error_img(self):
        deleted = 0
        to_remove = []
        for item in self.dataset_items:
            corrupted = False
            try:
                size = os.path.getsize(item.image_path)
                if size == 0:
                    corrupted = True
                else:
                    try:
                        with Image.open(item.image_path) as img:
                            img.verify()
                    except NameError:
                        pass
            except Exception:
                corrupted = True

            if corrupted:
                to_remove.append(item)
                
        if not to_remove:
            QMessageBox.information(self, "Result", "No corrupted or 0KB images found.")
            return

        reply = QMessageBox.question(self, "Confirm Deletion", 
                                     f"Found {len(to_remove)} corrupted/0KB images. Delete them and their associated text files?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            for item in to_remove:
                try:
                    if os.path.exists(item.image_path): os.remove(item.image_path)
                    if os.path.exists(item.text_path): os.remove(item.text_path)
                    if self.parent_editor:
                        self.parent_editor.model.remove_item(item)
                    deleted += 1
                except Exception as e:
                    print(f"Failed to delete {item.image_path}: {e}")
            if self.parent_editor:
                self.parent_editor.update_nav_slider()
                self.parent_editor.refresh_stats()
            QMessageBox.information(self, "Success", f"Deleted {deleted} error images.")

    def strip_spaces(self):
        count = 0
        for item in self.dataset_items:
            content = item.load_caption()
            if not content: continue
            tags = [t.strip() for t in content.split(",") if t.strip()]
            new_content = ", ".join(tags)
            if new_content != content:
                item.save_caption(new_content)
                count += 1
        QMessageBox.information(self, "Success", f"Cleaned spacing in {count} files.")
        if self.parent_editor: self.parent_editor.refresh_stats()

    def low_frequency_purge(self):
        # Count all tags
        all_tags = []
        for item in self.dataset_items:
            tags = [t.strip().lower() for t in item.load_caption().split(",") if t.strip()]
            all_tags.extend(tags)
        
        counts = Counter(all_tags)
        
        # Sub-dialog to select threshold and show tags
        purge_dialog = QDialog(self)
        purge_dialog.setWindowTitle("Rare Tag Purge")
        purge_dialog.setMinimumWidth(400)
        purge_layout = QVBoxLayout(purge_dialog)
        
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Threshold (remove tags with frequency <=):"))
        threshold_spin = QSpinBox()
        threshold_spin.setRange(1, 100)
        threshold_spin.setValue(1)
        h_layout.addWidget(threshold_spin)
        purge_layout.addLayout(h_layout)
        
        tags_scroll = QScrollArea()
        tags_scroll.setWidgetResizable(True)
        tags_content = QWidget()
        self.tags_vbox = QVBoxLayout(tags_content)
        tags_scroll.setWidget(tags_content)
        purge_layout.addWidget(tags_scroll)
        
        def refresh_rare_list():
            # Clear layout
            while self.tags_vbox.count():
                child = self.tags_vbox.takeAt(0)
                if child.widget(): child.widget().deleteLater()
            
            thresh = threshold_spin.value()
            rare_tags = [tag for tag, count in counts.items() if count <= thresh]
            rare_tags.sort()
            
            if not rare_tags:
                self.tags_vbox.addWidget(QLabel("No tags found below this threshold."))
            else:
                self.tags_vbox.addWidget(QLabel(f"Found {len(rare_tags)} rare tags:"))
                for tag in rare_tags:
                    self.tags_vbox.addWidget(QLabel(f"- {tag} ({counts[tag]})"))
            self.tags_vbox.addStretch()
            
        threshold_spin.valueChanged.connect(refresh_rare_list)
        refresh_rare_list()
        
        btn_confirm = QPushButton("Confirm Purge (Remove from all Files)")
        btn_confirm.setStyleSheet("background-color: #a30000; color: white; padding: 10px; font-weight: bold;")
        purge_layout.addWidget(btn_confirm)
        
        confirmed = False
        def on_confirm():
            nonlocal confirmed
            confirmed = True
            purge_dialog.accept()
            
        btn_confirm.clicked.connect(on_confirm)
        
        if purge_dialog.exec() == QDialog.DialogCode.Accepted and confirmed:
            thresh = threshold_spin.value()
            rare_tags_set = {tag for tag, count in counts.items() if count <= thresh}
            
            count_modified = 0
            for item in self.dataset_items:
                content = item.load_caption()
                tags = [t.strip() for t in content.split(",") if t.strip()]
                new_tags = [t for t in tags if t.lower() not in rare_tags_set]
                
                if len(new_tags) != len(tags):
                    item.save_caption(", ".join(new_tags))
                    count_modified += 1
            
            QMessageBox.information(self, "Success", f"Purged rare tags from {count_modified} files.")
            if self.parent_editor: self.parent_editor.refresh_stats()

    def export_to_csv(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save CSV for LLM", self.current_directory, "CSV Files (*.csv)")
        if not file_path:
            return
            
        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["file_name", "caption"]) # Standard header
                for item in self.dataset_items:
                    writer.writerow([item.filename, item.load_caption()])
            
            QMessageBox.information(self, "Success", f"Exported dataset to {os.path.basename(file_path)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export CSV: {e}")
