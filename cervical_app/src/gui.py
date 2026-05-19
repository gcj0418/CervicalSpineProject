"""
GUI for Cobb angle calculation using PyQt5.
Features: image display with zoom/pan, threaded inference, progress bar.
"""
import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QProgressBar, QFileDialog, QMessageBox,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QFrame, QSizePolicy, QStatusBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRectF
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon, QPalette, QColor, QPainter
import cv2
import numpy as np

from inference import SplitVLDModel
from clinical_parameters import (
    compute_c2c7_lordosis, compute_max_cobb, diagnose, draw_cobb,
    compute_c2c7_sva, compute_t1_slope, compute_disc_heights,
    compute_vertebral_displacement, compute_facet_joint_angles,
    draw_sva, draw_t1_slope, draw_disc_heights,
    draw_displacements, draw_facet_angles,
)


class InferenceWorker(QThread):
    """Run model inference in a background thread."""
    finished = pyqtSignal(object, object, object, object, object)  # pts_front, pts_back, orig_image, vis_image, spacing
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, split_model, image_path, device='cpu'):
        super().__init__()
        self.split_model = split_model
        self.image_path = image_path
        self.device = device

    def run(self):
        try:
            self.progress.emit(20, "正在加载图片...")
            self.progress.emit(50, "正在进行融合推理...")
            pts_front, pts_back, orig_image, spacing = self.split_model.predict(
                self.image_path
            )
            self.progress.emit(80, "正在绘制可视化...")
            vis = draw_cobb(orig_image, pts_front)
            self.progress.emit(100, "完成")
            self.finished.emit(pts_front, pts_back, orig_image, vis, spacing)
        except Exception as e:
            self.error.emit(str(e))


class ImageViewer(QGraphicsView):
    """Custom graphics view with wheel-zoom and right-drag pan."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._pixmap_item.setTransformationMode(Qt.SmoothTransformation)
        self._scene.addItem(self._pixmap_item)

        self.setRenderHints(
            self.renderHints()
            | QPainter.Antialiasing
            | QPainter.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setBackgroundBrush(QColor("#f0f0f0"))
        self.setFrameShape(QFrame.NoFrame)

        self._zoom = 1.0

    def set_image(self, qpixmap):
        self._pixmap_item.setPixmap(qpixmap)
        self._scene.setSceneRect(QRectF(qpixmap.rect()))
        self._zoom = 1.0
        self.resetTransform()
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        zoom_factor = 1.15 if delta > 0 else 0.87
        self._zoom *= zoom_factor
        self.scale(zoom_factor, zoom_factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
        super().mousePressEvent(event)


class CobbApp(QMainWindow):
    def __init__(self, split_model, device='cpu'):
        super().__init__()
        self.split_model = split_model
        self.device = device

        self.current_image = None
        self.current_pts = None
        self.vis_image = None
        self.current_spacing = None
        self.current_results = {}
        self.param_toggles = {}
        self.worker = None

        self.setWindowTitle("颈椎矢状位临床参数计算工具")
        self.setMinimumSize(1100, 900)
        self.resize(1200, 950)

        self._build_ui()
        self._apply_style()

    def _build_ui(self):
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # ---- Left: Image viewer ----
        self.viewer = ImageViewer()
        self.viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._show_placeholder()
        main_layout.addWidget(self.viewer, stretch=3)

        # ---- Right: Control panel ----
        panel = QWidget()
        panel.setMaximumWidth(380)
        panel.setMinimumWidth(300)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(12, 12, 12, 12)
        panel_layout.setSpacing(14)

        # Title
        title = QLabel("颈椎侧位 Cobb 角自动测量")
        title.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        panel_layout.addWidget(title)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_open = QPushButton("选择图片")
        self.btn_open.setMinimumHeight(40)
        self.btn_open.setFont(QFont("Microsoft YaHei", 11))
        self.btn_open.setCursor(Qt.PointingHandCursor)
        self.btn_open.clicked.connect(self.open_image)

        self.btn_save = QPushButton("保存结果")
        self.btn_save.setMinimumHeight(40)
        self.btn_save.setFont(QFont("Microsoft YaHei", 11))
        self.btn_save.setCursor(Qt.PointingHandCursor)
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save_result)

        btn_layout.addWidget(self.btn_open)
        btn_layout.addWidget(self.btn_save)
        panel_layout.addLayout(btn_layout)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("%p%  %v")
        self.progress.setMinimumHeight(22)
        panel_layout.addWidget(self.progress)

        self.status_label = QLabel("就绪")
        self.status_label.setFont(QFont("Microsoft YaHei", 10))
        self.status_label.setAlignment(Qt.AlignCenter)
        panel_layout.addWidget(self.status_label)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #cccccc;")
        panel_layout.addWidget(line)

        # Results card
        result_title = QLabel("测量结果")
        result_title.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        result_title.setAlignment(Qt.AlignCenter)
        panel_layout.addWidget(result_title)

        # Parameter grid: toggle button + value label
        self.param_toggles = {}
        grid = QGridLayout()
        grid.setSpacing(4)
        grid.setColumnStretch(1, 1)
        grid.setVerticalSpacing(6)

        def _make_toggle(key, text, row_idx, btn_width):
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setChecked(False)
            btn.setEnabled(False)
            btn.setMinimumHeight(28)
            btn.setFixedWidth(btn_width)
            btn.setFont(QFont("Microsoft YaHei", 10))
            btn.clicked.connect(self._update_result_display)
            lbl = QLabel("—")
            lbl.setFont(QFont("Microsoft YaHei", 11))
            lbl.setStyleSheet("color: #333333;")
            lbl.setWordWrap(True)
            grid.addWidget(btn, row_idx, 0)
            grid.addWidget(lbl, row_idx, 1)
            self.param_toggles[key] = {'btn': btn, 'lbl': lbl}

        def _add_sep(row_idx):
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setStyleSheet("color: #cccccc;")
            grid.addWidget(line, row_idx, 0, 1, 2)

        _make_toggle('cobb', 'Cobb', 0, 72)

        # Diagnosis placed right after Cobb
        self.diagnosis_label = QLabel("诊断: —")
        self.diagnosis_label.setFont(QFont("Microsoft YaHei", 12))
        self.diagnosis_label.setStyleSheet(
            "color: #1976d2; padding: 8px; background: #e3f2fd; border-radius: 6px;"
        )
        self.diagnosis_label.setAlignment(Qt.AlignCenter)
        grid.addWidget(self.diagnosis_label, 1, 0, 1, 2)

        _add_sep(2)
        _make_toggle('sva', 'SVA', 3, 72)
        _add_sep(4)
        _make_toggle('t1', 'T1', 5, 72)
        _add_sep(6)
        _make_toggle('disc', '椎间隙', 7, 72)
        _add_sep(8)
        _make_toggle('disp', '椎位移', 9, 72)
        _add_sep(10)
        _make_toggle('facet', '关节突', 11, 72)

        panel_layout.addLayout(grid)

        panel_layout.addStretch()

        # Tip
        tip = QLabel("提示：滚轮缩放图片，右键拖拽平移")
        tip.setFont(QFont("Microsoft YaHei", 9))
        tip.setStyleSheet("color: #888888;")
        tip.setAlignment(Qt.AlignCenter)
        panel_layout.addWidget(tip)

        main_layout.addWidget(panel, stretch=1)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("模型已加载 | 支持 PNG/JPG/NIfTI 格式")

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #fafafa;
            }
            QPushButton {
                background-color: #2196f3;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #1976d2;
            }
            QPushButton:checked {
                background-color: #4caf50;
            }
            QPushButton:checked:hover {
                background-color: #388e3c;
            }
            QPushButton:disabled {
                background-color: #bdbdbd;
                color: #eeeeee;
            }
            QPushButton#save_btn {
                background-color: #4caf50;
            }
            QPushButton#save_btn:hover {
                background-color: #388e3c;
            }
            QProgressBar {
                border: 1px solid #cccccc;
                border-radius: 4px;
                text-align: center;
                background: #ffffff;
            }
            QProgressBar::chunk {
                background-color: #2196f3;
                border-radius: 4px;
            }
            QLabel {
                color: #333333;
            }
        """)

    def _show_placeholder(self):
        placeholder = QPixmap(400, 300)
        placeholder.fill(QColor("#f0f0f0"))
        # We'll let the scene show text via a label overlay instead of drawing on pixmap
        self.viewer._scene.clear()
        self.viewer._pixmap_item = QGraphicsPixmapItem()
        self.viewer._pixmap_item.setPixmap(placeholder)
        self.viewer._scene.addItem(self.viewer._pixmap_item)

        text_item = self.viewer._scene.addText(
            "请点击「选择图片」加载颈椎侧位 X 光片",
            QFont("Microsoft YaHei", 12)
        )
        text_item.setDefaultTextColor(QColor("#888888"))
        rect = text_item.boundingRect()
        text_item.setPos((400 - rect.width()) / 2, (300 - rect.height()) / 2)
        self.viewer._scene.setSceneRect(QRectF(placeholder.rect()))

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 X 光片",
            "",
            "NIfTI (*.nii *.nii.gz);;PNG 图片 (*.png);;JPEG 图片 (*.jpg *.jpeg);;所有文件 (*.*)"
        )
        if not path:
            return

        self.btn_open.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.progress.setValue(0)
        self.status_label.setText("正在处理...")
        self.status_bar.showMessage(f"加载文件: {os.path.basename(path)}")

        self.worker = InferenceWorker(
            self.split_model, path, device=self.device
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, value, message):
        self.progress.setValue(value)
        self.status_label.setText(message)

    def _on_finished(self, pts_front, pts_back, orig_image, vis, spacing):
        self.current_pts = pts_front
        self.current_pts_back = pts_back
        self.current_image = orig_image
        self.vis_image = vis
        self.current_spacing = spacing

        # Compute all clinical parameters
        sp = spacing
        if isinstance(sp, (tuple, list)):
            sp = (float(sp[0]) + float(sp[1])) / 2.0
        if sp is None:
            sp = 1.0
        lordosis = compute_c2c7_lordosis(pts_front)
        max_cobb = compute_max_cobb(pts_front)
        sva = compute_c2c7_sva(pts_front, sp)
        t1_slope = compute_t1_slope(pts_front)
        disc_heights = compute_disc_heights(pts_front, sp)
        displacements = compute_vertebral_displacement(pts_front, sp)
        facet_angles = compute_facet_joint_angles(pts_back)

        self.current_results = {
            'lordosis': lordosis,
            'max_cobb': max_cobb,
            'sva': sva,
            't1_slope': t1_slope,
            'disc_heights': disc_heights,
            'displacements': displacements,
            'facet_angles': facet_angles,
        }

        # Enable all parameter toggles (default unchecked)
        for key in self.param_toggles:
            self.param_toggles[key]['btn'].setEnabled(True)
            self.param_toggles[key]['btn'].setChecked(False)

        # Default: show original image without overlays
        self._update_result_display()

        self.btn_open.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.status_label.setText("完成")
        self.status_bar.showMessage(f"已处理: {os.path.basename(self.worker.image_path)}")
        self.worker = None

    def _on_error(self, msg):
        self.btn_open.setEnabled(True)
        self.btn_save.setEnabled(False)
        for key in self.param_toggles:
            self.param_toggles[key]['btn'].setEnabled(False)
            self.param_toggles[key]['btn'].setChecked(False)
            self.param_toggles[key]['lbl'].setText("—")
        self.diagnosis_label.setText("诊断: —")
        self.diagnosis_label.setStyleSheet(
            "color: #888888; padding: 8px; background: #f0f0f0; border-radius: 6px;"
        )
        self.progress.setValue(0)
        self.status_label.setText("处理失败")
        self.status_bar.showMessage("就绪")
        QMessageBox.critical(self, "错误", f"处理图片失败:\n{msg}")
        self.worker = None

    def _set_diagnosis_style(self, diag):
        if "正常" in diag:
            self.diagnosis_label.setStyleSheet(
                "color: #2e7d32; padding: 8px; background: #e8f5e9; border-radius: 6px;"
            )
        elif "过度" in diag or "反弓" in diag:
            self.diagnosis_label.setStyleSheet(
                "color: #c62828; padding: 8px; background: #ffebee; border-radius: 6px;"
            )
        else:
            self.diagnosis_label.setStyleSheet(
                "color: #f57c00; padding: 8px; background: #fff3e0; border-radius: 6px;"
            )

    def _render_current_view(self):
        """Render image with all enabled parameter overlays."""
        if self.current_image is None:
            return self.current_image
        vis = self.current_image.copy()
        spacing = self.current_spacing or 1.0

        if self.param_toggles['cobb']['btn'].isChecked():
            vis = draw_cobb(vis, self.current_pts)
        if self.param_toggles['sva']['btn'].isChecked():
            vis = draw_sva(vis, self.current_pts, spacing)
        if self.param_toggles['t1']['btn'].isChecked():
            vis = draw_t1_slope(vis, self.current_pts)
        if self.param_toggles['disc']['btn'].isChecked():
            vis = draw_disc_heights(vis, self.current_pts, spacing)
        if self.param_toggles['disp']['btn'].isChecked():
            vis = draw_displacements(vis, self.current_pts, spacing)
        if self.param_toggles['facet']['btn'].isChecked():
            vis = draw_facet_angles(vis, self.current_pts_back)

        return vis

    def _update_result_display(self):
        if self.current_pts is None or self.current_image is None:
            return

        # Render image with overlays
        vis = self._render_current_view()
        self.current_display_image = vis

        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        h, w, ch = vis_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(vis_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.viewer.set_image(pixmap)

        # Update labels (always show values when results exist)
        res = self.current_results

        self.param_toggles['cobb']['lbl'].setText(
            f"C2-7: {res['lordosis']:.1f}°\n最大: {res['max_cobb']:.1f}°"
        )
        self.param_toggles['sva']['lbl'].setText(f"{res['sva']:+.1f} mm")
        self.param_toggles['t1']['lbl'].setText(f"{res['t1_slope']:.1f}°")

        dh = res['disc_heights']
        disc_lines = [f"{k}: {v:.1f} mm" for k, v in dh.items()]
        self.param_toggles['disc']['lbl'].setText('\n'.join(disc_lines))

        dp = res['displacements']
        disp_lines = [f"{k}: {v:+.1f} mm" for k, v in dp.items()]
        self.param_toggles['disp']['lbl'].setText('\n'.join(disp_lines))

        fa = res['facet_angles']
        facet_lines = [f"{k}: {v:.1f}°" for k, v in fa.items()]
        self.param_toggles['facet']['lbl'].setText('\n'.join(facet_lines))

        diag = diagnose(res['lordosis'])
        self.diagnosis_label.setText(f"诊断: {diag}")
        self._set_diagnosis_style(diag)

    def save_result(self):
        if getattr(self, 'current_display_image', None) is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存结果",
            "",
            "PNG 图片 (*.png);;JPEG 图片 (*.jpg *.jpeg)"
        )
        if path:
            try:
                cv2.imwrite(path, self.current_display_image)
            except Exception:
                vis_rgb = cv2.cvtColor(self.current_display_image, cv2.COLOR_BGR2RGB)
                from PIL import Image
                Image.fromarray(vis_rgb).save(path)
            QMessageBox.information(self, "成功", f"结果已保存到:\n{path}")

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.worker.wait(2000)
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))

    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vert_weights = os.path.join(base_dir, 'checkpoints', 'vertebrae.pth')
    facet_weights = os.path.join(base_dir, 'checkpoints', 'facets.pth')

    if not os.path.exists(vert_weights):
        QMessageBox.critical(None, "错误", f"椎体模型不存在:\n{vert_weights}")
        return
    if not os.path.exists(facet_weights):
        QMessageBox.critical(None, "错误", f"关节突模型不存在:\n{facet_weights}")
        return

    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    print(f"Loading models on {device}...")

    split_model = SplitVLDModel(vert_weights, facet_weights, use_tta=True, device=device)
    print("Split VLD models loaded successfully!")

    window = CobbApp(split_model, device=device)
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
