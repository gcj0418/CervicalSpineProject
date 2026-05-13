"""
GUI for Cobb angle calculation.
Uses Canvas + Scrollbar for reliable image display at correct aspect ratio.
"""
import sys
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

from inference import load_model, predict_fusion
from vld_inference import VLDModel
from cobb import compute_c2c7_lordosis, compute_max_cobb, diagnose, draw_cobb


class CobbApp:
    def __init__(self, root, model, config, vld_model):
        self.root = root
        self.model = model
        self.config = config
        self.vld_model = vld_model
        self.current_image = None
        self.current_pts = None
        self.vis_image = None
        self.photo = None

        root.title("颈椎 Cobb 角计算工具")
        root.geometry("1000x800")
        root.minsize(800, 600)

        # Configure root grid weights so canvas area expands
        root.grid_rowconfigure(1, weight=1)
        root.grid_columnconfigure(0, weight=1)

        # ---- Top frame: buttons ----
        top_frame = tk.Frame(root)
        top_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        self.btn_open = tk.Button(top_frame, text="选择图片", command=self.open_image, width=15, height=2)
        self.btn_open.pack(side=tk.LEFT, padx=5)

        self.btn_save = tk.Button(top_frame, text="保存结果", command=self.save_result, width=15, height=2, state=tk.DISABLED)
        self.btn_save.pack(side=tk.LEFT, padx=5)

        # ---- Middle frame: image display (Canvas + Scrollbars) ----
        mid_frame = tk.Frame(root, bg="#e0e0e0")
        mid_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        mid_frame.grid_rowconfigure(0, weight=1)
        mid_frame.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(mid_frame, bg="#f0f0f0", highlightthickness=0)
        self.h_scroll = tk.Scrollbar(mid_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.v_scroll = tk.Scrollbar(mid_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.h_scroll.set, yscrollcommand=self.v_scroll.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.h_scroll.grid(row=1, column=0, sticky="ew")

        # Placeholder text on canvas
        self.placeholder_id = self.canvas.create_text(
            0, 0,
            text="请点击「选择图片」加载颈椎侧位 X 光片",
            font=("Microsoft YaHei", 14),
            fill="#888888",
            anchor=tk.CENTER,
        )
        self._center_placeholder()

        # Bind resize to re-center placeholder
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # ---- Bottom frame: results ----
        bot_frame = tk.Frame(root)
        bot_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)

        self.result_text = tk.Label(bot_frame, text="", font=("Microsoft YaHei", 14), fg="red")
        self.result_text.pack()

        self.diagnosis_text = tk.Label(bot_frame, text="", font=("Microsoft YaHei", 12), fg="blue")
        self.diagnosis_text.pack(pady=5)

    def _on_canvas_configure(self, event=None):
        self._center_placeholder()

    def _center_placeholder(self):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        self.canvas.coords(self.placeholder_id, w // 2, h // 2)

    def open_image(self):
        path = filedialog.askopenfilename(
            title="选择 X 光片",
            filetypes=[("PNG 图片", "*.png"), ("JPEG 图片", "*.jpg;*.jpeg"), ("所有文件", "*.*")]
        )
        if not path:
            return

        try:
            pts, orig_image = predict_fusion(self.model, self.config, self.vld_model, path)
            self.current_pts = pts
            self.current_image = orig_image

            # Draw visualization on original-resolution image
            vis = draw_cobb(orig_image, pts)
            self.vis_image = vis

            # Convert to PIL
            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(vis_rgb)

            # Compute a display size that fits the canvas while keeping aspect ratio.
            # We leave a small margin so scrollbars don't immediately appear.
            canvas_w = max(self.canvas.winfo_width(), 400)
            canvas_h = max(self.canvas.winfo_height(), 300)
            margin = 20
            max_w = canvas_w - margin
            max_h = canvas_h - margin

            img_w, img_h = pil_img.size
            scale = min(max_w / img_w, max_h / img_h, 1.0)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)

            if scale < 1.0:
                pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

            self.photo = ImageTk.PhotoImage(pil_img)

            # Clear canvas and show image centered
            self.canvas.delete("all")
            self.canvas.create_image(
                canvas_w // 2, canvas_h // 2,
                image=self.photo, anchor=tk.CENTER
            )
            self.canvas.image = self.photo  # keep reference

            # Update scroll region (in case image is larger than canvas)
            self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

            # Show results
            lordosis = compute_c2c7_lordosis(pts)
            max_cobb = compute_max_cobb(pts)
            diag = diagnose(lordosis)

            self.result_text.config(text=f"C2-C7 前凸角: {lordosis:.1f}°    最大 Cobb 角: {max_cobb:.1f}°")
            self.diagnosis_text.config(text=f"诊断: {diag}")
            self.btn_save.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("错误", f"处理图片失败:\n{str(e)}")

    def save_result(self):
        if self.vis_image is None:
            return

        path = filedialog.asksaveasfilename(
            title="保存结果",
            defaultextension=".png",
            filetypes=[("PNG 图片", "*.png"), ("JPEG 图片", "*.jpg")]
        )
        if path:
            # cv2.imwrite can't handle unicode paths on Windows; use numpy + PIL
            try:
                cv2.imwrite(path, self.vis_image)
            except Exception:
                vis_rgb = cv2.cvtColor(self.vis_image, cv2.COLOR_BGR2RGB)
                Image.fromarray(vis_rgb).save(path)
            messagebox.showinfo("成功", f"结果已保存到:\n{path}")


def main():
    root = tk.Tk()

    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(base_dir, 'config', 'spine_renji_hrnet_w18_pretrained.yaml')
    model_path = os.path.join(base_dir, 'models', 'hrnet_renji.pth')

    if not os.path.exists(cfg_path):
        messagebox.showerror("错误", f"配置文件不存在:\n{cfg_path}")
        return
    if not os.path.exists(model_path):
        messagebox.showerror("错误", f"模型文件不存在:\n{model_path}")
        return

    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    print(f"Loading model on {device}...")

    model, config = load_model(cfg_path, model_path, device=device)
    print("HRNet model loaded successfully!")

    # Load VLD model
    vld_weights = os.path.join(base_dir, 'models', 'spinenet_renji.pth')
    if not os.path.exists(vld_weights):
        messagebox.showerror("错误", f"VLD 模型文件不存在:\n{vld_weights}")
        return
    vld_model = VLDModel(vld_weights, use_tta=True, device=device)
    print("VLD model loaded successfully!")

    app = CobbApp(root, model, config, vld_model)
    root.mainloop()


if __name__ == '__main__':
    main()
