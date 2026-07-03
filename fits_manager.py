import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import cv2
from PIL import Image, ImageTk, ImageDraw
from astropy.io import fits
from astropy.wcs import WCS
from tkinterdnd2 import DND_FILES, TkinterDnD
from scipy.interpolate import PchipInterpolator
from astropy.coordinates import SkyCoord, FK5
import astropy.units as u
from astropy.time import Time

# Curves widget managing separate curves for RGB/Luminance, Red, Green, Blue
class CurvesWidget(tk.Canvas):
    def __init__(self, parent, callback=None, **kwargs):
        kwargs.setdefault('width', 220)
        kwargs.setdefault('height', 220)
        kwargs.setdefault('bg', '#15151a')
        kwargs.setdefault('highlightthickness', 1)
        kwargs.setdefault('highlightbackground', '#4f46e5')
        super().__init__(parent, **kwargs)
        
        self.callback = callback
        
        self.points_dict = {
            "RGB/Luminance": [(0, 0), (255, 255)],
            "Red Channel Only": [(0, 0), (255, 255)],
            "Green Channel Only": [(0, 0), (255, 255)],
            "Blue Channel Only": [(0, 0), (255, 255)]
        }
        
        self.active_channel = "RGB/Luminance"
        self.selected_point_idx = None
        self.point_radius = 5
        
        self.hist_r = None
        self.hist_g = None
        self.hist_b = None
        self.hist_l = None
        
        self.bind("<Button-1>", self.on_click)
        self.bind("<B1-Motion>", self.on_drag)
        self.bind("<ButtonRelease-1>", self.on_release)
        self.bind("<Double-Button-1>", self.on_double_click)
        
        self.draw_grid()
        self.redraw()
        
    @property
    def points(self):
        return self.points_dict[self.active_channel]
        
    @points.setter
    def points(self, pts):
        self.points_dict[self.active_channel] = pts
        
    def set_active_channel(self, channel):
        self.active_channel = channel
        self.selected_point_idx = None
        self.redraw()

    def draw_grid(self):
        self.delete("grid")
        w, h = float(self.winfo_width()), float(self.winfo_height())
        if w < 10: w, h = 220.0, 220.0
        for i in range(1, 4):
            x = (w / 4) * i
            y = (h / 4) * i
            self.create_line(x, 0, x, h, fill="#27272f", dash=(2, 2), tags="grid")
            self.create_line(0, y, w, y, fill="#27272f", dash=(2, 2), tags="grid")

    def canvas_to_val(self, cx, cy):
        w = float(self.winfo_width()) if self.winfo_width() > 10 else 220.0
        h = float(self.winfo_height()) if self.winfo_height() > 10 else 220.0
        rx = (cx / w) * 255.0
        ry = (1.0 - (cy / h)) * 255.0
        return np.clip(rx, 0.0, 255.0), np.clip(ry, 0.0, 255.0)
        
    def val_to_canvas(self, vx, vy):
        w = float(self.winfo_width()) if self.winfo_width() > 10 else 220.0
        h = float(self.winfo_height()) if self.winfo_height() > 10 else 220.0
        cx = (vx / 255.0) * w
        cy = (1.0 - (vy / 255.0)) * h
        return cx, cy

    def update_histograms(self, base_img):
        if base_img is None:
            return
            
        h, w, c = base_img.shape
        if max(h, w) > 150:
            ratio = 150 / max(h, w)
            sample = cv2.resize(base_img, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_NEAREST)
        else:
            sample = base_img
            
        d_min, d_max = sample.min(), sample.max()
        if d_max > d_min:
            norm_8u = ((sample - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
        else:
            norm_8u = np.zeros(sample.shape, dtype=np.uint8)
            
        self.hist_r = cv2.calcHist([norm_8u], [0], None, [256], [0, 256])
        self.hist_g = cv2.calcHist([norm_8u], [1], None, [256], [0, 256])
        self.hist_b = cv2.calcHist([norm_8u], [2], None, [256], [0, 256])
        
        lum_8u = (0.2126 * norm_8u[:,:,0] + 0.7152 * norm_8u[:,:,1] + 0.0722 * norm_8u[:,:,2]).astype(np.uint8)
        self.hist_l = cv2.calcHist([lum_8u], [0], None, [256], [0, 256])
        
        max_val = max(self.hist_r.max(), self.hist_g.max(), self.hist_b.max(), self.hist_l.max())
        if max_val > 0:
            self.hist_r = (self.hist_r / max_val) * 150.0
            self.hist_g = (self.hist_g / max_val) * 150.0
            self.hist_b = (self.hist_b / max_val) * 150.0
            self.hist_l = (self.hist_l / max_val) * 150.0
            
        self.redraw()
        
    def redraw(self):
        self.delete("hist")
        self.delete("curve")
        self.delete("point")
        self.draw_grid()
        
        w = float(self.winfo_width()) if self.winfo_width() > 10 else 220.0
        h = float(self.winfo_height()) if self.winfo_height() > 10 else 220.0
        
        # 1. Draw Histograms
        if self.hist_r is not None:
            for channel_hist, color in [(self.hist_r, "#b91c1c"), (self.hist_g, "#047857"), (self.hist_b, "#1d4ed8"), (self.hist_l, "#4b5563")]:
                pts = []
                for vx in range(0, 256):
                    val = channel_hist[vx][0]
                    cx = (vx / 255.0) * w
                    cy = h - (val / 150.0) * (h * 0.7)
                    pts.append((cx, cy))
                
                for i in range(len(pts) - 1):
                    self.create_line(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1], fill=color, width=1, tags="hist")
        
        # 2. Draw all 4 curves
        colors_dict = {
            "RGB/Luminance": "#c084fc",
            "Red Channel Only": "#f87171",
            "Green Channel Only": "#34d399",
            "Blue Channel Only": "#60a5fa"
        }
        
        for key, points in self.points_dict.items():
            pts_sorted = sorted(points, key=lambda p: p[0])
            lut = self.get_lut_for_pts(pts_sorted)
            coords = []
            for vx in range(0, 256):
                vy = lut[vx]
                cx, cy = self.val_to_canvas(vx, vy)
                coords.append((cx, cy))
                
            is_active = (key == self.active_channel)
            curve_color = colors_dict[key]
            curve_width = 3.0 if is_active else 1.0
            curve_dash = None if is_active else (2, 2)
            
            for i in range(len(coords) - 1):
                self.create_line(coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1], 
                                 fill=curve_color, width=curve_width, dash=curve_dash, tags="curve")
            
            if is_active:
                for i, (vx, vy) in enumerate(pts_sorted):
                    cx, cy = self.val_to_canvas(vx, vy)
                    pt_color = "#f43f5e" if i == self.selected_point_idx else curve_color
                    self.create_oval(cx - self.point_radius, cy - self.point_radius, cx + self.point_radius, cy + self.point_radius, 
                                     fill=pt_color, outline="white", tags="point")

    def get_lut(self):
        return self.get_lut_for_pts(self.points)

    def get_lut_for_pts(self, pts_list):
        pts = sorted(pts_list, key=lambda p: p[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        
        if xs[0] > 0:
            xs.insert(0, 0.0)
            ys.insert(0, ys[0])
        if xs[-1] < 255:
            xs.append(255.0)
            ys.append(ys[-1])
            
        if len(xs) > 2:
            try:
                interp = PchipInterpolator(xs, ys)
                lut = interp(np.arange(256))
            except Exception:
                lut = np.interp(np.arange(256), xs, ys)
        else:
            lut = np.interp(np.arange(256), xs, ys)
            
        return np.clip(lut, 0, 255).astype(np.uint8)

    def on_click(self, event):
        cx, cy = event.x, event.y
        best_dist = 99999
        best_idx = None
        
        pts = self.points
        for i, (vx, vy) in enumerate(pts):
            px, py = self.val_to_canvas(vx, vy)
            dist = np.hypot(px - cx, py - cy)
            if dist < 12 and dist < best_dist:
                best_dist = dist
                best_idx = i
                
        if best_idx is not None:
            self.selected_point_idx = best_idx
        else:
            vx, vy = self.canvas_to_val(cx, cy)
            if not any(abs(p[0] - vx) < 8 for p in pts):
                pts.append((vx, vy))
                pts.sort(key=lambda p: p[0])
                self.selected_point_idx = pts.index((vx, vy))
                self.points = pts
                
        self.redraw()
        if self.callback:
            self.callback(is_dragging=True)

    def on_drag(self, event):
        if self.selected_point_idx is not None:
            idx = self.selected_point_idx
            vx, vy = self.canvas_to_val(event.x, event.y)
            pts = self.points
            
            if idx == 0:
                vx = 0.0
            elif idx == len(pts) - 1:
                vx = 255.0
            else:
                prev_x = pts[idx-1][0]
                next_x = pts[idx+1][0]
                vx = np.clip(vx, prev_x + 1.0, next_x - 1.0)
                
            pts[idx] = (vx, vy)
            self.points = pts
            self.redraw()
            if self.callback:
                self.callback(is_dragging=True)

    def on_release(self, event):
        self.selected_point_idx = None
        self.redraw()
        if self.callback:
            self.callback(is_dragging=False)

    def on_double_click(self, event):
        cx, cy = event.x, event.y
        pts = self.points
        for i, (vx, vy) in enumerate(pts):
            if i == 0 or i == len(pts) - 1:
                continue
            px, py = self.val_to_canvas(vx, vy)
            if np.hypot(px - cx, py - cy) < 12:
                pts.pop(i)
                self.points = pts
                self.selected_point_idx = None
                self.redraw()
                if self.callback:
                    self.callback(is_dragging=False)
                break


class FitsManagerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FitsManager")
        self.root.geometry("1400x870")
        
        # Color palette
        self.bg_color = "#121214"
        self.panel_color = "#1e1e24"
        self.control_bg = "#27272f"
        self.accent_color = "#4f46e5"
        self.text_color = "#f3f4f6"
        self.green_bright = "#00ff00"
        
        self.root.configure(bg=self.bg_color)
        
        # Configure styles
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure(".", background=self.panel_color, foreground=self.text_color, font=("Segoe UI", 9))
        self.style.configure("TLabel", background=self.panel_color, foreground=self.text_color)
        self.style.configure("TFrame", background=self.panel_color)
        self.style.configure("TLabelframe", background=self.panel_color, foreground=self.text_color, bordercolor=self.accent_color)
        self.style.configure("TLabelframe.Label", background=self.panel_color, foreground=self.text_color, font=("Segoe UI", 10, "bold"))
        self.style.configure("TButton", background=self.accent_color, foreground="white", borderwidth=0, padding=6, font=("Segoe UI", 9, "bold"))
        self.style.map("TButton", background=[("active", "#4338ca")])
        self.style.configure("Accent.TButton", background=self.accent_color, foreground="white")
        self.style.configure("TCheckbutton", background=self.panel_color, foreground=self.text_color)
        self.style.map("TCheckbutton", background=[("active", self.panel_color)], foreground=[("active", self.text_color)])
        
        self.style.configure("TCombobox", 
                             fieldbackground=self.control_bg, 
                             background=self.control_bg, 
                             foreground=self.text_color,
                             arrowcolor="white",
                             bordercolor=self.accent_color,
                             lightcolor=self.control_bg,
                             darkcolor=self.control_bg)
        
        self.style.map("TCombobox", 
                       fieldbackground=[("readonly", self.control_bg), ("active", self.control_bg)],
                       background=[("readonly", self.control_bg), ("active", self.control_bg)],
                       foreground=[("readonly", self.text_color), ("active", self.text_color)])
                       
        self.root.option_add("*TCombobox*Listbox.background", self.control_bg)
        self.root.option_add("*TCombobox*Listbox.foreground", self.text_color)
        self.root.option_add("*TCombobox*Listbox.selectBackground", self.accent_color)
        self.root.option_add("*TCombobox*Listbox.selectForeground", "white")
        
        self.style.configure("Treeview", background=self.control_bg, fieldbackground=self.control_bg, foreground=self.text_color, rowheight=20)
        self.style.configure("Treeview.Heading", background=self.panel_color, foreground=self.text_color, borderwidth=1)
        self.style.map("Treeview.Heading", background=[("active", self.accent_color)])
        
        # State variables
        self.fits_path = None
        self.fits_header = None
        self.fits_data = None  # Full-res numpy data
        self.debayered_cache = None  # Cached RGB base
        
        self.processed_img_full = None  # High-res RGB unit8
        self.processed_img_preview = None  # Downsampled base
        
        self.undo_stack = []
        self.redo_stack = []
        
        # Mirror states
        self.mirror_horizontal = False
        self.mirror_vertical = False
        
        # Interactive Zoom & Pan parameters
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.is_panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        
        # Ephemeris date parsing
        self.observation_time = None  # Time object
        
        # Temporary star coords
        self.temp_marker = None  
        
        # Crop / Selection variables
        self.crop_mode = False
        self.crop_start = None
        self.crop_rect_id = None
        
        # Color Balance
        self.balance_mode = "None"  
        self.sky_sample_rgb = None
        self.star_sample_rgb = None
        
        # Annotation variables
        self.annotation_mode = False
        self.annotations = []  # List of dicts: {'x': ratio_x, 'y': ratio_y, 'text': text}
        
        # UI controls vars
        self.bayer_pattern = tk.StringVar(value="None")
        self.channel_selection = tk.StringVar(value="RGB/Luminance")
        
        # Build UI layout
        self.create_widgets()
        self.bind_events()
        
        # Register Drag & Drop
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind("<<Drop>>", self.handle_file_drop)
        
    def create_widgets(self):
        # Reorganize top toolbar to occupy 2 rows to fit all buttons neatly without cutting off
        toolbar_container = tk.Frame(self.root, bg=self.panel_color, bd=0)
        toolbar_container.pack(side="top", fill="x")
        
        toolbar_row1 = tk.Frame(toolbar_container, bg=self.panel_color, height=35)
        toolbar_row1.pack(side="top", fill="x", padx=10, pady=(4, 2))
        
        toolbar_row2 = tk.Frame(toolbar_container, bg=self.panel_color, height=35)
        toolbar_row2.pack(side="top", fill="x", padx=10, pady=(2, 4))
        
        # Row 1 buttons: Loading, Undo/Redo, Zoom & View reset, Mirrors
        btn_open = tk.Button(toolbar_row1, text="Load FITS", command=self.load_fits, bg=self.accent_color, fg="white", font=("Segoe UI", 9, "bold"), bd=0, padx=12, pady=4)
        btn_open.pack(side="left", padx=(0, 10))
        
        btn_undo = tk.Button(toolbar_row1, text="Undo", command=self.undo, bg="#374151", fg="white", font=("Segoe UI", 9), bd=0, padx=12, pady=4)
        btn_undo.pack(side="left", padx=3)
        
        btn_redo = tk.Button(toolbar_row1, text="Redo", command=self.redo, bg="#374151", fg="white", font=("Segoe UI", 9), bd=0, padx=12, pady=4)
        btn_redo.pack(side="left", padx=3)
        
        # Zoom panel controls
        zoom_frame = tk.Frame(toolbar_row1, bg=self.panel_color)
        zoom_frame.pack(side="left", padx=15)
        
        btn_zoom_in = tk.Button(zoom_frame, text="🔍+", command=self.zoom_in, bg="#374151", fg="white", font=("Segoe UI", 9, "bold"), bd=0, padx=8)
        btn_zoom_in.pack(side="left", padx=2)
        btn_zoom_out = tk.Button(zoom_frame, text="🔍-", command=self.zoom_out, bg="#374151", fg="white", font=("Segoe UI", 9, "bold"), bd=0, padx=8)
        btn_zoom_out.pack(side="left", padx=2)
        btn_reset_zoom = tk.Button(zoom_frame, text="Reset View", command=self.reset_view, bg="#374151", fg="white", font=("Segoe UI", 9), bd=0, padx=6)
        btn_reset_zoom.pack(side="left", padx=5)
        
        # Mirror / Telescope simulation buttons
        mirror_frame = tk.Frame(toolbar_row1, bg=self.panel_color)
        mirror_frame.pack(side="left", padx=10)
        self.btn_mirror_h = tk.Button(mirror_frame, text="Flip Horizontal", command=self.toggle_mirror_h, bg="#374151", fg="white", font=("Segoe UI", 9), bd=0, padx=8)
        self.btn_mirror_h.pack(side="left", padx=2)
        self.btn_mirror_v = tk.Button(mirror_frame, text="Flip Vertical", command=self.toggle_mirror_v, bg="#374151", fg="white", font=("Segoe UI", 9), bd=0, padx=8)
        self.btn_mirror_v.pack(side="left", padx=2)
        
        # Row 2 buttons: Crop, Annotate, Samplers, Mark WCS, and Export Image
        self.btn_crop = tk.Button(toolbar_row2, text="Crop Mode: Off", command=self.toggle_crop_mode, bg="#374151", fg="white", font=("Segoe UI", 9), bd=0, padx=12, pady=4)
        self.btn_crop.pack(side="left", padx=(0, 5))
        
        self.btn_annotate = tk.Button(toolbar_row2, text="Annotate Mode: Off", command=self.toggle_annotation_mode, bg="#374151", fg="white", font=("Segoe UI", 9), bd=0, padx=12, pady=4)
        self.btn_annotate.pack(side="left", padx=3)
        
        # Color Balance Sampler Mode Buttons
        balance_frame = tk.Frame(toolbar_row2, bg=self.panel_color)
        balance_frame.pack(side="left", padx=10)
        
        self.btn_bal_sky = tk.Button(balance_frame, text="Sky Background (Black)", command=self.enable_sky_balance, bg="#374151", fg="white", font=("Segoe UI", 9), bd=0, padx=8)
        self.btn_bal_sky.pack(side="left", padx=2)
        self.btn_bal_star = tk.Button(balance_frame, text="White Star (White)", command=self.enable_star_balance, bg="#374151", fg="white", font=("Segoe UI", 9), bd=0, padx=8)
        self.btn_bal_star.pack(side="left", padx=2)
        
        # Grayscale and Reset controls
        control_frame = tk.Frame(toolbar_row2, bg=self.panel_color)
        control_frame.pack(side="left", padx=5)
        
        self.var_grayscale = tk.BooleanVar(value=False)
        self.chk_grayscale = tk.Checkbutton(control_frame, text="B&W (Grayscale)", variable=self.var_grayscale, command=lambda: self.process_and_update(is_dragging=False), bg=self.panel_color, fg=self.text_color, selectcolor=self.control_bg, activebackground=self.panel_color, activeforeground=self.text_color)
        self.chk_grayscale.pack(side="left", padx=5)
        
        btn_reset_colors = tk.Button(control_frame, text="Reset Color & Curves", command=self.reset_color_manipulation, bg="#b91c1c", fg="white", font=("Segoe UI", 9, "bold"), bd=0, padx=10, pady=4)
        btn_reset_colors.pack(side="left", padx=5)
        
        # RADEC Mark star button
        btn_mark_radec = tk.Button(toolbar_row2, text="Mark RA/DEC Target", command=self.mark_radec_input, bg="#4f46e5", fg="white", font=("Segoe UI", 9, "bold"), bd=0, padx=12, pady=4)
        btn_mark_radec.pack(side="left", padx=10)
        
        # Export Image Button (padded properly to not cut off)
        btn_export = tk.Button(toolbar_row2, text="Export Image", command=self.export_image, bg="#10b981", fg="white", font=("Segoe UI", 10, "bold"), bd=0, padx=16, pady=5)
        btn_export.pack(side="right", padx=10)
        
        # Main splitter
        main_pane = tk.PanedWindow(self.root, orient="horizontal", bg=self.bg_color, bd=0, sashwidth=6)
        main_pane.pack(fill="both", expand=True)
        
        # Left Panel (Controls & Metadata)
        left_panel = tk.Frame(main_pane, bg=self.panel_color, width=340)
        left_panel.pack_propagate(False)
        main_pane.add(left_panel, minsize=335)
        
        # Canvas Frame (Right Panel)
        right_panel = tk.Frame(main_pane, bg=self.bg_color)
        main_pane.add(right_panel, minsize=500)
        
        # Scrollable Left Content
        canvas_left = tk.Canvas(left_panel, bg=self.panel_color, bd=0, highlightthickness=0)
        scrollbar_left = tk.Scrollbar(left_panel, orient="vertical", command=canvas_left.yview)
        scrollable_frame = tk.Frame(canvas_left, bg=self.panel_color)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas_left.configure(scrollregion=canvas_left.bbox("all"))
        )
        canvas_left.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas_left.configure(yscrollcommand=scrollbar_left.set)
        
        canvas_left.pack(side="left", fill="both", expand=True)
        scrollbar_left.pack(side="right", fill="y")
        
        lbl_drag_info = tk.Label(scrollable_frame, text="💡 Tip: Drag & Drop FITS file here to open", bg=self.panel_color, fg="#9ca3af", font=("Segoe UI", 8, "italic"))
        lbl_drag_info.pack(fill="x", padx=10, pady=(5, 0))
        
        # FITS Header Metadata section
        meta_lf = ttk.LabelFrame(scrollable_frame, text="Header Metadata")
        meta_lf.pack(fill="x", padx=10, pady=5)
        
        self.meta_tree = ttk.Treeview(meta_lf, columns=("Value"), show="tree headings", height=5)
        self.meta_tree.heading("#0", text="Key")
        self.meta_tree.heading("Value", text="Value")
        self.meta_tree.column("#0", width=110)
        self.meta_tree.column("Value", width=160)
        self.meta_tree.pack(fill="x", padx=5, pady=5)
        
        # Coordinates Readout
        coord_lf = ttk.LabelFrame(scrollable_frame, text="Coordinates (WCS Epoch Conversion)")
        coord_lf.pack(fill="x", padx=10, pady=5)
        self.lbl_cursor_coords = ttk.Label(coord_lf, text="Cursor: X=0, Y=0")
        self.lbl_cursor_coords.pack(anchor="w", padx=5, pady=2)
        
        self.lbl_wcs_j2000 = ttk.Label(coord_lf, text="J2000: RA=N/A, DEC=N/A", font=("Segoe UI", 9, "bold"))
        self.lbl_wcs_j2000.pack(anchor="w", padx=5, pady=2)
        
        self.lbl_wcs_jnow = ttk.Label(coord_lf, text="JNow: RA=N/A, DEC=N/A", font=("Segoe UI", 9, "bold"))
        self.lbl_wcs_jnow.pack(anchor="w", padx=5, pady=2)
        
        self.lbl_epoch_status = ttk.Label(coord_lf, text="Header epoch: J2000 (detected)", font=("Segoe UI", 8, "italic"))
        self.lbl_epoch_status.pack(anchor="w", padx=5, pady=2)
        
        # Debayer Settings
        debayer_lf = ttk.LabelFrame(scrollable_frame, text="Debayer")
        debayer_lf.pack(fill="x", padx=10, pady=5)
        
        bayer_opts = ["None", "RGGB", "BGGR", "GRBG", "GBRG"]
        self.debayer_menu = ttk.Combobox(debayer_lf, textvariable=self.bayer_pattern, values=bayer_opts, state="readonly")
        self.debayer_menu.pack(fill="x", padx=5, pady=5)
        self.debayer_menu.bind("<<ComboboxSelected>>", self.on_debayer_selection_change)
        
        # Channel manipulation
        chan_lf = ttk.LabelFrame(scrollable_frame, text="Channel Manipulation")
        chan_lf.pack(fill="x", padx=10, pady=5)
        
        chan_opts = ["RGB/Luminance", "Red Channel Only", "Green Channel Only", "Blue Channel Only"]
        self.chan_menu = ttk.Combobox(chan_lf, textvariable=self.channel_selection, values=chan_opts, state="readonly")
        self.chan_menu.pack(fill="x", padx=5, pady=5)
        self.chan_menu.bind("<<ComboboxSelected>>", self.on_channel_combobox_change)
        
        # Curves editor
        curves_lf = ttk.LabelFrame(scrollable_frame, text="Curves Adjustment")
        curves_lf.pack(fill="x", padx=10, pady=5)
        
        btn_autostretch = tk.Button(curves_lf, text="Auto Adaptation (Stretch)", command=self.apply_autostretch, bg=self.accent_color, fg="white", font=("Segoe UI", 9), bd=0, pady=4)
        btn_autostretch.pack(fill="x", padx=5, pady=5)
        
        self.curves_widget = CurvesWidget(curves_lf, callback=self.on_curves_changed)
        self.curves_widget.pack(padx=5, pady=5)
        
        lbl_curves_tip = tk.Label(curves_lf, text="Click to add point, drag to modify.\nDouble click point to remove it.", bg=self.panel_color, fg="#9ca3af", font=("Segoe UI", 8))
        lbl_curves_tip.pack(pady=(0, 5))
        
        self.var_invert = tk.BooleanVar(value=False)
        self.chk_invert = ttk.Checkbutton(curves_lf, text="Negative Image (Invert)", variable=self.var_invert, command=lambda: self.process_and_update(is_dragging=False))
        self.chk_invert.pack(anchor="w", padx=5, pady=5)
        
        # Canvas Container Frame (Right Panel) holding Canvas and scrollbars
        canvas_container = tk.Frame(right_panel, bg=self.bg_color)
        canvas_container.pack(fill="both", expand=True)
        
        self.hbar = tk.Scrollbar(canvas_container, orient="horizontal")
        self.hbar.pack(side="bottom", fill="x")
        self.vbar = tk.Scrollbar(canvas_container, orient="vertical")
        self.vbar.pack(side="right", fill="y")
        
        self.canvas = tk.Canvas(canvas_container, bg="#111", highlightthickness=0,
                                xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        
        self.hbar.config(command=self.on_hbar_scroll)
        self.vbar.config(command=self.on_vbar_scroll)
        
    def bind_events(self):
        self.canvas.bind("<Motion>", self.on_canvas_motion)
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_click_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        
        self.canvas.bind("<ButtonPress-3>", self.on_pan_start)
        self.canvas.bind("<B3-Motion>", self.on_pan_drag)
        self.canvas.bind("<ButtonRelease-3>", self.on_pan_end)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-y>", lambda e: self.redo())
        self.root.bind("<Configure>", self.on_window_resize)

    def toggle_mirror_h(self):
        self.mirror_horizontal = not self.mirror_horizontal
        self.btn_mirror_h.config(bg=self.accent_color if self.mirror_horizontal else "#374151")
        self.process_and_update(is_dragging=False)
        
    def toggle_mirror_v(self):
        self.mirror_vertical = not self.mirror_vertical
        self.btn_mirror_v.config(bg=self.accent_color if self.mirror_vertical else "#374151")
        self.process_and_update(is_dragging=False)

    def enable_sky_balance(self):
        self.balance_mode = "Sky"
        self.canvas.config(cursor="crosshair")
        self.btn_bal_sky.config(bg="#d97706")
        self.btn_bal_star.config(bg="#374151")
        
    def enable_star_balance(self):
        self.balance_mode = "Star"
        self.canvas.config(cursor="crosshair")
        self.btn_bal_star.config(bg="#d97706")
        self.btn_bal_sky.config(bg="#374151")

    def zoom_in(self, target_x=None, target_y=None):
        self.zoom_to_target(1.3, target_x, target_y)
        
    def zoom_out(self, target_x=None, target_y=None):
        self.zoom_to_target(1.0 / 1.3, target_x, target_y)
        
    def zoom_to_target(self, factor, target_x=None, target_y=None):
        if self.processed_img_full is None:
            return
            
        h, w = self.processed_img_full.shape[:2]
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            return
            
        # Target coordinate in canvas space (default to center)
        cx = target_x if target_x is not None else (canvas_w / 2.0)
        cy = target_y if target_y is not None else (canvas_h / 2.0)
        
        # Pixel calculation relative to current scale & offsets
        fit_ratio = min(canvas_w / w, canvas_h / h)
        old_scale = fit_ratio * self.zoom_level
        
        # Calculate pixel under mouse before zoom
        orig_img_x = (cx - (canvas_w - w * old_scale) / 2.0 - self.pan_x) / old_scale
        orig_img_y = (cy - (canvas_h - h * old_scale) / 2.0 - self.pan_y) / old_scale
        
        # Update zoom level
        new_zoom = np.clip(self.zoom_level * factor, 0.2, 20.0)
        if new_zoom == self.zoom_level:
            return
        self.zoom_level = new_zoom
        
        # Calculate new scale
        new_scale = fit_ratio * self.zoom_level
        
        # Center the same image pixel at the target canvas coordinate after zooming
        self.pan_x = cx - (canvas_w - w * new_scale) / 2.0 - orig_img_x * new_scale
        self.pan_y = cy - (canvas_h - h * new_scale) / 2.0 - orig_img_y * new_scale
        
        self.render_canvas(is_dragging=False)
        
    def reset_view(self):
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.render_canvas(is_dragging=False)
        
    def on_mousewheel(self, event):
        # Zoom towards the mouse cursor position
        if event.delta > 0:
            self.zoom_in(event.x, event.y)
        else:
            self.zoom_out(event.x, event.y)

    def on_pan_start(self, event):
        self.is_panning = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        # Hand cursor during panning
        self.canvas.config(cursor="hand1")
        
    def on_pan_drag(self, event):
        if self.is_panning:
            dx = event.x - self.pan_start_x
            dy = event.y - self.pan_start_y
            self.pan_x += dx
            self.pan_y += dy
            self.pan_start_x = event.x
            self.pan_start_y = event.y
            self.render_canvas(is_dragging=True)
            
    def on_pan_end(self, event):
        self.is_panning = False
        self.canvas.config(cursor="")
        self.render_canvas(is_dragging=False)

    def on_hbar_scroll(self, *args):
        # Support scrolling using bottom scrollbar
        if len(args) >= 2:
            cmd = args[0]
            if cmd == "moveto":
                val = float(args[1])
                # Estimate new pan_x based on scrollbar position
                canvas_w = self.canvas.winfo_width()
                if self.processed_img_full is not None:
                    h, w = self.processed_img_full.shape[:2]
                    fit_ratio = min(canvas_w / w, self.canvas.winfo_height() / h)
                    total_w = w * fit_ratio * self.zoom_level
                    margin = (canvas_w - total_w) / 2.0
                    self.pan_x = -(val * total_w) - margin
            self.render_canvas(is_dragging=False)

    def on_vbar_scroll(self, *args):
        # Support scrolling using right scrollbar
        if len(args) >= 2:
            cmd = args[0]
            if cmd == "moveto":
                val = float(args[1])
                canvas_h = self.canvas.winfo_height()
                if self.processed_img_full is not None:
                    h, w = self.processed_img_full.shape[:2]
                    fit_ratio = min(self.canvas.winfo_width() / w, canvas_h / h)
                    total_h = h * fit_ratio * self.zoom_level
                    margin = (canvas_h - total_h) / 2.0
                    self.pan_y = -(val * total_h) - margin
            self.render_canvas(is_dragging=False)

    def handle_file_drop(self, event):
        file_path = event.data
        if file_path.startswith("{") and file_path.endswith("}"):
            file_path = file_path[1:-1]
        if os.path.exists(file_path):
            self.load_fits_from_path(file_path)
            
    def load_fits(self):
        file_path = filedialog.askopenfilename(filetypes=[("FITS files", "*.fits *.fit"), ("All files", "*.*")])
        if file_path:
            self.load_fits_from_path(file_path)
            
    def load_fits_from_path(self, file_path):
        try:
            with fits.open(file_path) as hdul:
                hdu = None
                for h in hdul:
                    if h.data is not None and h.data.ndim in [2, 3]:
                        hdu = h
                        break
                
                if hdu is None:
                    messagebox.showerror("Error", "No image data found in FITS file.")
                    return
                
                self.fits_path = file_path
                self.fits_header = hdu.header
                self.fits_data = hdu.data.astype(np.float32)
                self.debayered_cache = None
                self.temp_marker = None
                
                self.sky_sample_rgb = None
                self.star_sample_rgb = None
                self.balance_mode = "None"
                self.btn_bal_sky.config(bg="#374151")
                self.btn_bal_star.config(bg="#374151")
                self.var_grayscale.set(False)
                
                if 'BAYERPAT' in self.fits_header:
                    self.bayer_pattern.set(self.fits_header['BAYERPAT'].upper())
                else:
                    self.bayer_pattern.set("None")
                
                self.undo_stack.clear()
                self.redo_stack.clear()
                self.annotations.clear()
                self.zoom_level = 1.0
                self.pan_x = 0.0
                self.pan_y = 0.0
                self.mirror_horizontal = False
                self.mirror_vertical = False
                self.btn_mirror_h.config(bg="#374151")
                self.btn_mirror_v.config(bg="#374151")
                
                self.observation_time = Time.now()
                if 'DATE-OBS' in self.fits_header:
                    try:
                        self.observation_time = Time(self.fits_header['DATE-OBS'], format='isot')
                    except Exception:
                        pass
                
                for k in self.curves_widget.points_dict:
                    self.curves_widget.points_dict[k] = [(0, 0), (255, 255)]
                self.curves_widget.redraw()
                
                self.meta_tree.delete(*self.meta_tree.get_children())
                important_keys = ["OBJECT", "EXPTIME", "TELESCOP", "INSTRUME", "FILTER", "DATE-OBS", "BAYERPAT", "EQUINOX"]
                for key in important_keys:
                    if key in self.fits_header:
                        self.meta_tree.insert("", "end", text=key, values=(str(self.fits_header[key]),))
                for key, val in self.fits_header.items():
                    if key not in important_keys and key.strip() != "":
                        self.meta_tree.insert("", "end", text=key, values=(str(val),))
                
                try:
                    self.wcs = WCS(self.fits_header, naxis=2)
                except Exception:
                    self.wcs = None
                
                epoch_val = self.fits_header.get('EQUINOX', 2000.0)
                if epoch_val == 2000.0:
                    self.lbl_epoch_status.config(text=f"Header epoch: J2000 (detected)")
                else:
                    self.lbl_epoch_status.config(text=f"Header epoch: JNow / EQUINOX {epoch_val}")
                
                self.push_state()
                self.process_and_update(is_dragging=False)
                
        except Exception as e:
            messagebox.showerror("Error", f"Could not load FITS: {str(e)}")

    def push_state(self):
        state = {
            'data': self.fits_data.copy(),
            'bayer': self.bayer_pattern.get(),
            'channel': self.channel_selection.get(),
            'points_dict': {k: list(v) for k, v in self.curves_widget.points_dict.items()},
            'invert': self.var_invert.get(),
            'grayscale': self.var_grayscale.get(),
            'annotations': list(self.annotations),
            'sky_sample': self.sky_sample_rgb,
            'star_sample': self.star_sample_rgb
        }
        self.undo_stack.append(state)
        if len(self.undo_stack) > 30:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        
    def undo(self):
        if len(self.undo_stack) <= 1:
            return
        state = self.undo_stack.pop()
        self.redo_stack.append(state)
        
        prev_state = self.undo_stack[-1]
        self.restore_state(prev_state)
        
    def redo(self):
        if not self.redo_stack:
            return
        state = self.redo_stack.pop()
        self.undo_stack.append(state)
        self.restore_state(state)
        
    def restore_state(self, state):
        self.fits_data = state['data'].copy()
        self.bayer_pattern.set(state['bayer'])
        self.channel_selection.set(state['channel'])
        self.debayered_cache = None
        
        self.curves_widget.points_dict = {k: list(v) for k, v in state['points_dict'].items()}
        self.curves_widget.redraw()
        
        self.var_invert.set(state['invert'])
        self.var_grayscale.set(state.get('grayscale', False))
        self.annotations = list(state['annotations'])
        self.sky_sample_rgb = state['sky_sample']
        self.star_sample_rgb = state['star_sample']
        
        self.process_and_update(is_dragging=False)

    def on_debayer_selection_change(self, event):
        self.debayered_cache = None
        self.process_and_update(is_dragging=False)

    def on_channel_combobox_change(self, event):
        self.curves_widget.set_active_channel(self.channel_selection.get())
        self.process_and_update(is_dragging=False)

    def on_curves_changed(self, is_dragging=False):
        self.process_and_update(is_dragging=is_dragging)

    def update_debayer_cache(self):
        if self.fits_data is None:
            return
            
        data = self.fits_data.copy()
        bayer = self.bayer_pattern.get()
        
        if bayer != "None" and data.ndim == 2:
            d_min, d_max = data.min(), data.max()
            if d_max > d_min:
                norm = ((data - d_min) / (d_max - d_min) * 65535.0).astype(np.uint16)
            else:
                norm = data.astype(np.uint16)
                
            code = None
            if bayer == "RGGB":
                code = cv2.COLOR_BayerBG2BGR
            elif bayer == "BGGR":
                code = cv2.COLOR_BayerRG2BGR
            elif bayer == "GRBG":
                code = cv2.COLOR_BayerGB2BGR
            elif bayer == "GBRG":
                code = cv2.COLOR_BayerGR2BGR
                
            if code is not None:
                img_bgr = cv2.cvtColor(norm, code)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                self.debayered_cache = img_rgb.astype(np.float32) / 256.0
            else:
                self.debayered_cache = np.stack([data, data, data], axis=-1)
        else:
            if data.ndim == 2:
                self.debayered_cache = np.stack([data, data, data], axis=-1)
            elif data.ndim == 3 and data.shape[0] == 3:
                self.debayered_cache = np.transpose(data, (1, 2, 0))
            else:
                self.debayered_cache = data.copy()
                
        self.curves_widget.update_histograms(self.debayered_cache)

    def process_and_update(self, is_dragging=False):
        if self.fits_data is None:
            return
            
        if self.debayered_cache is None:
            self.update_debayer_cache()
            
        base_img = self.debayered_cache.copy()
        h, w = base_img.shape[:2]
        
        if is_dragging:
            max_preview_dim = 600
            if max(h, w) > max_preview_dim:
                ratio = max_preview_dim / max(h, w)
                preview_w = int(w * ratio)
                preview_h = int(h * ratio)
                base_img = cv2.resize(base_img, (preview_w, preview_h), interpolation=cv2.INTER_NEAREST)
        
        # 1. Normalize image using global bounds of the full loaded FITS image (debayered_cache) 
        # to ensure zoom and crop do not alter normalization or curves correction.
        global_min = self.debayered_cache.min()
        global_max = self.debayered_cache.max()
        global_range = global_max - global_min if global_max > global_min else 1.0
        
        data_norm = (base_img - global_min) / global_range
        data_norm = np.clip(data_norm, 0.0, 1.0)
            
        # Apply Color Balance offsets & gains relative to the global normalized image scale
        if self.sky_sample_rgb is not None or self.star_sample_rgb is not None:
            black_offset = np.array(self.sky_sample_rgb if self.sky_sample_rgb is not None else [0.0, 0.0, 0.0])
            white_gain = np.array(self.star_sample_rgb if self.star_sample_rgb is not None else [1.0, 1.0, 1.0])
            
            # Recalculate normalized values with color balance calibration offsets/gains:
            # (value - black) / (white - black)
            denom = white_gain - black_offset
            denom[denom <= 0.0001] = 0.0001
            
            data_norm = (data_norm - black_offset) / denom
            data_norm = np.clip(data_norm, 0.0, 1.0)
            
        # 2. Convert to uint8
        img_8u = (data_norm * 255.0).astype(np.uint8)
        
        # 3. Apply Multi-Channel Curves
        lut_rgb = self.curves_widget.get_lut_for_pts(self.curves_widget.points_dict["RGB/Luminance"])
        img_processed = cv2.LUT(img_8u, lut_rgb)
        
        lut_r = self.curves_widget.get_lut_for_pts(self.curves_widget.points_dict["Red Channel Only"])
        lut_g = self.curves_widget.get_lut_for_pts(self.curves_widget.points_dict["Green Channel Only"])
        lut_b = self.curves_widget.get_lut_for_pts(self.curves_widget.points_dict["Blue Channel Only"])
        
        img_processed[:, :, 0] = cv2.LUT(img_processed[:, :, 0], lut_r)
        img_processed[:, :, 1] = cv2.LUT(img_processed[:, :, 1], lut_g)
        img_processed[:, :, 2] = cv2.LUT(img_processed[:, :, 2], lut_b)
        
        # 3b. Convert to Grayscale (B&W) if toggled
        if self.var_grayscale.get():
            # Standard relative luminance weights for stars and skies
            gray = (0.2126 * img_processed[:, :, 0] + 0.7152 * img_processed[:, :, 1] + 0.0722 * img_processed[:, :, 2]).astype(np.uint8)
            img_processed = np.stack([gray, gray, gray], axis=-1)
            
        # Apply Mirror / Telescope flips
        if self.mirror_horizontal:
            img_processed = cv2.flip(img_processed, 1)
        if self.mirror_vertical:
            img_processed = cv2.flip(img_processed, 0)
            
        # 4. Invert Colors
        if self.var_invert.get():
            img_processed = 255 - img_processed
            
        if is_dragging:
            self.processed_img_preview = img_processed
        else:
            self.processed_img_full = img_processed
            self.processed_img_preview = img_processed
            
        self.render_canvas(is_dragging=is_dragging)

    def reset_color_manipulation(self):
        if self.fits_data is None:
            return
            
        self.push_state()
        
        # Reset color balance samplers
        self.sky_sample_rgb = None
        self.star_sample_rgb = None
        
        # Reset curves points to linear defaults
        for k in self.curves_widget.points_dict:
            self.curves_widget.points_dict[k] = [(0, 0), (255, 255)]
            
        self.curves_widget.redraw()
        self.process_and_update(is_dragging=False)
        messagebox.showinfo("Reset", "Color Balance gains and Curves adjustments have been reset.")

    def apply_autostretch(self):
        if self.fits_data is None:
            return
            
        self.push_state()
        data = self.fits_data
        
        if data.ndim == 3:
            lum = 0.2126 * data[:,:,0] + 0.7152 * data[:,:,1] + 0.0722 * data[:,:,2]
        else:
            lum = data
            
        median = np.median(lum)
        mad = np.median(np.abs(lum - median))
        
        sigma_black = 2.8
        sigma_white = 8.0
        
        black_val = max(lum.min(), median - sigma_black * mad)
        white_val = min(lum.max(), median + sigma_white * mad)
        
        full_min = lum.min()
        full_max = lum.max()
        if full_max > full_min:
            black_idx = int(((black_val - full_min) / (full_max - full_min)) * 255.0)
            white_idx = int(((white_val - full_min) / (full_max - full_min)) * 255.0)
        else:
            black_idx = 0
            white_idx = 255
            
        black_idx = max(0, min(black_idx, 250))
        white_idx = max(black_idx + 5, min(white_idx, 255))
        
        mid_x = int((black_idx + white_idx) * 0.4)
        mid_y = 140
        
        self.curves_widget.points = [
            (0, 0),
            (black_idx, 0),
            (mid_x, mid_y),
            (white_idx, 255),
            (255, 255)
        ]
        self.curves_widget.redraw()
        self.process_and_update(is_dragging=False)

    def draw_star_crosshair(self, draw, ax, ay, size, width=2):
        inner_gap = int(size * 0.25)
        inner_gap = max(10, inner_gap)
        outer_radius = int(size * 0.85)
        outer_radius = max(35, outer_radius)
        
        draw.line([(ax - outer_radius, ay), (ax - inner_gap, ay)], fill=self.green_bright, width=width)
        draw.line([(ax + inner_gap, ay), (ax + outer_radius, ay)], fill=self.green_bright, width=width)
        draw.line([(ax, ay - outer_radius), (ax, ay - inner_gap)], fill=self.green_bright, width=width)
        draw.line([(ax, ay + inner_gap), (ax, ay + outer_radius)], fill=self.green_bright, width=width)

    def render_canvas(self, is_dragging=False):
        img_to_draw = self.processed_img_preview if is_dragging else self.processed_img_full
        if img_to_draw is None:
            return
            
        h, w = img_to_draw.shape[:2]
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            return
            
        fit_ratio = min(canvas_w / w, canvas_h / h)
        total_scale = fit_ratio * self.zoom_level
        
        new_w = int(w * total_scale)
        new_h = int(h * total_scale)
        
        self.img_scale_ratio = total_scale
        self.img_offset_x = (canvas_w - new_w) // 2 + int(self.pan_x)
        self.img_offset_y = (canvas_h - new_h) // 2 + int(self.pan_y)
        
        pil_img = Image.fromarray(img_to_draw)
        if new_w < 5 or new_h < 5:
            return
            
        resized_pil = pil_img.resize((new_w, new_h), Image.Resampling.NEAREST if is_dragging else Image.Resampling.BILINEAR)
        
        draw = ImageDraw.Draw(resized_pil)
        
        # 1. Draw Saved Annotations
        for ann in self.annotations:
            rx, ry = ann['x'], ann['y']
            if self.mirror_horizontal:
                rx = 1.0 - rx
            if self.mirror_vertical:
                ry = 1.0 - ry
                
            ax = int(rx * new_w)
            ay = int(ry * new_h)
            
            # Larger crosshair radius as requested
            self.draw_star_crosshair(draw, ax, ay, size=24, width=2)
            draw.text((ax + 26, ay + 26), ann['text'], fill=self.green_bright)
            
        # 2. Draw Temporary Target marker
        if self.temp_marker:
            rx, ry = self.temp_marker['ratio_x'], self.temp_marker['ratio_y']
            if self.mirror_horizontal:
                rx = 1.0 - rx
            if self.mirror_vertical:
                ry = 1.0 - ry
                
            tx = int(rx * new_w)
            ty = int(ry * new_h)
            
            # Larger crosshair radius
            self.draw_star_crosshair(draw, tx, ty, size=32, width=2)
            draw.text((tx + 34, ty + 34), f"Target: RA={self.temp_marker['ra']}\nDEC={self.temp_marker['dec']}", fill="#f43f5e")
            
        self.tk_image = ImageTk.PhotoImage(resized_pil)
        
        self.canvas.delete("all")
        self.canvas.create_image(self.img_offset_x, self.img_offset_y, anchor="nw", image=self.tk_image)
        
        # Configure scrollregion on the Canvas and dynamically calculate scrollbar thumb positions
        self.canvas.config(scrollregion=(min(0, self.img_offset_x), min(0, self.img_offset_y), 
                                         max(canvas_w, self.img_offset_x + new_w), max(canvas_h, self.img_offset_y + new_h)))
        
        # Manually compute scrollbar fraction views
        if new_w > canvas_w:
            left_frac = max(0.0, min(1.0, -self.pan_x / new_w))
            right_frac = max(0.0, min(1.0, (canvas_w - self.pan_x) / new_w))
            self.hbar.set(left_frac, right_frac)
        else:
            self.hbar.set(0.0, 1.0)
            
        if new_h > canvas_h:
            top_frac = max(0.0, min(1.0, -self.pan_y / new_h))
            bottom_frac = max(0.0, min(1.0, (canvas_h - self.pan_y) / new_h))
            self.vbar.set(top_frac, bottom_frac)
        else:
            self.vbar.set(0.0, 1.0)
        
        if self.crop_mode and self.crop_rect_id:
            self.canvas.delete(self.crop_rect_id)
            self.crop_rect_id = None

    def on_canvas_motion(self, event):
        if self.fits_data is None:
            return
            
        canvas_x = event.x - self.img_offset_x
        canvas_y = event.y - self.img_offset_y
        
        w_px = int(canvas_x / self.img_scale_ratio)
        h_px = int(canvas_y / self.img_scale_ratio)
        
        h_orig, w_orig = self.fits_data.shape[-2:]
        
        if self.mirror_horizontal:
            w_px = (w_orig - 1) - w_px
        if self.mirror_vertical:
            h_px = (h_orig - 1) - h_px
            
        if 0 <= w_px < w_orig and 0 <= h_px < h_orig:
            self.lbl_cursor_coords.config(text=f"Cursor: X={w_px}, Y={h_px}")
            
            if self.wcs:
                try:
                    sky_coord = self.wcs.pixel_to_world(w_px, h_px)
                    
                    # Native epoch of the WCS (Standard astropy coordinates default to FK5/J2000 or similar based on WCS structure)
                    # Let's explicitly define a J2000 FK5 coordinate frame and precess to observation date (JNow)
                    coord_j2000 = sky_coord.transform_to(FK5(equinox='J2000'))
                    coord_jnow = sky_coord.transform_to(FK5(equinox=self.observation_time))
                    
                    j2000_ra = coord_j2000.ra.to_string(unit="hour", sep="hms", precision=2)
                    j2000_dec = coord_j2000.dec.to_string(unit="degree", sep="dms", precision=2)
                    
                    jnow_ra = coord_jnow.ra.to_string(unit="hour", sep="hms", precision=2)
                    jnow_dec = coord_jnow.dec.to_string(unit="degree", sep="dms", precision=2)
                    
                    self.lbl_wcs_j2000.config(text=f"J2000: RA={j2000_ra}, DEC={j2000_dec}")
                    self.lbl_wcs_jnow.config(text=f"JNow:  RA={jnow_ra}, DEC={jnow_dec}")
                except Exception as e:
                    self.lbl_wcs_j2000.config(text="J2000: Error converting")
                    self.lbl_wcs_jnow.config(text="JNow: Error converting")
            else:
                self.lbl_wcs_j2000.config(text="J2000: No WCS coords")
                self.lbl_wcs_jnow.config(text="JNow: No WCS coords")
        else:
            self.lbl_cursor_coords.config(text="Cursor: Out of bounds")
            self.lbl_wcs_j2000.config(text="J2000: N/A")
            self.lbl_wcs_jnow.config(text="JNow: N/A")

    def toggle_crop_mode(self):
        self.crop_mode = not self.crop_mode
        if self.crop_mode:
            self.annotation_mode = False
            self.balance_mode = "None"
            self.btn_bal_sky.config(bg="#374151")
            self.btn_bal_star.config(bg="#374151")
            self.btn_annotate.config(text="Annotate Mode: Off", bg="#374151")
            self.btn_crop.config(text="Crop Mode: ON", bg="#e11d48")
            self.canvas.config(cursor="cross")
        else:
            self.btn_crop.config(text="Crop Mode: Off", bg="#374151")
            self.canvas.config(cursor="")
            if self.crop_rect_id:
                self.canvas.delete(self.crop_rect_id)
                self.crop_rect_id = None

    def toggle_annotation_mode(self):
        self.annotation_mode = not self.annotation_mode
        if self.annotation_mode:
            self.crop_mode = False
            self.balance_mode = "None"
            self.btn_bal_sky.config(bg="#374151")
            self.btn_bal_star.config(bg="#374151")
            self.btn_crop.config(text="Crop Mode: Off", bg="#374151")
            self.btn_annotate.config(text="Annotate Mode: ON", bg="#10b981")
            self.canvas.config(cursor="tcross")
        else:
            self.btn_annotate.config(text="Annotate Mode: Off", bg="#374151")
            self.canvas.config(cursor="")

    def on_canvas_click_press(self, event):
        if self.fits_data is None:
            return
            
        canvas_x = event.x - self.img_offset_x
        canvas_y = event.y - self.img_offset_y
        
        img_to_draw = self.processed_img_preview
        if img_to_draw is None:
            return
            
        h, w = img_to_draw.shape[:2]
        
        # Color Balance Sampler Point Click Logic
        if self.balance_mode != "None" and 0 <= canvas_x < self.img_scale_ratio * w and 0 <= canvas_y < self.img_scale_ratio * h:
            px_x = int(canvas_x / self.img_scale_ratio)
            px_y = int(canvas_y / self.img_scale_ratio)
            
            h_orig, w_orig = self.debayered_cache.shape[:2]
            px_x = int(px_x * (w_orig / w))
            px_y = int(px_y * (h_orig / h))
            
            if self.mirror_horizontal:
                px_x = (w_orig - 1) - px_x
            if self.mirror_vertical:
                px_y = (h_orig - 1) - px_y
                
            # Sample a larger 10x10 pixel patch
            x_min, x_max = max(0, px_x - 5), min(w_orig, px_x + 5)
            y_min, y_max = max(0, px_y - 5), min(h_orig, px_y + 5)
            patch = self.debayered_cache[y_min:y_max, x_min:x_max] # shape (H, W, 3)
            
            global_min = self.debayered_cache.min()
            global_max = self.debayered_cache.max()
            global_range = global_max - global_min if global_max > global_min else 1.0
            
            # Normalize patch values to 0..1 range using global bounds
            patch_norm = (patch - global_min) / global_range
            patch_norm = np.clip(patch_norm, 0.0, 1.0)
            
            self.push_state()
            
            if self.balance_mode == "Sky":
                # Sky background (black point): Exclude very bright pixels (e.g. stars > 50% above local patch mean)
                patch_lum = 0.2126 * patch_norm[:, :, 0] + 0.7152 * patch_norm[:, :, 1] + 0.0722 * patch_norm[:, :, 2]
                mean_lum = np.mean(patch_lum)
                
                # Mask out pixels where local luminance is greater than 1.5 * mean_lum
                valid_mask = patch_lum <= (1.5 * mean_lum)
                if not np.any(valid_mask):
                    valid_mask = np.ones_like(patch_lum, dtype=bool)
                    
                # Calculate the average values of the channels
                sampled_rgb = np.mean(patch_norm[valid_mask], axis=0)
                
                # Proportional calculation on a neutral gray: 
                # instead of storing individual color channel shifts, we set a neutral gray level
                # by computing the average luminance of the sampled sky and assigning it evenly to R, G, B.
                sky_gray_val = 0.2126 * sampled_rgb[0] + 0.7152 * sampled_rgb[1] + 0.0722 * sampled_rgb[2]
                
                # Subtract target scuro (e.g., target background stays at 0.03 / 3%)
                dark_gray_target = 0.03
                offset_val = sky_gray_val - dark_gray_target
                offset_val = max(0.0, min(offset_val, 0.95))
                
                # Align R, G, B to exactly the same neutral gray offset
                offset_rgb = [offset_val, offset_val, offset_val]
                
                self.sky_sample_rgb = offset_rgb
                messagebox.showinfo("Color Balance", 
                                    f"Sky Background calibrated (Neutral Gray offset subtracted):\n"
                                    f"Red={offset_rgb[0]:.4f}, Green={offset_rgb[1]:.4f}, Blue={offset_rgb[2]:.4f}\n"
                                    f"(Background calibrated to ~3% neutral dark gray)")
                                    
            elif self.balance_mode == "Star":
                # White Star: Take brightest pixels in the patch (e.g. top 25% brightest in luminance)
                patch_lum = 0.2126 * patch_norm[:, :, 0] + 0.7152 * patch_norm[:, :, 1] + 0.0722 * patch_norm[:, :, 2]
                q75 = np.percentile(patch_lum, 75)
                star_mask = patch_lum >= q75
                
                sampled_rgb = np.mean(patch_norm[star_mask], axis=0)
                
                # Apply sky background offset (if any) prior to finding white gain target
                sky_offset = np.array(self.sky_sample_rgb if self.sky_sample_rgb is not None else [0.0, 0.0, 0.0])
                net_rgb = np.clip(sampled_rgb - sky_offset, 0.0001, 1.0)
                
                # Determine scaling target (max net channel value)
                max_ch = max(net_rgb)
                if max_ch <= 0.0001: max_ch = 1.0
                
                # We want: net_rgb * gain = target
                # Let's align channels. The channel gains are max_ch / net_rgb.
                gains = max_ch / net_rgb
                
                # Retain 15% warmth bias on red channel if red has higher signal (gain < green/blue)
                if gains[0] < gains[1] and gains[0] < gains[2]:
                    gains[0] = gains[0] + (1.0 - gains[0]) * 0.15
                    
                # Calculate calibrated white points: white_point = black_offset + (net_rgb * gain)
                # Since balanced values are: (pixel_val - black_offset) / (white_point - black_offset) = normalized_val
                # To align white_gain, we define star_sample_rgb = black_offset + (net_rgb * gains)
                star_calibrated_rgb = list(sky_offset + (net_rgb * gains))
                star_calibrated_rgb = list(np.clip(star_calibrated_rgb, 0.001, 1.0))
                
                self.star_sample_rgb = star_calibrated_rgb
                
                messagebox.showinfo("Color Balance", 
                                    f"White Star calibrated (target values):\n"
                                    f"Red={star_calibrated_rgb[0]:.4f}, Green={star_calibrated_rgb[1]:.4f}, Blue={star_calibrated_rgb[2]:.4f}\n"
                                    f"(Star warmth tone preserved slightly)")
                
            self.balance_mode = "None"
            self.canvas.config(cursor="")
            self.btn_bal_sky.config(bg="#374151")
            self.btn_bal_star.config(bg="#374151")
            self.process_and_update(is_dragging=False)
            return

        if 0 <= canvas_x < self.img_scale_ratio * w and 0 <= canvas_y < self.img_scale_ratio * h:
            if self.crop_mode:
                self.crop_start = (event.x, event.y)
            elif self.annotation_mode:
                import tkinter.simpledialog as sd
                text = sd.askstring("Add Annotation", "Star description:", parent=self.root)
                if text is not None and text.strip() != "":
                    self.push_state()
                    ratio_x = canvas_x / (self.img_scale_ratio * w)
                    ratio_y = canvas_y / (self.img_scale_ratio * h)
                    
                    if self.mirror_horizontal:
                        ratio_x = 1.0 - ratio_x
                    if self.mirror_vertical:
                        ratio_y = 1.0 - ratio_y
                        
                    final_text = text.strip()
                    
                    # Resolve J2000 coordinates if WCS is available
                    if self.wcs:
                        try:
                            # Map canonical ratios back to original pixel coordinates
                            h_orig, w_orig = self.fits_data.shape[-2:]
                            orig_px_x = int(ratio_x * w_orig)
                            orig_px_y = int(ratio_y * h_orig)
                            
                            sky_coord = self.wcs.pixel_to_world(orig_px_x, orig_px_y)
                            coord_j2000 = sky_coord.transform_to(FK5(equinox='J2000'))
                            
                            ra_str = coord_j2000.ra.to_string(unit="hour", sep="hms", precision=1)
                            dec_str = coord_j2000.dec.to_string(unit="degree", sep="dms", precision=1)
                            
                            append_coords = messagebox.askyesno("Include Coordinates", 
                                                                f"Do you want to append J2000 RA/DEC coordinates to this annotation?\n\n({ra_str}, {dec_str})", 
                                                                parent=self.root)
                            if append_coords:
                                final_text = f"{final_text} (RA:{ra_str} DEC:{dec_str})"
                        except Exception:
                            pass
                        
                    self.annotations.append({
                        'x': ratio_x,
                        'y': ratio_y,
                        'text': final_text
                    })
                    self.render_canvas(is_dragging=False)

    def on_canvas_drag(self, event):
        if self.crop_mode and self.crop_start:
            if self.crop_rect_id:
                self.canvas.delete(self.crop_rect_id)
            self.crop_rect_id = self.canvas.create_rectangle(self.crop_start[0], self.crop_start[1], event.x, event.y, outline="red", width=2)

    def on_canvas_release(self, event):
        if self.crop_mode and self.crop_start:
            x1, y1 = self.crop_start
            x2, y2 = event.x, event.y
            self.crop_start = None
            
            if self.crop_rect_id:
                self.canvas.delete(self.crop_rect_id)
                self.crop_rect_id = None
                
            img_to_draw = self.processed_img_preview
            if img_to_draw is None:
                return
                
            h, w = img_to_draw.shape[:2]
            
            img_x1 = max(0, x1 - self.img_offset_x)
            img_y1 = max(0, y1 - self.img_offset_y)
            img_x2 = min(int(w * self.img_scale_ratio), x2 - self.img_offset_x)
            img_y2 = min(int(h * self.img_scale_ratio), y2 - self.img_offset_y)
            
            w_px1 = int(img_x1 / self.img_scale_ratio)
            h_px1 = int(img_y1 / self.img_scale_ratio)
            w_px2 = int(img_x2 / self.img_scale_ratio)
            h_px2 = int(img_y2 / self.img_scale_ratio)
            
            h_orig, w_orig = self.fits_data.shape[-2:]
            
            if self.mirror_horizontal:
                w_px1 = (w_orig - 1) - w_px1
                w_px2 = (w_orig - 1) - w_px2
            if self.mirror_vertical:
                h_px1 = (h_orig - 1) - h_px1
                h_px2 = (h_orig - 1) - h_px2
                
            start_x, end_x = min(w_px1, w_px2), max(w_px1, w_px2)
            start_y, end_y = min(h_px1, h_px2), max(h_px1, h_px2)
            
            if (end_x - start_x) > 5 and (end_y - start_y) > 5:
                self.push_state()
                if self.fits_data.ndim == 3:
                    self.fits_data = self.fits_data[:, start_y:end_y, start_x:end_x]
                else:
                    self.fits_data = self.fits_data[start_y:end_y, start_x:end_x]
                    
                self.debayered_cache = None
                self.temp_marker = None
                self.sky_sample_rgb = None
                self.star_sample_rgb = None
                
                new_annotations = []
                for ann in self.annotations:
                    px = ann['x'] * w_orig
                    py = ann['y'] * h_orig
                    if start_x <= px < end_x and start_y <= py < end_y:
                        new_annotations.append({
                            'x': (px - start_x) / (end_x - start_x),
                            'y': (py - start_y) / (end_y - start_y),
                            'text': ann['text']
                        })
                self.annotations = new_annotations
                
                self.toggle_crop_mode()
                self.process_and_update(is_dragging=False)

    def mark_radec_input(self):
        if self.fits_data is None:
            messagebox.showerror("Error", "Load a FITS file first.")
            return
            
        if self.wcs is None:
            messagebox.showerror("Error", "No valid WCS headers found in FITS to resolve RA/DEC.")
            return
            
        import tkinter.simpledialog as sd
        epoch_choice = messagebox.askyesnocancel("Select Epoch", "Is your input coordinate in J2000?\n\n(Yes = J2000, No = JNow, Cancel = Abort)")
        if epoch_choice is None: return
        
        ra_in = sd.askstring("Mark RA/DEC Target", "Right Ascension (e.g. 16h41m41s or 250.42 degrees):", parent=self.root)
        if not ra_in: return
        dec_in = sd.askstring("Mark RA/DEC Target", "Declination (e.g. +36d27m40s or 36.46 degrees):", parent=self.root)
        if not dec_in: return
        
        try:
            if 'h' in ra_in or 'm' in ra_in or 's' in ra_in:
                input_coord = SkyCoord(ra=ra_in, dec=dec_in, frame='fk5')
            else:
                input_coord = SkyCoord(ra=float(ra_in), dec=float(dec_in), unit=(u.deg, u.deg), frame='fk5')
            
            header_epoch = self.fits_header.get('EQUINOX', 2000.0)
            
            if epoch_choice is True:
                if header_epoch == 2000.0:
                    target_wcs_coord = input_coord
                else:
                    target_wcs_coord = input_coord.transform_to(FK5(equinox=self.observation_time))
            else:
                if header_epoch == 2000.0:
                    target_wcs_coord = input_coord.transform_to(FK5(equinox='J2000'))
                else:
                    target_wcs_coord = input_coord
            
            px_x, px_y = self.wcs.world_to_pixel(target_wcs_coord)
            px_x = int(round(float(px_x)))
            px_y = int(round(float(px_y)))
            
            h_orig, w_orig = self.fits_data.shape[-2:]
            
            if 0 <= px_x < w_orig and 0 <= px_y < h_orig:
                if epoch_choice is True:
                    coord_j2000 = input_coord
                    coord_jnow = input_coord.transform_to(FK5(equinox=self.observation_time))
                else:
                    coord_jnow = input_coord
                    coord_j2000 = input_coord.transform_to(FK5(equinox='J2000'))
                
                self.temp_marker = {
                    'ra': coord_j2000.ra.to_string(unit="hour", sep="hms", precision=2),
                    'dec': coord_j2000.dec.to_string(unit="degree", sep="dms", precision=2),
                    'ratio_x': px_x / w_orig,
                    'ratio_y': px_y / h_orig
                }
                
                canvas_w = self.canvas.winfo_width()
                canvas_h = self.canvas.winfo_height()
                fit_ratio = min(canvas_w / w_orig, canvas_h / h_orig)
                total_scale = fit_ratio * self.zoom_level
                
                target_draw_x = (w_orig - 1 - px_x) if self.mirror_horizontal else px_x
                target_draw_y = (h_orig - 1 - px_y) if self.mirror_vertical else px_y
                
                self.pan_x = (canvas_w / 2) - (target_draw_x * total_scale) - ((canvas_w - (w_orig * total_scale)) / 2)
                self.pan_y = (canvas_h / 2) - (target_draw_y * total_scale) - ((canvas_h - (h_orig * total_scale)) / 2)
                
                self.render_canvas(is_dragging=False)
                messagebox.showinfo("Target Marked", f"Star marked at pixels X={px_x}, Y={px_y}.\nCentering viewport.")
            else:
                messagebox.showerror("Out of bounds", f"The coordinates resolved to pixel (X={px_x}, Y={px_y}) which is out of image bounds.")
                
        except Exception as e:
            messagebox.showerror("Coordinate Error", f"Failed to parse or map coordinates:\n{str(e)}")

    def on_window_resize(self, event):
        self.render_canvas(is_dragging=False)

    def export_image(self):
        if self.processed_img_full is None:
            messagebox.showerror("Error", "No image to export.")
            return
            
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[
            ("PNG image", "*.png"),
            ("JPEG image", "*.jpg;*.jpeg"),
            ("TIFF image", "*.tiff;*.tif"),
            ("All files", "*.*")
        ])
        
        if not file_path:
            return
            
        try:
            pil_export = Image.fromarray(self.processed_img_full)
            draw = ImageDraw.Draw(pil_export)
            
            h, w = self.processed_img_full.shape[:2]
            
            # Load a high-res TTF font if possible, fallback to default scaled
            font_size = int(max(w, h) * 0.015)
            font_size = max(24, min(font_size, 96))
            
            font = None
            try:
                from PIL import ImageFont
                # Try loading standard system fonts
                for font_name in ["arial.ttf", "DejaVuSans.ttf", "tahoma.ttf", "cour.ttf"]:
                    try:
                        font = ImageFont.truetype(font_name, size=font_size)
                        break
                    except IOError:
                        continue
            except Exception:
                pass
                
            # 1. Draw Saved Annotations
            for ann in self.annotations:
                rx, ry = ann['x'], ann['y']
                if self.mirror_horizontal:
                    rx = 1.0 - rx
                if self.mirror_vertical:
                    ry = 1.0 - ry
                    
                ax = int(rx * w)
                ay = int(ry * h)
                
                cross_size = int(max(w, h) * 0.025)
                cross_size = max(40, min(cross_size, 160))
                
                # Draw thick lines for full resolution exports (width=5)
                self.draw_star_crosshair(draw, ax, ay, size=cross_size, width=5)
                
                text_offset_x = int(cross_size * 0.8)
                text_offset_y = int(cross_size * 0.8)
                
                # Draw thick readable text
                if font:
                    draw.text((ax + text_offset_x, ay + text_offset_y), ann['text'], fill=self.green_bright, font=font)
                else:
                    # Fallback thick text simulation using offset shadow if font is None
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            draw.text((ax + text_offset_x + dx, ay + text_offset_y + dy), ann['text'], fill=self.green_bright)
                
            # 2. Draw Temporary Target marker if present
            if self.temp_marker:
                rx, ry = self.temp_marker['ratio_x'], self.temp_marker['ratio_y']
                if self.mirror_horizontal:
                    rx = 1.0 - rx
                if self.mirror_vertical:
                    ry = 1.0 - ry
                    
                tx = int(rx * w)
                ty = int(ry * h)
                
                cross_size = int(max(w, h) * 0.03)
                cross_size = max(50, min(cross_size, 200))
                
                self.draw_star_crosshair(draw, tx, ty, size=cross_size, width=6)
                
                text_offset_x = int(cross_size * 0.8)
                text_offset_y = int(cross_size * 0.8)
                
                text_content = f"Target: RA={self.temp_marker['ra']}\nDEC={self.temp_marker['dec']}"
                
                if font:
                    draw.text((tx + text_offset_x, ty + text_offset_y), text_content, fill=self.green_bright, font=font)
                else:
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            draw.text((tx + text_offset_x + dx, ty + text_offset_y + dy), text_content, fill=self.green_bright)
                
            pil_export.save(file_path)
            messagebox.showinfo("Success", f"Image exported successfully to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export image:\n{str(e)}")


try:
    from astropy.wcs.utils import proj_plane_pixel_scales
except ImportError:
    pass

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = FitsManagerApp(root)
    root.update()
    app.render_canvas(is_dragging=False)
    root.mainloop()
