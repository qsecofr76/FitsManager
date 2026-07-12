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
import time

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
        
        self.raw_hist_r = None
        self.raw_hist_g = None
        self.raw_hist_b = None
        self.raw_hist_l = None
        
        self.hist_r = None
        self.hist_g = None
        self.hist_b = None
        self.hist_l = None
        self.log_scale = True
        
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
        if max(h, w) > 512:
            ratio = 512 / max(h, w)
            sample = cv2.resize(base_img, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_NEAREST)
        else:
            sample = base_img
            
        d_min, d_max = sample.min(), sample.max()
        if d_max > d_min:
            norm_8u = ((sample - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
        else:
            norm_8u = np.zeros(sample.shape, dtype=np.uint8)
            
        self.raw_hist_r = cv2.calcHist([norm_8u], [0], None, [256], [0, 256])
        self.raw_hist_g = cv2.calcHist([norm_8u], [1], None, [256], [0, 256])
        self.raw_hist_b = cv2.calcHist([norm_8u], [2], None, [256], [0, 256])
        
        lum_8u = (0.2126 * norm_8u[:,:,0] + 0.7152 * norm_8u[:,:,1] + 0.0722 * norm_8u[:,:,2]).astype(np.uint8)
        self.raw_hist_l = cv2.calcHist([lum_8u], [0], None, [256], [0, 256])
        
        self.compute_scaled_histograms()
        
    def compute_scaled_histograms(self):
        if self.raw_hist_r is None:
            return
            
        if self.log_scale:
            hr = np.log1p(self.raw_hist_r)
            hg = np.log1p(self.raw_hist_g)
            hb = np.log1p(self.raw_hist_b)
            hl = np.log1p(self.raw_hist_l)
        else:
            hr = self.raw_hist_r.copy()
            hg = self.raw_hist_g.copy()
            hb = self.raw_hist_b.copy()
            hl = self.raw_hist_l.copy()
            
        max_val = max(hr.max(), hg.max(), hb.max(), hl.max())
        if max_val > 0:
            self.hist_r = (hr / max_val) * 150.0
            self.hist_g = (hg / max_val) * 150.0
            self.hist_b = (hb / max_val) * 150.0
            self.hist_l = (hl / max_val) * 150.0
        else:
            self.hist_r = np.zeros_like(hr)
            self.hist_g = np.zeros_like(hg)
            self.hist_b = np.zeros_like(hb)
            self.hist_l = np.zeros_like(hl)
            
        self.redraw()
        
    def redraw(self):
        self.delete("hist")
        self.delete("curve")
        self.delete("point")
        self.draw_grid()
        
        w = float(self.winfo_width()) if self.winfo_width() > 10 else 220.0
        h = float(self.winfo_height()) if self.winfo_height() > 10 else 220.0
        
        # 1. Draw Histograms in Siril style (transparent filled overlapping polygons)
        if self.hist_r is not None:
            hist_img = Image.new("RGBA", (int(w), int(h)), (0, 0, 0, 0))
            draw = ImageDraw.Draw(hist_img)
            
            # Semi-transparent fill colors for R, G, B channels
            channels = [
                (self.hist_r, (239, 68, 68, 70)),   # Red
                (self.hist_g, (16, 185, 129, 70)),  # Green
                (self.hist_b, (59, 130, 246, 70))   # Blue
            ]
            
            for channel_hist, fill_color in channels:
                poly_pts = [(0, h)]
                for vx in range(0, 256):
                    val = channel_hist[vx][0]
                    cx = (vx / 255.0) * w
                    cy = h - (val / 150.0) * (h * 0.85)
                    poly_pts.append((cx, cy))
                poly_pts.append((w, h))
                
                draw.polygon(poly_pts, fill=fill_color)
                
            # Sharp outlines for R, G, B channels
            outlines = [
                (self.hist_r, (239, 68, 68, 200)),
                (self.hist_g, (16, 185, 129, 200)),
                (self.hist_b, (59, 130, 246, 200))
            ]
            for channel_hist, outline_color in outlines:
                pts = []
                for vx in range(0, 256):
                    val = channel_hist[vx][0]
                    cx = (vx / 255.0) * w
                    cy = h - (val / 150.0) * (h * 0.85)
                    pts.append((cx, cy))
                for i in range(len(pts) - 1):
                    draw.line([pts[i], pts[i+1]], fill=outline_color, width=1)
                    
            # Draw Luminance/Gray as a subtle gray outline
            if self.hist_l is not None:
                pts_l = []
                for vx in range(0, 256):
                    val = self.hist_l[vx][0]
                    cx = (vx / 255.0) * w
                    cy = h - (val / 150.0) * (h * 0.85)
                    pts_l.append((cx, cy))
                for i in range(len(pts_l) - 1):
                    draw.line([pts_l[i], pts_l[i+1]], fill=(229, 231, 235, 120), width=1)
                    
            self.hist_photo = ImageTk.PhotoImage(hist_img)
            self.create_image(0, 0, image=self.hist_photo, anchor="nw", tags="hist")
        
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


class ColorIndexDialog(tk.Toplevel):
    def __init__(self, parent, is_gaia=True, default_val=0.8):
        super().__init__(parent)
        self.transient(parent)
        self.grab_set()
        self.title("Target Color Selection")
        self.geometry("450x260")
        self.resizable(False, False)
        
        self.result = default_val
        self.is_gaia = is_gaia
        self.options = []
        
        # Style match
        self.configure(bg="#1e293b")
        
        lbl_title = tk.Label(self, text="Target Object Type / Color Correction", bg="#1e293b", fg="#f8fafc", font=("Segoe UI", 11, "bold"))
        lbl_title.pack(padx=15, pady=(15, 5), anchor="w")
        
        desc_text = (
            "Because digital camera sensors are color-sensitive, calculating accurate\n"
            "magnitudes requires a small correction based on the object's color.\n"
            "Please choose the object type that best matches your target:"
        )
        lbl_desc = tk.Label(self, text=desc_text, bg="#1e293b", fg="#94a3b8", font=("Segoe UI", 9), justify="left")
        lbl_desc.pack(padx=15, pady=(0, 15), anchor="w")
        
        # Options list
        if is_gaia:
            self.options = [
                ("Yellow Dwarf (Sun-like) / Asteroid", 0.8),
                ("Nova (Typical / Early Outburst)", 0.6),
                ("Supernova (Typical Ia / Near Max)", 0.2),
                ("Red Dwarf / M-Class Star", 2.0),
                ("Orange Dwarf / K-Class Star", 1.2),
                ("White Main Sequence / A-Class Star", 0.0),
                ("Hot Blue Giant / O-Class Star", -0.3),
                ("Custom Value", None)
            ]
        else:
            self.options = [
                ("Yellow Dwarf (Sun-like) / Asteroid", 0.6),
                ("Nova (Typical / Early Outburst)", 0.4),
                ("Supernova (Typical Ia / Near Max)", 0.0),
                ("Red Dwarf / M-Class Star", 1.4),
                ("Orange Dwarf / K-Class Star", 1.0),
                ("White Main Sequence / A-Class Star", 0.0),
                ("Hot Blue Giant / O-Class Star", -0.3),
                ("Custom Value", None)
            ]
            
        opt_names = [opt[0] for opt in self.options]
        
        frame_input = tk.Frame(self, bg="#1e293b")
        frame_input.pack(fill="x", padx=15, pady=5)
        
        lbl_type = tk.Label(frame_input, text="Object Type:", bg="#1e293b", fg="#f8fafc", font=("Segoe UI", 9, "bold"))
        lbl_type.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.combo = ttk.Combobox(frame_input, values=opt_names, state="readonly", width=35)
        self.combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.combo.set(opt_names[0])
        self.combo.bind("<<ComboboxSelected>>", self.on_combo_change)
        
        lbl_val = tk.Label(frame_input, text="Color Index Value:", bg="#1e293b", fg="#f8fafc", font=("Segoe UI", 9, "bold"))
        lbl_val.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        
        self.entry_val = tk.Entry(frame_input, bg="#334155", fg="white", insertbackground="white", bd=1, width=12, font=("Consolas", 10))
        self.entry_val.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.entry_val.insert(0, str(self.options[0][1]))
        self.entry_val.config(state="disabled")
        
        # Buttons
        frame_btns = tk.Frame(self, bg="#1e293b")
        frame_btns.pack(fill="x", padx=15, pady=20)
        
        btn_ok = tk.Button(frame_btns, text="OK", command=self.on_ok, bg="#10b981", fg="white", font=("Segoe UI", 9, "bold"), bd=0, width=10, pady=4)
        btn_ok.pack(side="right", padx=5)
        
        btn_cancel = tk.Button(frame_btns, text="Cancel", command=self.destroy, bg="#334155", fg="white", font=("Segoe UI", 9), bd=0, width=10, pady=4)
        btn_cancel.pack(side="right", padx=5)
        
        # Center in parent
        self.wait_visibility()
        parent_geom = parent.geometry()
        try:
            parts = parent_geom.split('+')
            px = int(parts[1])
            py = int(parts[2])
            pw, ph = map(int, parts[0].split('x'))
            cx = px + (pw - 450) // 2
            cy = py + (ph - 260) // 2
            self.geometry(f"450x260+{cx}+{cy}")
        except Exception:
            pass
            
        self.wait_window(self)
        
    def on_combo_change(self, event):
        sel_idx = self.combo.current()
        val = self.options[sel_idx][1]
        if val is not None:
            self.entry_val.config(state="normal")
            self.entry_val.delete(0, tk.END)
            self.entry_val.insert(0, str(val))
            self.entry_val.config(state="disabled")
        else:
            self.entry_val.config(state="normal")
            self.entry_val.delete(0, tk.END)
            self.entry_val.focus()
            
    def on_ok(self):
        try:
            self.result = float(self.entry_val.get())
            self.destroy()
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid decimal number for the color index.", parent=self)


class FitsManagerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FitsManager")
        self.root.geometry("1400x870")
        
        # Window icon configuration
        import os
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_icon.png")
        if os.path.exists(icon_path):
            try:
                self.icon_img = ImageTk.PhotoImage(Image.open(icon_path))
                self.root.iconphoto(True, self.icon_img)
            except Exception:
                pass
        
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
        self.wcs_saved_to_disk = False
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
        self.src_left_int = 0
        self.src_top_int = 0
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
        
        # Catalog and Photometry attributes
        self.catalog_stars = []
        self.show_catalog_stars = tk.BooleanVar(value=True)
        self.asteroid_objects = []
        self.show_asteroids = tk.BooleanVar(value=True)
        self.calibration_mode = False
        self.measurement_mode = False
        self.photometry_mode = False
        self.gaia_search_mode = False
        self.gaia_search_var = tk.BooleanVar(value=False)
        self.show_annotations = tk.BooleanVar(value=True)
        self.show_limiting_mag_stars = tk.BooleanVar(value=True)
        self.visual_crop_box = None
        self.calib_stars = []
        self.show_transients = tk.BooleanVar(value=True)
        self.transient_objects = []
        
        # Cleanup caches at startup if files count exceeds 300 per directory
        self.cleanup_caches_startup()
        
        # Settings loader
        self.settings_path = "settings.json"
        self.settings = {}
        self.load_settings()
        
        # Astrometry jobs loader
        self.jobs_path = "astrometry_jobs.json"
        self.jobs = {}
        self.load_jobs()
        
        # DSS Reference Background variables
        self.loaded_dss_tiles = []  # List of loaded DSS tiles: [{'data': dss_data, 'wcs': dss_wcs, 'ra': ra, 'dec': dec, 'width_arcmin': w}]
        self.dss_blend_ratio = tk.DoubleVar(value=0.0)
        
        # UI controls vars
        self.bayer_pattern = tk.StringVar(value="None")
        self.channel_selection = tk.StringVar(value="RGB/Luminance")
        self.var_invert = tk.BooleanVar(value=False)
        self.var_grayscale = tk.BooleanVar(value=False)
        
        # Toolbar buttons reference placeholders
        self.btn_bal_sky = None
        self.btn_bal_star = None
        self.btn_mirror_h = None
        self.btn_mirror_v = None
        
        # Build UI layout
        self.create_widgets()
        self.bind_events()
        
        # Register Drag & Drop
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind("<<Drop>>", self.handle_file_drop)
        
    def create_widgets(self):
        # 1. Create Top Menu Bar
        self.menu_bar = tk.Menu(self.root)
        self.root.config(menu=self.menu_bar)
        
        # File dropdown
        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        file_menu.add_command(label="Open FITS...", command=self.load_fits)
        file_menu.add_command(label="Export Image...", command=self.export_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        self.menu_bar.add_cascade(label="File", menu=file_menu)
        
        # View dropdown
        view_menu = tk.Menu(self.menu_bar, tearoff=0)
        view_menu.add_command(label="Mirror Horizontal", command=self.toggle_mirror_h)
        view_menu.add_command(label="Mirror Vertical", command=self.toggle_mirror_v)
        view_menu.add_separator()
        view_menu.add_checkbutton(label="Grayscale (B&W)", variable=self.var_grayscale, command=lambda: self.process_and_update(is_dragging=False))
        view_menu.add_checkbutton(label="Invert Colors (Negative)", variable=self.var_invert, command=lambda: self.process_and_update(is_dragging=False))
        view_menu.add_separator()
        view_menu.add_command(label="Reset Zoom & Pan", command=self.reset_view)
        self.menu_bar.add_cascade(label="View", menu=view_menu)
        
        # Color Calibration dropdown
        color_menu = tk.Menu(self.menu_bar, tearoff=0)
        color_menu.add_command(label="Auto Adaptation (Stretch)", command=self.apply_autostretch)
        color_menu.add_command(label="Sky Background Calibration...", command=self.enable_sky_balance)
        color_menu.add_command(label="White Star Calibration...", command=self.enable_star_balance)
        color_menu.add_command(label="Remove Background Gradient", command=self.remove_background_gradient)
        color_menu.add_separator()
        color_menu.add_command(label="Reset Colors & Curves", command=self.reset_color_manipulation)
        self.menu_bar.add_cascade(label="Color Calibration", menu=color_menu)
        
        # Astrometry dropdown
        astrom_menu = tk.Menu(self.menu_bar, tearoff=0)
        
        self.ann_menu = tk.Menu(astrom_menu, tearoff=0)
        self.ann_menu.add_command(label="Save Annotations to FITS", command=self.save_annotations_to_fits)
        self.ann_menu.add_command(label="Import Annotations from FITS", command=self.import_annotations_from_fits)
        
        astrom_menu.add_command(label="Mark Target RA/Dec...", command=self.mark_radec_input)
        astrom_menu.add_command(label="Clear Target/Annotations", command=self.clear_annotations)
        astrom_menu.add_checkbutton(label="Show Annotations", variable=self.show_annotations, command=lambda: self.render_canvas(is_dragging=False))
        astrom_menu.add_cascade(label="Annotations Manager", menu=self.ann_menu)
        astrom_menu.add_separator()
        astrom_menu.add_command(label="Query MPC Asteroids (Mag < 21)", command=self.query_mpc_asteroids)
        astrom_menu.add_checkbutton(label="Show Asteroids", variable=self.show_asteroids, command=lambda: self.render_canvas(is_dragging=False))
        astrom_menu.add_separator()
        astrom_menu.add_command(label="Download DSS Sky Background", command=self.download_dss_background)
        astrom_menu.add_command(label="Load PanStarrs background image", command=self.download_panstarrs_background)
        astrom_menu.add_separator()
        astrom_menu.add_command(label="Vizier star plate solver/astroalign (fast)", command=self.platesolve_vizier_astroalign)
        astrom_menu.add_separator()
        astrom_menu.add_command(label="Online Plate Solve (Nova Astrometry)", command=self.platesolve_nova_astrometry)
        astrom_menu.add_command(label="Check Online Solver Status", command=self.check_astrometry_job_status)
        astrom_menu.add_command(label="Configure Astrometry.net API Key...", command=self.configure_astrometry_api_key)
        astrom_menu.add_separator()
        astrom_menu.add_checkbutton(label="Gaia Star Search Mode", variable=self.gaia_search_var, command=self.toggle_gaia_search_mode)
        astrom_menu.add_separator()
        astrom_menu.add_command(label="Toggle Mark Calibration Stars", command=self.toggle_calibration_mode)
        astrom_menu.add_command(label="Toggle Measure Target Star Magnitude", command=self.toggle_measurement_mode)
        astrom_menu.add_command(label="Download Vizier Calibration Stars", command=self.download_catalog_stars)
        astrom_menu.add_checkbutton(label="Show Vizier Calibration Stars", variable=self.show_catalog_stars, command=lambda: self.render_canvas(is_dragging=False))
        astrom_menu.add_command(label="Auto-Calibrate Photometry", command=self.auto_calibrate_photometry)
        astrom_menu.add_command(label="Estimate Limiting Magnitude", command=self.estimate_limiting_magnitude)
        astrom_menu.add_checkbutton(label="Show Magnitude Limit Stars", variable=self.show_limiting_mag_stars, command=lambda: self.render_canvas(is_dragging=False))
        astrom_menu.add_command(label="Clear Calibration Stars", command=self.clear_calibration_stars)
        astrom_menu.add_separator()
        astrom_menu.add_command(label="Query TNS/Supernova/Nova", command=self.query_tns_transients)
        astrom_menu.add_checkbutton(label="Show Supernova/nova", variable=self.show_transients, command=lambda: self.render_canvas(is_dragging=False))
        self.menu_bar.add_cascade(label="Astrometry", menu=astrom_menu)

        # 2. Create Top Toolbar (Single row)
        toolbar_container = tk.Frame(self.root, bg=self.panel_color, bd=0)
        toolbar_container.pack(side="top", fill="x")
        
        toolbar_row = tk.Frame(toolbar_container, bg=self.panel_color, height=35)
        toolbar_row.pack(side="top", fill="x", padx=10, pady=4)
        
        btn_open = tk.Button(toolbar_row, text="Load FITS", command=self.load_fits, bg=self.accent_color, fg="white", font=("Segoe UI", 9, "bold"), bd=0, padx=12, pady=4)
        btn_open.pack(side="left", padx=(0, 10))
        
        btn_undo = tk.Button(toolbar_row, text="Undo", command=self.undo, bg="#374151", fg="white", font=("Segoe UI", 9), bd=0, padx=12, pady=4)
        btn_undo.pack(side="left", padx=3)
        
        btn_redo = tk.Button(toolbar_row, text="Redo", command=self.redo, bg="#374151", fg="white", font=("Segoe UI", 9), bd=0, padx=12, pady=4)
        btn_redo.pack(side="left", padx=3)
        
        btn_reset_zoom = tk.Button(toolbar_row, text="Reset View", command=self.reset_view, bg="#374151", fg="white", font=("Segoe UI", 9), bd=0, padx=12, pady=4)
        btn_reset_zoom.pack(side="left", padx=10)
        
        self.btn_crop = tk.Button(toolbar_row, text="Crop Mode: Off", command=self.toggle_crop_mode, bg="#374151", fg="white", font=("Segoe UI", 9), bd=0, padx=12, pady=4)
        self.btn_crop.pack(side="left", padx=3)
        
        self.btn_annotate = tk.Button(toolbar_row, text="Annotate Mode: Off", command=self.toggle_annotation_mode, bg="#374151", fg="white", font=("Segoe UI", 9), bd=0, padx=12, pady=4)
        self.btn_annotate.pack(side="left", padx=3)
        
        self.btn_calibration = tk.Button(toolbar_row, text="Mark Calib Stars: Off", command=self.toggle_calibration_mode, bg="#374151", fg="white", font=("Segoe UI", 9), bd=0, padx=12, pady=4)
        self.btn_calibration.pack(side="left", padx=3)
        
        self.btn_measurement = tk.Button(toolbar_row, text="Measure Target: Off", command=self.toggle_measurement_mode, bg="#374151", fg="white", font=("Segoe UI", 9), bd=0, padx=12, pady=4)
        self.btn_measurement.pack(side="left", padx=3)
        
        btn_export = tk.Button(toolbar_row, text="Export Image", command=self.export_image, bg="#10b981", fg="white", font=("Segoe UI", 10, "bold"), bd=0, padx=16, pady=5)
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
        canvas_left = tk.Canvas(left_panel, bg=self.panel_color, bd=0, highlightthickness=0, width=320)
        scrollbar_left = tk.Scrollbar(left_panel, orient="vertical", command=canvas_left.yview)
        scrollable_frame = tk.Frame(canvas_left, bg=self.panel_color, width=320)
        
        # Helper to propagate mouse wheel to the canvas_left
        def on_left_mousewheel(event):
            canvas_left.yview_scroll(int(-1 * (event.delta / 120)), "units")
            
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas_left.configure(scrollregion=canvas_left.bbox("all"))
        )
        canvas_id = canvas_left.create_window((0, 0), window=scrollable_frame, anchor="nw", width=320)
        canvas_left.configure(yscrollcommand=scrollbar_left.set)
        
        # Bind mousewheel events recursively or directly
        canvas_left.bind("<MouseWheel>", on_left_mousewheel)
        scrollable_frame.bind("<MouseWheel>", on_left_mousewheel)
        
        scrollbar_left.pack(side="right", fill="y")
        canvas_left.pack(side="left", fill="both", expand=True)
        
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
        
        self.btn_save_wcs_header = ttk.Button(coord_lf, text="Save WCS to FITS Header", command=self.save_wcs_to_fits_file)
        # We won't pack it initially; update_wcs_hint_visibility will manage it
        
        self.lbl_wcs_hint = tk.Label(coord_lf, text="⚠️ No WCS projection found\nClick here to Plate Solve (Vizier)", bg="black", fg="#facc15", font=("Segoe UI", 9, "bold", "underline"), bd=1, relief="solid", padx=5, pady=5, justify="center", cursor="hand2")
        self.lbl_wcs_hint.bind("<Button-1>", self.on_wcs_hint_click)
        
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
        
        self.chk_invert = ttk.Checkbutton(curves_lf, text="Negative Image (Invert)", variable=self.var_invert, command=lambda: self.process_and_update(is_dragging=False))
        self.chk_invert.pack(anchor="w", padx=5, pady=5)
        
        self.var_log_hist = tk.BooleanVar(value=True)
        self.chk_log_hist = ttk.Checkbutton(curves_lf, text="Logarithmic Histogram", variable=self.var_log_hist, command=self.on_log_hist_changed)
        self.chk_log_hist.pack(anchor="w", padx=5, pady=5)
        
        # Color Balance & Sliders (under curves_lf)
        sliders_lf = ttk.LabelFrame(scrollable_frame, text="Color Balance & Levels")
        sliders_lf.pack(fill="x", padx=10, pady=5)
        
        self.slider_red_offset = tk.Scale(sliders_lf, from_=-100, to=100, orient="horizontal", label="Red Shift", bg=self.panel_color, fg=self.text_color, highlightthickness=0, bd=0, command=lambda v: self.process_and_update(is_dragging=True))
        self.slider_red_offset.set(0)
        self.slider_red_offset.pack(fill="x", padx=5, pady=2)
        self.slider_red_offset.bind("<ButtonRelease-1>", lambda e: self.process_and_update(is_dragging=False))
        
        self.slider_green_offset = tk.Scale(sliders_lf, from_=-100, to=100, orient="horizontal", label="Green Shift", bg=self.panel_color, fg=self.text_color, highlightthickness=0, bd=0, command=lambda v: self.process_and_update(is_dragging=True))
        self.slider_green_offset.set(0)
        self.slider_green_offset.pack(fill="x", padx=5, pady=2)
        self.slider_green_offset.bind("<ButtonRelease-1>", lambda e: self.process_and_update(is_dragging=False))
        
        self.slider_blue_offset = tk.Scale(sliders_lf, from_=-100, to=100, orient="horizontal", label="Blue Shift", bg=self.panel_color, fg=self.text_color, highlightthickness=0, bd=0, command=lambda v: self.process_and_update(is_dragging=True))
        self.slider_blue_offset.set(0)
        self.slider_blue_offset.pack(fill="x", padx=5, pady=2)
        self.slider_blue_offset.bind("<ButtonRelease-1>", lambda e: self.process_and_update(is_dragging=False))
        
        self.slider_brightness = tk.Scale(sliders_lf, from_=-100, to=100, orient="horizontal", label="Brightness", bg=self.panel_color, fg=self.text_color, highlightthickness=0, bd=0, command=lambda v: self.process_and_update(is_dragging=True))
        self.slider_brightness.set(0)
        self.slider_brightness.pack(fill="x", padx=5, pady=2)
        self.slider_brightness.bind("<ButtonRelease-1>", lambda e: self.process_and_update(is_dragging=False))
        
        self.slider_contrast = tk.Scale(sliders_lf, from_=-100, to=100, orient="horizontal", label="Contrast", bg=self.panel_color, fg=self.text_color, highlightthickness=0, bd=0, command=lambda v: self.process_and_update(is_dragging=True))
        self.slider_contrast.set(0)
        self.slider_contrast.pack(fill="x", padx=5, pady=2)
        self.slider_contrast.bind("<ButtonRelease-1>", lambda e: self.process_and_update(is_dragging=False))
        
        self.slider_smooth = tk.Scale(sliders_lf, from_=0, to=10, orient="horizontal", label="Smooth / Denoise (Gaussian)", bg=self.panel_color, fg=self.text_color, highlightthickness=0, bd=0, command=lambda v: self.process_and_update(is_dragging=True))
        self.slider_smooth.set(0)
        self.slider_smooth.pack(fill="x", padx=5, pady=2)
        self.slider_smooth.bind("<ButtonRelease-1>", lambda e: self.process_and_update(is_dragging=False))
        
        btn_reset_sliders = tk.Button(sliders_lf, text="Reset Sliders", command=self.reset_sliders, bg=self.accent_color, fg="white", font=("Segoe UI", 9), bd=0, pady=4)
        btn_reset_sliders.pack(fill="x", padx=5, pady=5)
        
        # Canvas Container Frame (Right Panel) holding Canvas and scrollbars
        canvas_container = tk.Frame(right_panel, bg=self.bg_color)
        canvas_container.pack(fill="both", expand=True, side="top")
        
        # Bottom controls for Pan-STARRS blend slider
        bottom_bar = tk.Frame(right_panel, bg=self.panel_color, height=45)
        bottom_bar.pack(side="bottom", fill="x", padx=10, pady=5)
        
        lbl_blend = tk.Label(bottom_bar, text="DSS Sky Blend:", bg=self.panel_color, fg=self.text_color, font=("Segoe UI", 9, "bold"))
        lbl_blend.pack(side="left", padx=5)
        
        # Slider control for blend ratio
        self.slider_blend = tk.Scale(bottom_bar, from_=0.0, to=1.0, resolution=0.01, orient="horizontal", variable=self.dss_blend_ratio, command=lambda v: self.render_canvas(is_dragging=False), showvalue=True, bg=self.panel_color, fg=self.text_color, highlightthickness=0, troughcolor=self.bg_color, length=300)
        self.slider_blend.pack(side="left", padx=10)
        
        self.hbar = tk.Scrollbar(canvas_container, orient="horizontal")
        self.hbar.pack(side="bottom", fill="x")
        self.vbar = tk.Scrollbar(canvas_container, orient="vertical")
        self.vbar.pack(side="right", fill="y")
        
        self.canvas = tk.Canvas(canvas_container, bg="#111", highlightthickness=0)
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
        if self.btn_mirror_h:
            self.btn_mirror_h.config(bg=self.accent_color if self.mirror_horizontal else "#374151")
        self.process_and_update(is_dragging=False)
        
    def toggle_mirror_v(self):
        self.mirror_vertical = not self.mirror_vertical
        if self.btn_mirror_v:
            self.btn_mirror_v.config(bg=self.accent_color if self.mirror_vertical else "#374151")
        self.process_and_update(is_dragging=False)

    def enable_sky_balance(self):
        self.balance_mode = "Sky"
        self.canvas.config(cursor="crosshair")
        if self.btn_bal_sky:
            self.btn_bal_sky.config(bg="#d97706")
        if self.btn_bal_star:
            self.btn_bal_star.config(bg="#374151")
        
    def enable_star_balance(self):
        self.balance_mode = "Star"
        self.canvas.config(cursor="crosshair")
        if self.btn_bal_star:
            self.btn_bal_star.config(bg="#d97706")
        if self.btn_bal_sky:
            self.btn_bal_sky.config(bg="#374151")

    def zoom_in(self, target_x=None, target_y=None, is_interactive=False):
        self.zoom_to_target(1.3, target_x, target_y, is_interactive)
        
    def zoom_out(self, target_x=None, target_y=None, is_interactive=False):
        self.zoom_to_target(1.0 / 1.3, target_x, target_y, is_interactive)
        
    def zoom_to_target(self, factor, target_x=None, target_y=None, is_interactive=False):
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
        
        self.render_canvas(is_dragging=is_interactive)
        
        # Debounce the high-quality static render
        if is_interactive:
            if getattr(self, '_zoom_debounce_job', None) is not None:
                self.root.after_cancel(self._zoom_debounce_job)
            self._zoom_debounce_job = self.root.after(200, self._finalize_zoom)
            
    def _finalize_zoom(self):
        self._zoom_debounce_job = None
        self.render_canvas(is_dragging=False)
        
    def reset_view(self):
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.render_canvas(is_dragging=False)
        
    def on_mousewheel(self, event):
        # Zoom towards the mouse cursor position interactively
        if event.delta > 0:
            self.zoom_in(event.x, event.y, is_interactive=True)
        else:
            self.zoom_out(event.x, event.y, is_interactive=True)

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
                canvas_w = self.canvas.winfo_width()
                if self.processed_img_full is not None:
                    h, w = self.processed_img_full.shape[:2]
                    fit_ratio = min(canvas_w / w, self.canvas.winfo_height() / h)
                    total_w = w * fit_ratio * self.zoom_level
                    self.pan_x = ((total_w - canvas_w) / 2.0) - (val * total_w)
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
                    self.pan_y = ((total_h - canvas_h) / 2.0) - (val * total_h)
            self.render_canvas(is_dragging=False)

    def handle_file_drop(self, event):
        file_path = event.data
        if file_path.startswith("{") and file_path.endswith("}"):
            file_path = file_path[1:-1]
        if os.path.exists(file_path):
            self.load_fits_from_path(file_path)
            
    def load_fits(self):
        file_path = filedialog.askopenfilename(filetypes=[
            ("Image & FITS files", "*.fits *.fit *.png *.jpg *.jpeg *.tiff *.tif"),
            ("FITS files", "*.fits *.fit"),
            ("PNG images", "*.png"),
            ("JPEG images", "*.jpg *.jpeg"),
            ("TIFF images", "*.tiff *.tif"),
            ("All files", "*.*")
        ])
        if file_path:
            self.load_fits_from_path(file_path)
            
    def load_fits_from_path(self, file_path):
        try:
            ext = os.path.splitext(file_path)[1].lower()
            is_fits = ext in ['.fits', '.fit']
            
            self.fits_path = file_path
            self.debayered_cache = None
            self.temp_marker = None
            self.visual_crop_box = None
            self.remove_gradient_active = False
            self.background_gradient_map = None
            self.catalog_stars = []
            self.calib_stars = []
            self.limiting_mag_check_stars = []
            self.show_limiting_mag_stars.set(True)
            self.show_catalog_stars.set(False)
            self.asteroid_objects = []
            self.show_asteroids.set(True)
            self.transient_objects = []
            self.show_transients.set(True)
            self.loaded_dss_tiles = []
            self.dss_blend_ratio.set(0.0)
            
            self.sky_sample_rgb = None
            self.star_sample_rgb = None
            self.balance_mode = "None"
            if self.btn_bal_sky:
                self.btn_bal_sky.config(bg="#374151")
            if self.btn_bal_star:
                self.btn_bal_star.config(bg="#374151")
            self.var_grayscale.set(False)
            self.var_invert.set(False)
            
            # Reset sliders
            if hasattr(self, 'slider_red_offset') and self.slider_red_offset:
                self.slider_red_offset.set(0)
            if hasattr(self, 'slider_green_offset') and self.slider_green_offset:
                self.slider_green_offset.set(0)
            if hasattr(self, 'slider_blue_offset') and self.slider_blue_offset:
                self.slider_blue_offset.set(0)
            if hasattr(self, 'slider_brightness') and self.slider_brightness:
                self.slider_brightness.set(0)
            if hasattr(self, 'slider_contrast') and self.slider_contrast:
                self.slider_contrast.set(0)
            if hasattr(self, 'slider_smooth') and self.slider_smooth:
                self.slider_smooth.set(0)
                
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.annotations.clear()
            self.zoom_level = 1.0
            self.pan_x = 0.0
            self.pan_y = 0.0
            self.mirror_horizontal = False
            self.mirror_vertical = False
            if self.btn_mirror_h:
                self.btn_mirror_h.config(bg="#374151")
            if self.btn_mirror_v:
                self.btn_mirror_v.config(bg="#374151")
            
            for k in self.curves_widget.points_dict:
                self.curves_widget.points_dict[k] = [(0, 0), (255, 255)]
            self.curves_widget.redraw()
            
            has_ann = False
            
            if is_fits:
                with fits.open(file_path) as hdul:
                    hdu = None
                    for h in hdul:
                        if h.data is not None and h.data.ndim in [2, 3]:
                            hdu = h
                            break
                    
                    if hdu is None:
                        messagebox.showerror("Error", "No image data found in FITS file.")
                        return
                    
                    self.fits_header = hdu.header
                    self.wcs_saved_to_disk = ("CRVAL1" in self.fits_header)
                    self.fits_data = hdu.data.astype(np.float32)
                    has_ann = ('ANNOTATIONS' in hdul)
            else:
                # Load non-FITS via PIL
                from PIL import Image
                pil_img = Image.open(file_path)
                np_img = np.array(pil_img, dtype=np.float32)
                if len(np_img.shape) == 3:
                    if np_img.shape[2] == 4:
                        np_img = np_img[:, :, :3]
                    self.fits_data = np.mean(np_img, axis=2)
                else:
                    self.fits_data = np_img
                
                self.fits_header = fits.Header()
                self.wcs_saved_to_disk = False
                
                # Check for sidecar file: <filename>.info.json
                sidecar_path = file_path + ".info.json"
                sidecar_data = {}
                if os.path.exists(sidecar_path):
                    import json
                    try:
                        with open(sidecar_path, "r", encoding="utf-8") as sf:
                            sidecar_data = json.load(sf)
                    except Exception:
                        pass
                
                if "header" in sidecar_data:
                    for k, v in sidecar_data["header"].items():
                        self.fits_header[k] = v
                if "annotations" in sidecar_data:
                    self.annotations = sidecar_data["annotations"]
                
                # Fallback parameters from settings if missing
                if 'FOCALLEN' not in self.fits_header or self.fits_header['FOCALLEN'] == "":
                    if self.settings.get("focal_length"):
                        self.fits_header['FOCALLEN'] = float(self.settings["focal_length"])
                if 'XPIXSZ' not in self.fits_header or self.fits_header['XPIXSZ'] == "":
                    if self.settings.get("pixel_size"):
                        self.fits_header['XPIXSZ'] = float(self.settings["pixel_size"])
                        self.fits_header['YPIXSZ'] = float(self.settings["pixel_size"])
                
                # Fallback observation time
                if 'DATE-OBS' not in self.fits_header:
                    mtime = os.path.getmtime(file_path)
                    self.fits_header['DATE-OBS'] = Time(mtime, format='unix').isot
            
            # Set Bayer pattern
            if 'BAYERPAT' in self.fits_header:
                self.bayer_pattern.set(self.fits_header['BAYERPAT'].upper())
            else:
                self.bayer_pattern.set("None")
                
            # Set Observation time
            self.observation_time = Time.now()
            if 'DATE-OBS' in self.fits_header:
                try:
                    self.observation_time = Time(self.fits_header['DATE-OBS'], format='isot')
                except Exception:
                    pass
            
            # Metadata tree
            self.meta_tree.delete(*self.meta_tree.get_children())
            if not is_fits:
                self.meta_tree.insert("", "end", text="IMAGE_TYPE", values=("Non-FITS (Loaded via PIL)",))
                sidecar_path = file_path + ".info.json"
                if os.path.exists(sidecar_path):
                    self.meta_tree.insert("", "end", text="SIDECAR_INFO", values=("Loaded from .info.json",))
            
            important_keys = ["OBJECT", "EXPTIME", "TELESCOP", "INSTRUME", "FILTER", "DATE-OBS", "BAYERPAT", "EQUINOX"]
            for key in important_keys:
                if key in self.fits_header:
                    self.meta_tree.insert("", "end", text=key, values=(str(self.fits_header[key]),))
            for key, val in self.fits_header.items():
                if key not in important_keys and key.strip() != "":
                    self.meta_tree.insert("", "end", text=key, values=(str(val),))
            
            # Resolve WCS
            job_info = self.jobs.get(self.fits_path, {})
            if job_info.get("status") == "success" and "wcs" in job_info:
                try:
                    cached_header = fits.Header()
                    for k, v in job_info["wcs"].items():
                        cached_header[k] = v
                    self.wcs = WCS(cached_header)
                    if self.wcs and not self.wcs.has_celestial:
                        self.wcs = None
                except Exception:
                    self.wcs = None
            else:
                try:
                    self.wcs = WCS(self.fits_header, naxis=2)
                    if self.wcs and not self.wcs.has_celestial:
                        self.wcs = None
                except Exception:
                    self.wcs = None
            
            self.update_wcs_hint_visibility()
            
            epoch_val = self.fits_header.get('EQUINOX', 2000.0)
            if epoch_val == 2000.0:
                self.lbl_epoch_status.config(text=f"Header epoch: J2000 (detected)")
            else:
                self.lbl_epoch_status.config(text=f"Header epoch: JNow / EQUINOX {epoch_val}")
            
            self.push_state()
            self.process_and_update(is_dragging=False)
            
            if has_ann:
                messagebox.showinfo("Importable Annotations", 
                                    "This FITS file contains annotations extension table.\n"
                                    "You can import them via:\n"
                                    "Astrometry -> Annotations Manager -> Import Annotations from FITS.", 
                                    parent=self.root)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {str(e)}")

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
            'star_sample': self.star_sample_rgb,
            'visual_crop_box': self.visual_crop_box
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
        self.visual_crop_box = state.get('visual_crop_box', None)
        
        self.process_and_update(is_dragging=False)

    def on_debayer_selection_change(self, event):
        self.debayered_cache = None
        self.process_and_update(is_dragging=False)

    def on_channel_combobox_change(self, event):
        self.curves_widget.set_active_channel(self.channel_selection.get())
        self.process_and_update(is_dragging=False)

    def on_curves_changed(self, is_dragging=False):
        self.process_and_update(is_dragging=is_dragging)

    def on_log_hist_changed(self):
        self.curves_widget.log_scale = self.var_log_hist.get()
        self.curves_widget.compute_scaled_histograms()

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
        if getattr(self, 'remove_gradient_active', False) and getattr(self, 'background_gradient_map', None) is not None:
            base_img = np.clip(base_img - self.background_gradient_map, 0.0, None)
            
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
            
        # 2. Convert to uint16 to preserve precision during curve correction
        img_16u = (data_norm * 65535.0).astype(np.uint16)
        
        # 3. Apply Multi-Channel Curves at 16-bit precision
        lut_rgb = self.curves_widget.get_lut_for_pts(self.curves_widget.points_dict["RGB/Luminance"])
        
        lut_r = self.curves_widget.get_lut_for_pts(self.curves_widget.points_dict["Red Channel Only"])
        lut_g = self.curves_widget.get_lut_for_pts(self.curves_widget.points_dict["Green Channel Only"])
        lut_b = self.curves_widget.get_lut_for_pts(self.curves_widget.points_dict["Blue Channel Only"])
        
        # Compose RGB/Luminance curve with channel curves to preserve 16-bit precision mapping
        lut_r_comb = lut_r[lut_rgb]
        lut_g_comb = lut_g[lut_rgb]
        lut_b_comb = lut_b[lut_rgb]
        
        # Interpolate the composed 256-point LUTs to 65536 points for mapping the 16-bit image channels to 8-bit
        lut_r_65536 = np.interp(np.linspace(0, 255, 65536), np.arange(256), lut_r_comb).astype(np.uint8)
        lut_g_65536 = np.interp(np.linspace(0, 255, 65536), np.arange(256), lut_g_comb).astype(np.uint8)
        lut_b_65536 = np.interp(np.linspace(0, 255, 65536), np.arange(256), lut_b_comb).astype(np.uint8)
        
        img_processed = np.zeros(img_16u.shape, dtype=np.uint8)
        img_processed[:, :, 0] = lut_r_65536[img_16u[:, :, 0]]
        img_processed[:, :, 1] = lut_g_65536[img_16u[:, :, 1]]
        img_processed[:, :, 2] = lut_b_65536[img_16u[:, :, 2]]
        
        # Apply Slider adjustments (Color Shifts, Brightness, Contrast) on the STRETCHED 8-bit image
        r_offset = self.slider_red_offset.get() / 100.0
        g_offset = self.slider_green_offset.get() / 100.0
        b_offset = self.slider_blue_offset.get() / 100.0
        brightness = self.slider_brightness.get() / 100.0
        contrast = self.slider_contrast.get() / 100.0
        
        img_float = img_processed.astype(np.float32) / 255.0
        
        # Apply red, green, blue shifts
        img_float[:, :, 0] += r_offset
        img_float[:, :, 1] += g_offset
        img_float[:, :, 2] += b_offset
        
        # Apply brightness
        if brightness != 0.0:
            img_float += brightness
            
        # Apply contrast
        if contrast != 0.0:
            if contrast >= 0.0:
                factor = 1.0 + contrast * 3.0
            else:
                factor = 1.0 + contrast
            img_float = (img_float - 0.5) * factor + 0.5
            
        img_float = np.clip(img_float, 0.0, 1.0)
        img_processed = (img_float * 255.0).astype(np.uint8)
        
        # Apply Gaussian Smoothing filter to "clean" background noise if enabled
        smooth_val = self.slider_smooth.get()
        if smooth_val > 0:
            ksize = 2 * smooth_val + 1
            img_processed = cv2.GaussianBlur(img_processed, (ksize, ksize), 0)
            
        # Apply visual crop if active
        if self.visual_crop_box is not None:
            start_x, start_y, end_x, end_y = self.visual_crop_box
            img_processed = img_processed[start_y:end_y, start_x:end_x]
        
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
            
        # Update curves widget histograms with the final processed (stretched) image in real-time
        self.curves_widget.update_histograms(img_processed)
        
        self.render_canvas(is_dragging=is_dragging)

    def reset_color_manipulation(self):
        if self.fits_data is None:
            return
            
        self.push_state()
        
        # Reset color balance samplers, visual crop box, and background gradient
        self.sky_sample_rgb = None
        self.star_sample_rgb = None
        self.visual_crop_box = None
        self.remove_gradient_active = False
        self.background_gradient_map = None
        
        # Reset curves points to linear defaults
        for k in self.curves_widget.points_dict:
            self.curves_widget.points_dict[k] = [(0, 0), (255, 255)]
            
        self.curves_widget.redraw()
        self.process_and_update(is_dragging=False)
        messagebox.showinfo("Reset", "Color Balance gains, Curves adjustments, and Background Gradient have been reset.")

    def remove_background_gradient(self):
        if self.debayered_cache is None:
            messagebox.showerror("Error", "Load an image first.", parent=self.root)
            return
            
        self.root.config(cursor="watch")
        self.root.update()
        
        try:
            h, w = self.debayered_cache.shape[:2]
            chans = self.debayered_cache.shape[2] if len(self.debayered_cache.shape) == 3 else 1
            
            # Estimate background using a 16x16 grid of block medians
            grid_size = 16
            block_h = max(4, h // grid_size)
            block_w = max(4, w // grid_size)
            
            # Generate x, y coordinates for block centers
            xs = []
            ys = []
            zs = [[] for _ in range(chans)]
            
            for r in range(grid_size):
                y_c = r * block_h + block_h / 2.0
                for c in range(grid_size):
                    x_c = c * block_w + block_w / 2.0
                    
                    # Get block slice
                    r_end = min(h, (r+1)*block_h)
                    c_end = min(w, (c+1)*block_w)
                    if chans > 1:
                        block = self.debayered_cache[r*block_h:r_end, c*block_w:c_end]
                    else:
                        block = self.debayered_cache[r*block_h:r_end, c*block_w:c_end]
                    if block.size == 0:
                        continue
                        
                    xs.append(x_c)
                    ys.append(y_c)
                    if chans > 1:
                        for ch in range(chans):
                            zs[ch].append(np.median(block[:, :, ch]))
                    else:
                        zs[0].append(np.median(block))
                        
            xs = np.array(xs)
            ys = np.array(ys)
            
            # Design matrix for 2nd order polynomial: 1, x, y, x^2, y^2, x*y
            A = np.column_stack([np.ones_like(xs), xs, ys, xs**2, ys**2, xs*ys])
            
            # Fit polynomial coefficients for each channel
            coefs = []
            for ch in range(chans):
                # Least squares fit
                c_fit, _, _, _ = np.linalg.lstsq(A, zs[ch], rcond=None)
                coefs.append(c_fit)
                
            # Evaluate the fitted polynomial across the entire image grid to create the gradient map
            y_indices, x_indices = np.indices((h, w), dtype=np.float32)
            
            self.background_gradient_map = np.zeros_like(self.debayered_cache)
            if chans > 1:
                for ch in range(chans):
                    c_fit = coefs[ch]
                    z_fit = c_fit[0] + c_fit[1]*x_indices + c_fit[2]*y_indices + c_fit[3]*(x_indices**2) + c_fit[4]*(y_indices**2) + c_fit[5]*(x_indices*y_indices)
                    z_fit_rel = z_fit - z_fit.min()
                    self.background_gradient_map[:, :, ch] = z_fit_rel.astype(np.float32)
            else:
                c_fit = coefs[0]
                z_fit = c_fit[0] + c_fit[1]*x_indices + c_fit[2]*y_indices + c_fit[3]*(x_indices**2) + c_fit[4]*(y_indices**2) + c_fit[5]*(x_indices*y_indices)
                z_fit_rel = z_fit - z_fit.min()
                self.background_gradient_map = z_fit_rel.astype(np.float32)
                
            self.remove_gradient_active = True
            self.process_and_update(is_dragging=False)
            messagebox.showinfo("Background Gradient Removal", "Background gradient removed successfully (cosmetic only).", parent=self.root)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to remove background gradient:\n{str(e)}", parent=self.root)
        finally:
            self.root.config(cursor="")

    def apply_autostretch(self):
        if self.fits_data is None:
            return
            
        self.push_state()
        
        if self.debayered_cache is None:
            self.update_debayer_cache()
            
        base_img = self.debayered_cache.copy()
        
        # Normalize exactly as done in process_and_update to ensure identical scales
        global_min = self.debayered_cache.min()
        global_max = self.debayered_cache.max()
        global_range = global_max - global_min if global_max > global_min else 1.0
        
        data_norm = (base_img - global_min) / global_range
        data_norm = np.clip(data_norm, 0.0, 1.0)
        
        if self.sky_sample_rgb is not None or self.star_sample_rgb is not None:
            black_offset = np.array(self.sky_sample_rgb if self.sky_sample_rgb is not None else [0.0, 0.0, 0.0])
            white_gain = np.array(self.star_sample_rgb if self.star_sample_rgb is not None else [1.0, 1.0, 1.0])
            denom = white_gain - black_offset
            denom[denom <= 0.0001] = 0.0001
            data_norm = (data_norm - black_offset) / denom
            data_norm = np.clip(data_norm, 0.0, 1.0)
            
        if self.fits_data.ndim == 3:
            # Color FITS image: calculate stretch parameters separately for Red, Green, and Blue channels
            # to align peaks (color balance) and dilate each noise bell perfectly.
            channels = [("Red Channel Only", data_norm[:,:,0]), 
                        ("Green Channel Only", data_norm[:,:,1]), 
                        ("Blue Channel Only", data_norm[:,:,2])]
            
            for ch_name, chan in channels:
                median = np.median(chan)
                mad = np.median(np.abs(chan - median))
                if mad < 0.0001:
                    mad = np.std(chan)
                if mad < 0.0001:
                    mad = 0.0001
                
                # Convert MAD to standard deviation (sigma) of background noise
                sigma = 1.4826 * mad
                
                # Siril default clipping factor
                shadows_clip_factor = -2.8
                black_val = max(0.0, median + shadows_clip_factor * sigma)
                white_val = 1.0
                target_bg = 0.25
                
                range_val = white_val - black_val if white_val > black_val else 1.0
                median_clipped = max(0.0001, (median - black_val) / range_val)
                
                denom = median_clipped * (2.0 * target_bg - 1.0) - target_bg
                if abs(denom) < 0.0001:
                    m = 0.0016
                else:
                    m = (median_clipped * (target_bg - 1.0)) / denom
                
                m = max(0.0001, min(m, 0.9999))
                
                black_idx = int(black_val * 255.0)
                black_idx = max(0, min(black_idx, 240))
                
                def eval_mtf(x_val):
                    if x_val <= black_val:
                        return 0.0
                    if x_val >= white_val:
                        return 1.0
                    xn = (x_val - black_val) / (white_val - black_val)
                    num = (m - 1.0) * xn
                    den = (2.0 * m - 1.0) * xn - m
                    if abs(den) < 0.0001:
                        return xn
                    return num / den

                p1_x = black_val + 0.005 * (white_val - black_val)
                p1_y = eval_mtf(p1_x)
                p2_x = black_val + 0.05 * (white_val - black_val)
                p2_y = eval_mtf(p2_x)
                p3_x = black_val + 0.35 * (white_val - black_val)
                p3_y = eval_mtf(p3_x)
                
                p1_idx_x = int(p1_x * 255.0)
                p1_idx_y = int(p1_y * 255.0)
                p2_idx_x = int(p2_x * 255.0)
                p2_idx_y = int(p2_y * 255.0)
                p3_idx_x = int(p3_x * 255.0)
                p3_idx_y = int(p3_y * 255.0)
                
                p1_idx_x = max(black_idx + 1, min(p1_idx_x, 252))
                p2_idx_x = max(p1_idx_x + 1, min(p2_idx_x, 253))
                p3_idx_x = max(p2_idx_x + 1, min(p3_idx_x, 254))
                
                self.curves_widget.points_dict[ch_name] = [
                    (0, 0),
                    (black_idx, 0),
                    (p1_idx_x, p1_idx_y),
                    (p2_idx_x, p2_idx_y),
                    (p3_idx_x, p3_idx_y),
                    (255, 255)
                ]
            
            # Reset the main RGB/Luminance curve
            self.curves_widget.points_dict["RGB/Luminance"] = [(0, 0), (255, 255)]
            
        else:
            # Grayscale FITS image: stretch the main RGB/Luminance curve
            # Since grayscale self.debayered_cache is H x W x 3, use channel 0.
            chan = data_norm[:,:,0]
            
            median = np.median(chan)
            mad = np.median(np.abs(chan - median))
            if mad < 0.0001:
                mad = np.std(chan)
            if mad < 0.0001:
                mad = 0.0001
                
            sigma = 1.4826 * mad
            
            shadows_clip_factor = -2.8
            black_val = max(0.0, median + shadows_clip_factor * sigma)
            white_val = 1.0
            target_bg = 0.25
            
            range_val = white_val - black_val if white_val > black_val else 1.0
            median_clipped = max(0.0001, (median - black_val) / range_val)
            
            denom = median_clipped * (2.0 * target_bg - 1.0) - target_bg
            if abs(denom) < 0.0001:
                m = 0.0016
            else:
                m = (median_clipped * (target_bg - 1.0)) / denom
            
            m = max(0.0001, min(m, 0.9999))
            
            black_idx = int(black_val * 255.0)
            black_idx = max(0, min(black_idx, 240))
            
            def eval_mtf(x_val):
                if x_val <= black_val:
                    return 0.0
                if x_val >= white_val:
                    return 1.0
                xn = (x_val - black_val) / (white_val - black_val)
                num = (m - 1.0) * xn
                den = (2.0 * m - 1.0) * xn - m
                if abs(den) < 0.0001:
                    return xn
                return num / den

            p1_x = black_val + 0.005 * (white_val - black_val)
            p1_y = eval_mtf(p1_x)
            p2_x = black_val + 0.05 * (white_val - black_val)
            p2_y = eval_mtf(p2_x)
            p3_x = black_val + 0.35 * (white_val - black_val)
            p3_y = eval_mtf(p3_x)
            
            p1_idx_x = int(p1_x * 255.0)
            p1_idx_y = int(p1_y * 255.0)
            p2_idx_x = int(p2_x * 255.0)
            p2_idx_y = int(p2_y * 255.0)
            p3_idx_x = int(p3_x * 255.0)
            p3_idx_y = int(p3_y * 255.0)
            
            p1_idx_x = max(black_idx + 1, min(p1_idx_x, 252))
            p2_idx_x = max(p1_idx_x + 1, min(p2_idx_x, 253))
            p3_idx_x = max(p2_idx_x + 1, min(p3_idx_x, 254))
            
            self.curves_widget.points_dict["RGB/Luminance"] = [
                (0, 0),
                (black_idx, 0),
                (p1_idx_x, p1_idx_y),
                (p2_idx_x, p2_idx_y),
                (p3_idx_x, p3_idx_y),
                (255, 255)
            ]
            # Reset channel curves
            self.curves_widget.points_dict["Red Channel Only"] = [(0, 0), (255, 255)]
            self.curves_widget.points_dict["Green Channel Only"] = [(0, 0), (255, 255)]
            self.curves_widget.points_dict["Blue Channel Only"] = [(0, 0), (255, 255)]
            
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
        
        # Enforce panning clamps to stay within visible borders
        if new_w > canvas_w:
            max_pan_x = (new_w - canvas_w) / 2.0
            self.pan_x = float(np.clip(self.pan_x, -max_pan_x, max_pan_x))
        else:
            self.pan_x = 0.0
            
        if new_h > canvas_h:
            max_pan_y = (new_h - canvas_h) / 2.0
            self.pan_y = float(np.clip(self.pan_y, -max_pan_y, max_pan_y))
        else:
            self.pan_y = 0.0
            
        self.img_scale_ratio = total_scale
        
        # Calculate viewport bounds
        base_offset_x = (canvas_w - new_w) // 2 + int(self.pan_x)
        base_offset_y = (canvas_h - new_h) // 2 + int(self.pan_y)
        
        left_px = -base_offset_x
        top_px = -base_offset_y
        right_px = canvas_w - base_offset_x
        bottom_px = canvas_h - base_offset_y
        
        # Calculate visible crop bounding box on the resized image
        crop_left = max(0, left_px)
        crop_top = max(0, top_px)
        crop_right = min(new_w, right_px)
        crop_bottom = min(new_h, bottom_px)
        
        # Calculate bounding box on the original FITS image
        src_left_int = int(np.floor(crop_left / total_scale))
        src_top_int = int(np.floor(crop_top / total_scale))
        src_right_int = int(np.ceil(crop_right / total_scale))
        src_bottom_int = int(np.ceil(crop_bottom / total_scale))
        
        # Clamp to bounds
        src_left_int = max(0, min(w, src_left_int))
        src_top_int = max(0, min(h, src_top_int))
        src_right_int = max(0, min(w, src_right_int))
        src_bottom_int = max(0, min(h, src_bottom_int))
        
        self.src_left_int = src_left_int
        self.src_top_int = src_top_int
        
        crop_w_src = src_right_int - src_left_int
        crop_h_src = src_bottom_int - src_top_int
        
        if crop_w_src < 2 or crop_h_src < 2:
            return
            
        # Crop only the visible sky portion
        crop_img = img_to_draw[src_top_int:src_bottom_int, src_left_int:src_right_int]
        
        # Resized crop dimensions
        crop_new_w = int(crop_w_src * total_scale)
        crop_new_h = int(crop_h_src * total_scale)
        
        if crop_new_w < 5 or crop_new_h < 5:
            return
            
        self.img_offset_x = base_offset_x + int(src_left_int * total_scale)
        self.img_offset_y = base_offset_y + int(src_top_int * total_scale)
        
        pil_img = Image.fromarray(crop_img)
        resized_pil = pil_img.resize((crop_new_w, crop_new_h), Image.Resampling.NEAREST if is_dragging else Image.Resampling.BILINEAR)
        
        # Apply DSS blending if loaded, blend_ratio > 0, and not dragging for interactive smoothness
        blend_alpha = self.dss_blend_ratio.get()
        if not is_dragging and getattr(self, 'loaded_dss_tiles', None) and blend_alpha > 0.0 and self.wcs is not None:
            try:
                resized_np = np.array(resized_pil)
                
                # Dynamic caching of warped DSS background based on viewport parameters to maximize slider frame rate
                view_key = (crop_new_w, crop_new_h, src_left_int, src_top_int, self.zoom_level, self.mirror_horizontal, self.mirror_vertical, len(self.loaded_dss_tiles))
                if (getattr(self, 'dss_warped_cache', None) is None or 
                    getattr(self, 'dss_mask_cache', None) is None or 
                    getattr(self, 'dss_cache_view_key', None) != view_key):
                    
                    # Build canvas crop grid
                    y_indices, x_indices = np.indices((crop_new_h, crop_new_w), dtype=np.float32)
                    
                    if self.mirror_horizontal:
                        rx = 1.0 - (x_indices + 0.5) / crop_new_w
                    else:
                        rx = (x_indices + 0.5) / crop_new_w
                        
                    if self.mirror_vertical:
                        ry = 1.0 - (y_indices + 0.5) / crop_new_h
                    else:
                        ry = (y_indices + 0.5) / crop_new_h
                        
                    fx = src_left_int + rx * crop_w_src - 0.5
                    fy = src_top_int + ry * crop_h_src - 0.5
                    
                    # Remap using astropy WCS coordinates
                    ra_vals, dec_vals = self.wcs.pixel_to_world_values(fx, fy)
                    
                    accum_warped = np.zeros((crop_new_h, crop_new_w), dtype=np.uint8)
                    accum_mask = np.zeros((crop_new_h, crop_new_w), dtype=bool)
                    
                    # Process and stack each loaded sky tile
                    for tile in self.loaded_dss_tiles:
                        dss_wcs = tile['wcs']
                        dss_data = tile['data']
                        
                        ps_x, ps_y = dss_wcs.world_to_pixel_values(ra_vals, dec_vals)
                        ps_h, ps_w = dss_data.shape[:2]
                        tile_mask = (ps_x >= 0) & (ps_x < ps_w) & (ps_y >= 0) & (ps_y < ps_h)
                        
                        if tile_mask.any():
                            warped_tile = cv2.remap(dss_data, ps_x.astype(np.float32), ps_y.astype(np.float32),
                                                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                            accum_warped[tile_mask] = warped_tile[tile_mask]
                            accum_mask = accum_mask | tile_mask
                    
                    # Match channel dimensions
                    if len(resized_np.shape) == 3 and resized_np.shape[2] == 3:
                        warped_blend = cv2.merge([accum_warped, accum_warped, accum_warped])
                    else:
                        warped_blend = accum_warped
                        
                    self.dss_warped_cache = warped_blend
                    self.dss_mask_cache = accum_mask
                    self.dss_cache_view_key = view_key
                
                # Fetch from cache
                warped_blend = self.dss_warped_cache
                valid_mask = self.dss_mask_cache
                
                blended_img = resized_np.copy()
                # Blending only in the overlapping region where DSS data is valid
                if valid_mask.any():
                    blended_img[valid_mask] = (blended_img[valid_mask].astype(np.float32) * (1.0 - blend_alpha) + 
                                               warped_blend[valid_mask].astype(np.float32) * blend_alpha).astype(np.uint8)
                                               
                resized_pil = Image.fromarray(blended_img)
            except Exception as e:
                pass
                
        draw = ImageDraw.Draw(resized_pil)
        
        # 1. Coordinate mapping helper for visual crop offsets
        h_orig, w_orig = self.debayered_cache.shape[:2]
        
        def map_coords(px_x, px_y):
            x_val = px_x
            y_val = px_y
            if self.visual_crop_box is not None:
                start_x, start_y, end_x, end_y = self.visual_crop_box
                x_val = x_val - start_x
                y_val = y_val - start_y
                
            rx = x_val / w
            ry = y_val / h
            if self.mirror_horizontal:
                rx = 1.0 - rx
            if self.mirror_vertical:
                ry = 1.0 - ry
                
            sx = int(rx * new_w) - int(src_left_int * total_scale)
            sy = int(ry * new_h) - int(src_top_int * total_scale)
            return sx, sy
            
        # Draw Saved Annotations (if enabled)
        if self.show_annotations.get():
            for ann in self.annotations:
                px = ann['x'] * w_orig
                py = ann['y'] * h_orig
                ax, ay = map_coords(px, py)
                
                if 0 <= ax < crop_new_w and 0 <= ay < crop_new_h:
                    self.draw_star_crosshair(draw, ax, ay, size=24, width=2)
                    draw.text((ax + 26, ay + 26), ann['text'], fill=self.green_bright)
            
        # 2. Draw Temporary Target marker
        if self.temp_marker:
            px = self.temp_marker['ratio_x'] * w_orig
            py = self.temp_marker['ratio_y'] * h_orig
            tx, ty = map_coords(px, py)
            
            if 0 <= tx < crop_new_w and 0 <= ty < crop_new_h:
                if self.temp_marker.get('type') == 'gaia_query':
                    r = 10
                    draw.rectangle([tx - r, ty - r, tx + r, ty + r], outline=self.green_bright, width=2)
                    draw.text((tx + 15, ty - 8), "Gaia Query Target", fill=self.green_bright)
                    
                    if self.temp_marker.get('catalog_ratio_x') is not None:
                        cx_px = self.temp_marker['catalog_ratio_x'] * w_orig
                        cy_px = self.temp_marker['catalog_ratio_y'] * h_orig
                        cx_c, cy_c = map_coords(cx_px, cy_px)
                        
                        if 0 <= cx_c < crop_new_w and 0 <= cy_c < crop_new_h:
                            size = 6
                            draw.line([cx_c - size, cy_c - size, cx_c + size, cy_c + size], fill="#ef4444", width=2)
                            draw.line([cx_c - size, cy_c + size, cx_c + size, cy_c - size], fill="#ef4444", width=2)
                            draw.text((cx_c + 8, cy_c - 8), "Gaia Star Position", fill="#ef4444")
                else:
                    self.draw_star_crosshair(draw, tx, ty, size=32, width=2)
                    draw.text((tx + 34, ty + 34), f"Target: RA={self.temp_marker['ra']}\nDEC={self.temp_marker['dec']}", fill="#f43f5e")
            
        # Draw Limiting Magnitude Check Stars (colored circles)
        if self.show_limiting_mag_stars.get() and getattr(self, 'limiting_mag_check_stars', None):
            for star in self.limiting_mag_check_stars:
                sx, sy = map_coords(star['x'], star['y'])
                if 0 <= sx < crop_new_w and 0 <= sy < crop_new_h:
                    r_star = 12
                    color = star.get('color', '#facc15')
                    draw.ellipse([sx - r_star, sy - r_star, sx + r_star, sy + r_star], outline=color, width=2)
                    draw.text((sx + 14, sy - 8), f"G={star['mag']:.1f}", fill=color)

        # 3. Draw Catalog Stars Overlay (cyan circles)
        if self.show_catalog_stars.get() and self.catalog_stars and self.wcs:
            header_epoch = self.fits_header.get('EQUINOX', 2000.0)
            for star in self.catalog_stars:
                try:
                    # Convert to FK5 to match epoch transformation of mark_radec_input
                    input_coord = SkyCoord(ra=star['ra']*u.deg, dec=star['dec']*u.deg, frame='icrs')
                    if header_epoch == 2000.0:
                        target_wcs_coord = input_coord.transform_to(FK5(equinox='J2000'))
                    else:
                        target_wcs_coord = input_coord.transform_to(FK5(equinox=self.observation_time))
                        
                    px_x, px_y = self.wcs.world_to_pixel(target_wcs_coord)
                    if 0 <= px_x < w_orig and 0 <= px_y < h_orig:
                        sx, sy = map_coords(px_x, px_y)
                        
                        if 0 <= sx < crop_new_w and 0 <= sy < crop_new_h:
                            r_star = 8
                            if star.get('is_variable'):
                                # Red outline for variable stars, label with V and rounded mag (no decimals)
                                draw.ellipse([sx - r_star, sy - r_star, sx + r_star, sy + r_star], outline="#ef4444", width=1)
                                draw.text((sx + 10, sy - 8), f"V{int(round(star['mag']))}", fill="#ef4444")
                            else:
                                # Electric blue outline for catalog stars
                                draw.ellipse([sx - r_star, sy - r_star, sx + r_star, sy + r_star], outline="#2979ff", width=1)
                                draw.text((sx + 10, sy - 8), f"{star['mag']:.2f}", fill="#2979ff")
                except Exception:
                    continue
                    
        # 4. Draw Selected Calibration Stars (green circles)
        if self.show_catalog_stars.get() and self.calib_stars:
            for star in self.calib_stars:
                sx, sy = map_coords(star['x'], star['y'])
                
                if 0 <= sx < crop_new_w and 0 <= sy < crop_new_h:
                    r_star = 12
                    # Thicker green circle for calibrated stars (no text label)
                    draw.ellipse([sx - r_star, sy - r_star, sx + r_star, sy + r_star], outline="#22c55e", width=2)
                    
        # 5. Draw MPC Asteroids Overlay (orange diamonds)
        if self.show_asteroids.get() and getattr(self, 'asteroid_objects', None) and self.wcs:
            for ast in self.asteroid_objects:
                try:
                    # Parse RA/Dec HMS/DMS
                    input_coord = SkyCoord(ast['ra_hms'], ast['dec_dms'], unit=(u.hourangle, u.deg))
                    px_x, px_y = self.wcs.world_to_pixel(input_coord)
                    if 0 <= px_x < w_orig and 0 <= px_y < h_orig:
                        ax, ay = map_coords(px_x, px_y)
                        
                        if 0 <= ax < crop_new_w and 0 <= ay < crop_new_h:
                            r_ast = 6
                            pts = [
                                (ax, ay - r_ast),
                                (ax + r_ast, ay),
                                (ax, ay + r_ast),
                                (ax - r_ast, ay)
                            ]
                            draw.polygon(pts, outline="#f97316", width=2)
                            draw.text((ax + 10, ay - 8), f"{ast['name']} ({ast['mag']:.1f})", fill="#f97316")
                except Exception:
                    continue
                    
        # 6. Draw TNS Transients Overlay (colored squares)
        if self.show_transients.get() and getattr(self, 'transient_objects', None) and self.wcs:
            for trans in self.transient_objects:
                try:
                    input_coord = SkyCoord(trans['ra'], trans['dec'], unit=(u.deg, u.deg), frame='icrs')
                    px_x, px_y = self.wcs.world_to_pixel(input_coord)
                    if 0 <= px_x < w_orig and 0 <= px_y < h_orig:
                        ax, ay = map_coords(px_x, px_y)
                        if 0 <= ax < crop_new_w and 0 <= ay < crop_new_h:
                            r_trans = 8
                            color = trans.get('color', '#a855f7')
                            draw.rectangle([ax - r_trans, ay - r_trans, ax + r_trans, ay + r_trans], outline=color, width=2)
                            
                            # Name, Discovery date, Magnitude
                            disc_date = trans['discovery_date'].split()[0]
                            t_type = trans['type'] if trans['type'] else "Unclassified"
                            label_text = f"{trans['name']} ({t_type})\nMag: {trans['mag']}\nDisc: {disc_date}"
                            draw.text((ax + 12, ay - 12), label_text, fill=color)
                except Exception:
                    continue
                
        self.tk_image = ImageTk.PhotoImage(resized_pil)
        
        self.canvas.delete("all")
        self.canvas.create_image(self.img_offset_x, self.img_offset_y, anchor="nw", image=self.tk_image)
        
        # Configure scrollregion on the Canvas and dynamically calculate scrollbar thumb positions
        self.canvas.config(scrollregion=(min(0, self.img_offset_x), min(0, self.img_offset_y), 
                                         max(canvas_w, self.img_offset_x + new_w), max(canvas_h, self.img_offset_y + new_h)))
        
        # Manually compute scrollbar fraction views
        if new_w > canvas_w:
            left_frac = max(0.0, min(1.0, ((new_w - canvas_w) / 2.0 - self.pan_x) / new_w))
            right_frac = max(0.0, min(1.0, ((new_w + canvas_w) / 2.0 - self.pan_x) / new_w))
            self.hbar.set(left_frac, right_frac)
        else:
            self.hbar.set(0.0, 1.0)
            
        if new_h > canvas_h:
            top_frac = max(0.0, min(1.0, ((new_h - canvas_h) / 2.0 - self.pan_y) / new_h))
            bottom_frac = max(0.0, min(1.0, ((new_h + canvas_h) / 2.0 - self.pan_y) / new_h))
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
        
        w_px = int(canvas_x / self.img_scale_ratio) + self.src_left_int
        h_px = int(canvas_y / self.img_scale_ratio) + self.src_top_int
        
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
            self.calibration_mode = False
            self.measurement_mode = False
            self.balance_mode = "None"
            self.gaia_search_mode = False
            self.gaia_search_var.set(False)
            if self.btn_bal_sky:
                self.btn_bal_sky.config(bg="#374151")
            if self.btn_bal_star:
                self.btn_bal_star.config(bg="#374151")
            self.btn_annotate.config(text="Annotate Mode: Off", bg="#374151")
            self.btn_calibration.config(text="Mark Calib Stars: Off", bg="#374151")
            self.btn_measurement.config(text="Measure Target: Off", bg="#374151")
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
            self.calibration_mode = False
            self.measurement_mode = False
            self.balance_mode = "None"
            self.gaia_search_mode = False
            self.gaia_search_var.set(False)
            if self.btn_bal_sky:
                self.btn_bal_sky.config(bg="#374151")
            if self.btn_bal_star:
                self.btn_bal_star.config(bg="#374151")
            self.btn_crop.config(text="Crop Mode: Off", bg="#374151")
            self.btn_calibration.config(text="Mark Calib Stars: Off", bg="#374151")
            self.btn_measurement.config(text="Measure Target: Off", bg="#374151")
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
        h_orig, w_orig = self.debayered_cache.shape[:2]
        
        px_x = int(canvas_x / self.img_scale_ratio) + self.src_left_int
        px_y = int(canvas_y / self.img_scale_ratio) + self.src_top_int
        
        # Adjust coordinate mapping if visual crop is active
        if self.visual_crop_box is not None:
            start_x, start_y, end_x, end_y = self.visual_crop_box
            px_x += start_x
            px_y += start_y
            
        # Gaia Star Query Mode Click Logic
        if getattr(self, 'gaia_search_mode', False):
            if 0 <= px_x < w_orig and 0 <= px_y < h_orig:
                corr_x, corr_y = px_x, px_y
                if self.mirror_horizontal:
                    corr_x = (w_orig - 1) - corr_x
                if self.mirror_vertical:
                    corr_y = (h_orig - 1) - corr_y
                    
                self.handle_gaia_search_click(corr_x, corr_y)
                return
                
        # Calibration Mode Click Logic
        if getattr(self, 'calibration_mode', False):
            if 0 <= px_x < w_orig and 0 <= px_y < h_orig:
                corr_x, corr_y = px_x, px_y
                if self.mirror_horizontal:
                    corr_x = (w_orig - 1) - corr_x
                if self.mirror_vertical:
                    corr_y = (h_orig - 1) - corr_y
                    
                self.handle_calibration_click(corr_x, corr_y)
                return
                
        # Measurement Mode Click Logic
        if getattr(self, 'measurement_mode', False):
            if 0 <= px_x < w_orig and 0 <= px_y < h_orig:
                corr_x, corr_y = px_x, px_y
                if self.mirror_horizontal:
                    corr_x = (w_orig - 1) - corr_x
                if self.mirror_vertical:
                    corr_y = (h_orig - 1) - corr_y
                    
                self.handle_measurement_click(corr_x, corr_y)
                return
            
        # Color Balance Sampler Point Click Logic
        if self.balance_mode != "None":
            if 0 <= px_x < w_orig and 0 <= px_y < h_orig:
                corr_x, corr_y = px_x, px_y
                if self.mirror_horizontal:
                    corr_x = (w_orig - 1) - corr_x
                if self.mirror_vertical:
                    corr_y = (h_orig - 1) - corr_y
                    
                # Sample a larger 10x10 pixel patch
                x_min, x_max = max(0, corr_x - 5), min(w_orig, corr_x + 5)
                y_min, y_max = max(0, corr_y - 5), min(h_orig, corr_y + 5)
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
                    
                # Calculate the average values of the channels separately
                sampled_rgb = np.mean(patch_norm[valid_mask], axis=0)
                
                # Target dark gray level in FITS normalized scale [0..1]
                dark_gray_target = 0.05
                
                # Calculate individual channel offsets to align R, G, B peaks exactly at dark_gray_target
                # Using formula: offset = (sample - target) / (1.0 - target)
                offset_rgb = []
                for c in range(3):
                    val = (sampled_rgb[c] - dark_gray_target) / (1.0 - dark_gray_target)
                    val = max(-0.5, min(val, 0.95))
                    offset_rgb.append(float(val))
                
                self.sky_sample_rgb = offset_rgb
                messagebox.showinfo("Color Balance", 
                                    f"Sky Background calibrated (Neutral Gray offsets applied per channel):\n"
                                    f"Red offset={offset_rgb[0]:.4f}\n"
                                    f"Green offset={offset_rgb[1]:.4f}\n"
                                    f"Blue offset={offset_rgb[2]:.4f}\n"
                                    f"(Background channels aligned to {dark_gray_target*100:.1f}% neutral dark gray)")
                                    
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
            if self.btn_bal_sky:
                self.btn_bal_sky.config(bg="#374151")
            if self.btn_bal_star:
                self.btn_bal_star.config(bg="#374151")
            self.process_and_update(is_dragging=False)
            return

        if 0 <= canvas_x < self.img_scale_ratio * w and 0 <= canvas_y < self.img_scale_ratio * h:
            if self.crop_mode:
                self.crop_start = (event.x, event.y)
            elif self.annotation_mode:
                corr_x, corr_y = px_x, px_y
                if self.mirror_horizontal:
                    corr_x = (w_orig - 1) - corr_x
                if self.mirror_vertical:
                    corr_y = (h_orig - 1) - corr_y
                    
                # Calculate centroid of the star for perfect alignment
                img_gray = 0.2126 * self.debayered_cache[:,:,0] + 0.7152 * self.debayered_cache[:,:,1] + 0.0722 * self.debayered_cache[:,:,2]
                _, cx, cy = self.measure_aperture_flux(img_gray, corr_x, corr_y)
                ratio_x = cx / w_orig
                ratio_y = cy / h_orig
                
                initial_val = ""
                if self.wcs:
                    try:
                        orig_px_x = int(ratio_x * w_orig)
                        orig_px_y = int(ratio_y * h_orig)
                        
                        sky_coord = self.wcs.pixel_to_world(orig_px_x, orig_px_y)
                        coord_j2000 = sky_coord.transform_to(FK5(equinox='J2000'))
                        
                        ra_str = coord_j2000.ra.to_string(unit="hour", sep="hms", precision=1)
                        dec_str = coord_j2000.dec.to_string(unit="degree", sep="dms", precision=1)
                        initial_val = f"RA:{ra_str} DEC:{dec_str}"
                    except Exception:
                        pass
                
                import tkinter.simpledialog as sd
                text = sd.askstring("Add Annotation", "Star description / Coordinates:", parent=self.root, initialvalue=initial_val)
                if text is not None and text.strip() != "":
                    self.push_state()
                    self.annotations.append({
                        'x': ratio_x,
                        'y': ratio_y,
                        'text': text.strip()
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
            
            w_px1 = int(img_x1 / self.img_scale_ratio) + self.src_left_int
            h_px1 = int(img_y1 / self.img_scale_ratio) + self.src_top_int
            w_px2 = int(img_x2 / self.img_scale_ratio) + self.src_left_int
            h_px2 = int(img_y2 / self.img_scale_ratio) + self.src_top_int
            
            if self.mirror_horizontal:
                w_px1 = (w - 1) - w_px1
                w_px2 = (w - 1) - w_px2
            if self.mirror_vertical:
                h_px1 = (h - 1) - h_px1
                h_px2 = (h - 1) - h_px2
                
            start_x, end_x = min(w_px1, w_px2), max(w_px1, w_px2)
            start_y, end_y = min(h_px1, h_px2), max(h_px1, h_px2)
            
            # Align crop coordinates to even boundaries if bayer pattern is active to preserve Bayer phase alignment
            if self.bayer_pattern.get() != "None":
                start_x = (start_x // 2) * 2
                start_y = (start_y // 2) * 2
                end_x = (end_x // 2) * 2
                end_y = (end_y // 2) * 2
                
            # If already cropped, convert relative coordinates to absolute original image coordinates
            if self.visual_crop_box is not None:
                orig_start_x, orig_start_y, orig_end_x, orig_end_y = self.visual_crop_box
                start_x += orig_start_x
                start_y += orig_start_y
                end_x += orig_start_x
                end_y += orig_start_y
                
            if (end_x - start_x) > 5 and (end_y - start_y) > 5:
                self.push_state()
                self.visual_crop_box = (start_x, start_y, end_x, end_y)
                
                # Reset zoom and panning values to fit the new cropped area perfectly on screen
                self.zoom_level = 1.0
                self.pan_x = 0.0
                self.pan_y = 0.0
                
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
                
                self.push_state()
                # Find the centroid of the targeted star to center the target marker perfectly
                img_gray = 0.2126 * self.debayered_cache[:,:,0] + 0.7152 * self.debayered_cache[:,:,1] + 0.0722 * self.debayered_cache[:,:,2]
                _, cx, cy = self.measure_aperture_flux(img_gray, px_x, px_y)
                
                ra_str = coord_j2000.ra.to_string(unit="hour", sep="hms", precision=2)
                dec_str = coord_j2000.dec.to_string(unit="degree", sep="dms", precision=2)
                self.annotations.append({
                    'x': cx / w_orig,
                    'y': cy / h_orig,
                    'text': f"Target RA:{ra_str} DEC:{dec_str}"
                })
                
                canvas_w = self.canvas.winfo_width()
                canvas_h = self.canvas.winfo_height()
                fit_ratio = min(canvas_w / w_orig, canvas_h / h_orig)
                total_scale = fit_ratio * self.zoom_level
                
                target_draw_x = (w_orig - 1 - cx) if self.mirror_horizontal else cx
                target_draw_y = (h_orig - 1 - cy) if self.mirror_vertical else cy
                
                self.pan_x = (canvas_w / 2) - (target_draw_x * total_scale) - ((canvas_w - (w_orig * total_scale)) / 2)
                self.pan_y = (canvas_h / 2) - (target_draw_y * total_scale) - ((canvas_h - (h_orig * total_scale)) / 2)
                
                self.render_canvas(is_dragging=False)
                messagebox.showinfo("Target Marked", f"Star marked at pixels X={cx:.1f}, Y={cy:.1f}.\nCentering viewport.")
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
            
            # Determine size of crosshair based on image dimensions
            cross_size = int(max(w, h) * 0.015)
            cross_size = max(24, min(cross_size, 96))
            cross_width = max(2, int(cross_size * 0.06))
            
            # Load a high-res TTF font if possible.
            # Use a compact scale factor (exactly 45% of crosshair size) to match screen visual proportions
            font_size = max(10, int(cross_size * 0.45))
            
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
                
            # Coordinate mapping helper for export_image
            h_orig, w_orig = self.debayered_cache.shape[:2]
            
            def map_coords_export(px_x, px_y):
                x_val = px_x
                y_val = px_y
                if self.visual_crop_box is not None:
                    start_x, start_y, end_x, end_y = self.visual_crop_box
                    x_val = x_val - start_x
                    y_val = y_val - start_y
                    
                rx = x_val / w
                ry = y_val / h
                if self.mirror_horizontal:
                    rx = 1.0 - rx
                if self.mirror_vertical:
                    ry = 1.0 - ry
                    
                ax = int(rx * w)
                ay = int(ry * h)
                return ax, ay
                
            # 1. Draw Saved Annotations (if enabled)
            if self.show_annotations.get():
                for ann in self.annotations:
                    px = ann['x'] * w_orig
                    py = ann['y'] * h_orig
                    ax, ay = map_coords_export(px, py)
                    
                    if 0 <= ax < w and 0 <= ay < h:
                        self.draw_star_crosshair(draw, ax, ay, size=cross_size, width=cross_width)
                        
                        text_offset_x = int(cross_size * 0.6)
                        text_offset_y = int(cross_size * 0.6)
                        
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
                px = self.temp_marker['ratio_x'] * w_orig
                py = self.temp_marker['ratio_y'] * h_orig
                tx, ty = map_coords_export(px, py)
                
                if 0 <= tx < w and 0 <= ty < h:
                    temp_cross_size = int(cross_size * 1.2)
                    temp_cross_width = max(2, int(temp_cross_size * 0.06))
                    self.draw_star_crosshair(draw, tx, ty, size=temp_cross_size, width=temp_cross_width)
                    
                    text_offset_x = int(temp_cross_size * 0.6)
                    text_offset_y = int(temp_cross_size * 0.6)
                    
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

    def toggle_calibration_mode(self):
        if self.fits_data is None:
            messagebox.showerror("Error", "Load a FITS file first.")
            return
            
        self.calibration_mode = not self.calibration_mode
        if self.calibration_mode:
            self.measurement_mode = False
            self.crop_mode = False
            self.annotation_mode = False
            self.balance_mode = "None"
            self.gaia_search_mode = False
            self.gaia_search_var.set(False)
            self.btn_crop.config(text="Crop Mode: Off", bg="#374151")
            self.btn_annotate.config(text="Annotate Mode: Off", bg="#374151")
            self.btn_measurement.config(text="Measure Target: Off", bg="#374151")
            if self.btn_bal_sky:
                self.btn_bal_sky.config(bg="#374151")
            if self.btn_bal_star:
                self.btn_bal_star.config(bg="#374151")
            self.btn_calibration.config(text="Mark Calib Stars: On", bg="#16a34a")
            self.canvas.config(cursor="crosshair")
            
            if len(self.catalog_stars) == 0:
                download = messagebox.askyesno("Photometry", "No catalog stars loaded. Would you like to query Vizier for catalog stars now?", parent=self.root)
                if download:
                    self.download_catalog_stars()
        else:
            self.btn_calibration.config(text="Mark Calib Stars: Off", bg="#374151")
            self.canvas.config(cursor="")
            self.render_canvas(is_dragging=False)

    def toggle_measurement_mode(self):
        if self.fits_data is None:
            messagebox.showerror("Error", "Load a FITS file first.")
            return
            
        self.measurement_mode = not self.measurement_mode
        if self.measurement_mode:
            self.calibration_mode = False
            self.crop_mode = False
            self.annotation_mode = False
            self.balance_mode = "None"
            self.gaia_search_mode = False
            self.gaia_search_var.set(False)
            self.btn_crop.config(text="Crop Mode: Off", bg="#374151")
            self.btn_annotate.config(text="Annotate Mode: Off", bg="#374151")
            self.btn_calibration.config(text="Mark Calib Stars: Off", bg="#374151")
            if self.btn_bal_sky:
                self.btn_bal_sky.config(bg="#374151")
            if self.btn_bal_star:
                self.btn_bal_star.config(bg="#374151")
            self.btn_measurement.config(text="Measure Target: On", bg="#16a34a")
            self.canvas.config(cursor="crosshair")
            
            if len(self.calib_stars) == 0:
                messagebox.showwarning("Photometry", 
                                       "Please select at least one catalog calibration star first to establish the zero-point of the image.\n"
                                       "You can use 'Auto-Calibrate Photometry' or manually click catalog stars with 'Mark Calib Stars' active.",
                                       parent=self.root)
        else:
            self.btn_measurement.config(text="Measure Target: Off", bg="#374151")
            self.canvas.config(cursor="")
            self.render_canvas(is_dragging=False)

    def download_catalog_stars(self):
        if self.fits_data is None or self.wcs is None:
            messagebox.showerror("Error", "A FITS file with valid WCS headers must be loaded first.")
            return
            
        self.root.config(cursor="watch")
        self.root.update()
        
        # Create a progress log window overlay as requested
        log_win = tk.Toplevel(self.root)
        log_win.title("Vizier Query Monitor")
        log_win.geometry("600x400")
        log_win.configure(bg="#1f2937")
        log_win.transient(self.root)
        
        lbl = tk.Label(log_win, text="Querying Vizier Catalogs (APASS / Gaia)...", bg="#1f2937", fg="white", font=("Segoe UI", 10, "bold"))
        lbl.pack(pady=(10, 5))
        
        txt_log = tk.Text(log_win, bg="#111827", fg="#10b981", insertbackground="white", font=("Consolas", 9), bd=0, padx=10, pady=10)
        txt_log.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        def log_msg(msg):
            txt_log.insert(tk.END, msg + "\n")
            txt_log.see(tk.END)
            log_win.update()

        try:
            h_orig, w_orig = self.debayered_cache.shape[:2]
            
            # Target annotation or viewport center coordinate
            ra_deg, dec_deg = None, None
            if getattr(self, 'annotations', None) and self.wcs:
                ann = self.annotations[0]
                px = ann['x'] * w_orig
                py = ann['y'] * h_orig
                sky = self.wcs.pixel_to_world(px, py)
                ra_deg = sky.ra.deg
                dec_deg = sky.dec.deg
                log_msg(f"[INFO] Targeting annotation: RA={ra_deg:.6f} deg, DEC={dec_deg:.6f} deg")
            else:
                canvas_w = self.canvas.winfo_width()
                canvas_h = self.canvas.winfo_height()
                cx = canvas_w / 2.0
                cy = canvas_h / 2.0
                img_x = (cx - self.img_offset_x) / self.img_scale_ratio
                img_y = (cy - self.img_offset_y) / self.img_scale_ratio
                if self.mirror_horizontal:
                    img_x = (w_orig - 1) - img_x
                if self.mirror_vertical:
                    img_y = (h_orig - 1) - img_y
                center_coord = self.wcs.pixel_to_world(img_x, img_y)
                ra_deg = center_coord.ra.deg
                dec_deg = center_coord.dec.deg
                log_msg(f"[INFO] Targeting viewport center: RA={ra_deg:.6f} deg, DEC={dec_deg:.6f} deg")
                
            # Viewport radius calculation
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()
            diag_canvas_px = np.sqrt(canvas_w**2 + canvas_h**2) / 2.0
            diag_img_px = diag_canvas_px / self.img_scale_ratio
            from astropy.wcs.utils import proj_plane_pixel_scales
            scales = proj_plane_pixel_scales(self.wcs)
            pixel_scale_deg = np.mean(scales)
            radius_deg = diag_img_px * pixel_scale_deg
            radius_arcmin = radius_deg * 60.0
            radius_arcmin = np.clip(radius_arcmin, 3.0, 45.0)
            log_msg(f"[INFO] Viewport query radius: {radius_arcmin:.2f} arcminutes")
            
            import urllib.request
            import urllib.parse
            import ssl
            
            ssl_context = ssl._create_unverified_context()
            
            c_val = f"{ra_deg:.6f} {dec_deg:.6f}".replace(',', '.')
            c_str = urllib.parse.quote(c_val)
            catalog_stars = []
            
            hosts = ["vizier.cfa.harvard.edu", "vizier.cds.unistra.fr"]
            
            for host in hosts:
                log_msg(f"\n[QUERY] Connecting to mirror: {host}...")
                
                # 1. Try APASS DR9
                url = f"https://{host}/viz-bin/asu-tsv?-source=II/336/apass9&-c={c_str}&-c.r={radius_arcmin:f}&-c.u=arcmin&-out.form=|&-out.add=RAJ2000,DEJ2000&-out=Vmag,Bmag,e_Vmag,e_Bmag&-sort=_r&-out.max=1000"
                log_msg(f"[QUERY] Sending APASS request:\n  {url}")
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                
                lines = []
                try:
                    with urllib.request.urlopen(req, timeout=8, context=ssl_context) as response:
                        lines = response.read().decode('utf-8').split('\n')
                    log_msg(f"[RESPONSE] APASS query returned {len(lines)} raw lines.")
                except Exception as ex:
                    log_msg(f"[WARN] APASS request failed or timed out: {ex}")
                    
                # Check for database/connection errors in response
                is_error = False
                for line in lines[:30]:
                    if "Error" in line or "not reachable" in line or "connection" in line:
                        is_error = True
                        break
                if is_error:
                    log_msg(f"[WARN] Database connection error reported by mirror: {host}")
                    lines = []
                    
                # 2. Try Gaia DR3 if APASS returned nothing or failed
                if len(lines) == 0:
                    url = f"https://{host}/viz-bin/asu-tsv?-source=I/355/gaiadr3&-c={c_str}&-c.r={radius_arcmin:f}&-c.u=arcmin&-out.form=|&-out.add=RA_ICRS,DE_ICRS&-out=Gmag,bp_rp,phot_variable_flag&-sort=_r&-out.max=1000"
                    log_msg(f"[QUERY] APASS failed. Trying Gaia DR3 query:\n  {url}")
                    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                    try:
                        with urllib.request.urlopen(req, timeout=8, context=ssl_context) as response:
                            lines = response.read().decode('utf-8').split('\n')
                        log_msg(f"[RESPONSE] Gaia query returned {len(lines)} raw lines.")
                    except Exception as ex:
                        log_msg(f"[WARN] Gaia request failed: {ex}")
                        continue
                
                # Check for database/connection errors in Gaia response
                is_error = False
                for line in lines[:30]:
                    if "Error" in line or "not reachable" in line or "connection" in line:
                        is_error = True
                        break
                if is_error:
                    log_msg(f"[WARN] Database connection error on Gaia query for host: {host}")
                    continue
                    
                # Parse lines using '|' as separator
                header_found = False
                cols = {}
                host_stars = []
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split('|')
                    if not header_found:
                        if any("RA" in p or "coord" in p or "_RAJ" in p for p in parts):
                            header_found = True
                            cols = {p.strip(): i for i, p in enumerate(parts)}
                            log_msg(f"[PARSER] Header columns resolved: {cols}")
                        continue
                    if line.startswith("-"):
                        continue
                    if len(parts) >= 3:
                        try:
                            ra_idx = cols.get("RAJ2000", cols.get("_RAJ2000", cols.get("RA_ICRS", cols.get("_RA_ICRS", 0))))
                            dec_idx = cols.get("DEJ2000", cols.get("_DEJ2000", cols.get("DE_ICRS", cols.get("_DE_ICRS", 1))))
                            
                            ra_star = float(parts[ra_idx])
                            dec_star = float(parts[dec_idx])
                            
                            is_variable = False
                            color_index = 0.0
                            mag = 0.0
                            
                            if "Vmag" in cols:
                                v_idx = cols["Vmag"]
                                b_idx = cols.get("Bmag")
                                ev_idx = cols.get("e_Vmag")
                                eb_idx = cols.get("e_Bmag")
                                
                                mag_str = parts[v_idx].strip()
                                if mag_str and mag_str != "" and mag_str != "---":
                                    mag = float(mag_str)
                                else:
                                    continue
                                    
                                b_str = parts[b_idx].strip() if b_idx is not None else ""
                                ev_str = parts[ev_idx].strip() if ev_idx is not None else ""
                                eb_str = parts[eb_idx].strip() if eb_idx is not None else ""
                                
                                if b_str and b_str != "" and b_str != "---":
                                    color_index = float(b_str) - mag
                                else:
                                    color_index = 0.6
                                    
                                if ev_str and ev_str != "" and ev_str != "---":
                                    if float(ev_str) > 0.15:
                                        is_variable = True
                                if eb_str and eb_str != "" and eb_str != "---":
                                    if float(eb_str) > 0.15:
                                        is_variable = True
                            else:
                                g_idx = cols.get("Gmag", cols.get("Phot.G", 2))
                                bprp_idx = cols.get("bp_rp")
                                var_idx = cols.get("phot_variable_flag")
                                
                                mag_str = parts[g_idx].strip()
                                if mag_str and mag_str != "" and mag_str != "---":
                                    mag = float(mag_str)
                                else:
                                    continue
                                    
                                if bprp_idx is not None:
                                    bprp_str = parts[bprp_idx].strip()
                                    if bprp_str and bprp_str != "" and bprp_str != "---":
                                        color_index = float(bprp_str)
                                    else:
                                        color_index = 0.8
                                else:
                                    color_index = 0.8
                                    
                                if var_idx is not None:
                                    var_str = parts[var_idx].strip()
                                    if "VARIABLE" in var_str:
                                        is_variable = True
                                        
                            host_stars.append({
                                'ra': ra_star,
                                'dec': dec_star,
                                'mag': mag,
                                'color_index': color_index,
                                'is_variable': is_variable
                            })
                        except Exception:
                            continue
                            
                log_msg(f"[PARSER] Successfully parsed {len(host_stars)} stars from mirror {host}.")
                
                # If we parsed stars, we're done!
                if len(host_stars) > 0:
                    if len(host_stars) > 300:
                        step = len(host_stars) // 300
                        catalog_stars = host_stars[::step][:300]
                        log_msg(f"[INFO] Limited catalog to 300 stars (uniformly sampled by distance/magnitude) to boost performance.")
                    else:
                        catalog_stars = host_stars
                    break
            
            self.catalog_stars = catalog_stars
            
            # Print a debug list of 10 stars with projected pixel coordinates
            log_msg("\n[DEBUG] Sample of 10 parsed stars and their projected pixels:")
            header_epoch = self.fits_header.get('EQUINOX', 2000.0)
            for idx, star in enumerate(catalog_stars[:10]):
                try:
                    input_coord = SkyCoord(ra=star['ra']*u.deg, dec=star['dec']*u.deg, frame='icrs')
                    if header_epoch == 2000.0:
                        target_wcs_coord = input_coord.transform_to(FK5(equinox='J2000'))
                    else:
                        target_wcs_coord = input_coord.transform_to(FK5(equinox=self.observation_time))
                    px_x, px_y = self.wcs.world_to_pixel(target_wcs_coord)
                    log_msg(f"  Star {idx+1}: RA={star['ra']:.5f}, DEC={star['dec']:.5f}, Mag={star['mag']:.2f} -> Pixel: x={px_x:.1f}, y={px_y:.1f} (PIL y={h_orig - 1 - px_y:.1f})")
                except Exception as ex:
                    log_msg(f"  Star {idx+1}: RA={star['ra']:.5f}, DEC={star['dec']:.5f} -> Projection failed: {ex}")
            
            self.show_catalog_stars.set(True)
            self.render_canvas(is_dragging=False)
            log_msg(f"\n[SUCCESS] Download completed! {len(catalog_stars)} stars catalogued.")
            messagebox.showinfo("Catalog", f"Successfully downloaded {len(catalog_stars)} stars from Vizier catalog.")
            
            if len(catalog_stars) > 0:
                auto_cal = messagebox.askyesno("Auto-Calibration", "Would you like to run 'Auto-Calibrate Photometry' right now using these downloaded stars?", parent=self.root)
                if auto_cal:
                    self.auto_calibrate_photometry()
            
        except Exception as e:
            log_msg(f"\n[ERROR] Process failed: {e}")
            messagebox.showerror("Error", f"Failed to download catalog stars: {e}")
        finally:
            self.root.config(cursor="")

    def query_mpc_asteroids(self):
        if self.fits_data is None or self.wcs is None:
            messagebox.showerror("Error", "An image with valid WCS headers must be loaded first.")
            return
            
        # Open Observation Time Dialog
        default_time_str = ""
        if 'DATE-OBS' in self.fits_header:
            default_time_str = str(self.fits_header['DATE-OBS'])
        elif getattr(self, 'observation_time', None) is not None:
            default_time_str = self.observation_time.isot
        else:
            from astropy.time import Time
            default_time_str = Time.now().isot
            
        dialog = tk.Toplevel(self.root)
        dialog.title("Observation Time")
        dialog.geometry("380x180")
        dialog.configure(bg=self.panel_color)
        dialog.transient(self.root)
        dialog.grab_set()
        
        lbl_info = tk.Label(dialog, text="Verify Observation Time (UTC ISO format):\nYYYY-MM-DDTHH:MM:SS", bg=self.panel_color, fg=self.text_color, font=("Segoe UI", 9, "bold"))
        lbl_info.pack(pady=15)
        
        ent_time = tk.Entry(dialog, bg=self.bg_color, fg=self.text_color, insertbackground="white", bd=0, highlightthickness=1, highlightbackground=self.control_bg, width=30)
        ent_time.insert(0, default_time_str)
        ent_time.pack(pady=5)
        
        confirmed_time = [None]
        
        def on_confirm():
            time_str = ent_time.get().strip()
            from astropy.time import Time
            try:
                parsed = Time(time_str, format='isot')
                confirmed_time[0] = parsed
                dialog.destroy()
            except Exception as parse_err:
                messagebox.showerror("Invalid Date", f"Please enter a valid ISO date-time string:\n{parse_err}", parent=dialog)
                
        btn_frame = tk.Frame(dialog, bg=self.panel_color)
        btn_frame.pack(pady=10, fill="x")
        
        btn_ok = tk.Button(btn_frame, text="Confirm & Search", command=on_confirm, bg="#10b981", fg="white", font=("Segoe UI", 9, "bold"), bd=0, padx=15, pady=5)
        btn_ok.pack(side="right", padx=20)
        
        btn_cancel = tk.Button(btn_frame, text="Cancel", command=dialog.destroy, bg="#374151", fg="white", font=("Segoe UI", 9, "bold"), bd=0, padx=15, pady=5)
        btn_cancel.pack(side="right", padx=10)
        
        self.root.wait_window(dialog)
        
        if confirmed_time[0] is None:
            return
            
        self.observation_time = confirmed_time[0]
        self.fits_header['DATE-OBS'] = self.observation_time.isot
        
        ext = os.path.splitext(self.fits_path)[1].lower()
        if ext not in ['.fits', '.fit']:
            sidecar_path = self.fits_path + ".info.json"
            import json
            try:
                sidecar_data = {}
                if os.path.exists(sidecar_path):
                    with open(sidecar_path, "r", encoding="utf-8") as sf:
                        sidecar_data = json.load(sf)
                sidecar_data["header"] = dict(self.fits_header)
                with open(sidecar_path, "w", encoding="utf-8") as sf:
                    json.dump(sidecar_data, sf, indent=4)
            except Exception:
                pass
                
        self.root.config(cursor="watch")
        self.root.update()
        
        # Create a progress log window overlay
        log_win = tk.Toplevel(self.root)
        log_win.title("MPC Asteroid Query Monitor")
        log_win.geometry("600x400")
        log_win.configure(bg="#1f2937")
        log_win.transient(self.root)
        
        lbl = tk.Label(log_win, text="Querying SkyBoT (IMCCE/MPC Resolver) for Asteroids...", bg="#1f2937", fg="white", font=("Segoe UI", 10, "bold"))
        lbl.pack(pady=(10, 5))
        
        txt_log = tk.Text(log_win, bg="#111827", fg="#f43f5e", insertbackground="white", font=("Consolas", 9), bd=0, padx=10, pady=10)
        txt_log.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        def log_msg(msg):
            txt_log.insert(tk.END, msg + "\n")
            txt_log.see(tk.END)
            log_win.update()

        try:
            h_orig, w_orig = self.debayered_cache.shape[:2]
            
            # Target the exact center of the full FITS image
            center_coord = self.wcs.pixel_to_world(w_orig / 2.0, h_orig / 2.0)
            ra_deg = center_coord.ra.deg
            dec_deg = center_coord.dec.deg
            log_msg(f"[INFO] Targeting full image center: RA={ra_deg:.6f} deg, DEC={dec_deg:.6f} deg")
            
            # Diagonal radius calculation for the full FITS footprint
            corner_coord = self.wcs.pixel_to_world(0, 0)
            radius_deg = center_coord.separation(corner_coord).deg
            # Limit the query radius to a reasonable maximum (e.g., 3.0 degrees) to prevent timeouts
            radius_deg = np.clip(radius_deg, 0.05, 3.0)
            log_msg(f"[INFO] Full image FOV query radius: {radius_deg:.3f} degrees")
            
            # 3. Determine epoch/time of observation
            epoch_str = "now"
            if getattr(self, 'observation_time', None) is not None:
                epoch_str = self.observation_time.isot.replace('T', ' ')
            elif 'DATE-OBS' in self.fits_header:
                epoch_str = self.fits_header['DATE-OBS'].replace('T', ' ')
            log_msg(f"[INFO] Epoch of observation: {epoch_str}")
            
            import urllib.request
            import urllib.parse
            import ssl
            import json
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            
            # Build URL parameters for SkyBoT conesearch API
            params = {
                '-ra': f"{ra_deg:.6f}",
                '-dec': f"{dec_deg:.6f}",
                '-sr': f"{radius_deg:.4f}",
                '-ep': epoch_str,
                '-mime': 'json',
                '-output': 'all',
                '-from': 'FitsManager'
            }
            
            query_str = urllib.parse.urlencode(params)
            url = f"https://ssp.imcce.fr/webservices/skybot/api/conesearch.php?{query_str}"
            log_msg(f"[QUERY] Sending request to IMCCE SkyBoT:\n  {url}")
            
            ssl_context = ssl._create_unverified_context()
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            
            with urllib.request.urlopen(req, timeout=20, context=ssl_context) as response:
                status_code = response.status
                if status_code == 204:
                    log_msg("\n[INFO] Response code 204: No solar system objects found in this region at this time.")
                    self.asteroid_objects = []
                    messagebox.showinfo("SkyBoT", "No asteroids found in this field of view at the epoch of observation.")
                    return
                elif status_code != 200:
                    raise Exception(f"HTTP Server returned status code {status_code}")
                    
                content = response.read().decode('utf-8')
                
            parsed = json.loads(content)
            log_msg(f"[RESPONSE] Successfully parsed {len(parsed)} objects.")
            
            # Filter objects by magnitude < 21
            filtered_asteroids = []
            for item in parsed:
                mag = item.get("VMag (mag)", item.get("V"))
                if mag is not None:
                    try:
                        mag_val = float(mag)
                        if mag_val < 21.0:
                            filtered_asteroids.append({
                                'name': item.get('Name', 'Unknown'),
                                'ra_hms': item.get('RA (hms)', item.get('RA (hour)')),
                                'dec_dms': item.get('DEC (dms)', item.get('DEC (deg)')),
                                'mag': mag_val,
                                'class': item.get('Class', '')
                            })
                    except ValueError:
                        pass
                        
            # Project celestial coordinates of filtered asteroids to pixel locations
            log_msg(f"\n[INFO] Found {len(filtered_asteroids)} asteroids with Mag < 21.0. Projecting...")
            
            self.asteroid_objects = filtered_asteroids
            self.show_asteroids.set(True)
            self.render_canvas(is_dragging=False)
            
            log_msg(f"\n[SUCCESS] Completed! {len(filtered_asteroids)} asteroids plotted on the image.")
            messagebox.showinfo("SkyBoT MPC", f"Found and plotted {len(filtered_asteroids)} asteroids with Mag < 21.")
            
        except Exception as e:
            log_msg(f"\n[ERROR] Process failed: {e}")
            messagebox.showerror("Error", f"Failed to query asteroids: {e}")
        finally:
            self.root.config(cursor="")

    def find_centroid(self, data, px, py, box_radius=5):
        px_val = float(np.atleast_1d(px)[0])
        py_val = float(np.atleast_1d(py)[0])
        iy, ix = int(round(py_val)), int(round(px_val))
        h, w = data.shape
        
        # 1. Local maximum search to locate the peak of luminosity in a 15x15 area
        search_r = 7
        y_min_s, y_max_s = max(0, iy - search_r), min(h, iy + search_r + 1)
        x_min_s, x_max_s = max(0, ix - search_r), min(w, ix + search_r + 1)
        search_patch = data[y_min_s:y_max_s, x_min_s:x_max_s]
        
        if search_patch.size > 0:
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(search_patch)
            peak_x = x_min_s + max_loc[0]
            peak_y = y_min_s + max_loc[1]
            iy, ix = int(round(peak_y)), int(round(peak_x))
            
        # 2. Fine centroiding on a smaller box around the peak
        y_min, y_max = max(0, iy - box_radius), min(h, iy + box_radius + 1)
        x_min, x_max = max(0, ix - box_radius), min(w, ix + box_radius + 1)
        patch = data[y_min:y_max, x_min:x_max]
        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        total_val = patch.sum()
        if total_val > 0.0001:
            cx = (patch * xx).sum() / total_val
            cy = (patch * yy).sum() / total_val
            return float(cx), float(cy)
        return float(ix), float(iy)

    def measure_aperture_flux(self, data, px, py, r_ap=6, r_in=10, r_out=15):
        centroid_x, centroid_y = self.find_centroid(data, px, py, box_radius=4)
        iy, ix = int(round(centroid_y)), int(round(centroid_x))
        crop_size = r_out + 2
        h, w = data.shape
        y_min, y_max = max(0, iy - crop_size), min(h, iy + crop_size + 1)
        x_min, x_max = max(0, ix - crop_size), min(w, ix + crop_size + 1)
        patch = data[y_min:y_max, x_min:x_max]
        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        dists = np.hypot(xx - centroid_x, yy - centroid_y)
        source_mask = dists <= r_ap
        bg_mask = (dists >= r_in) & (dists <= r_out)
        bg_pixels = patch[bg_mask]
        if len(bg_pixels) > 0:
            bg_level = np.median(bg_pixels)
        else:
            bg_level = 0.0
        source_sum = patch[source_mask].sum()
        num_source_pixels = source_mask.sum()
        net_flux = source_sum - (num_source_pixels * bg_level)
        return max(0.0001, net_flux), centroid_x, centroid_y

    def handle_calibration_click(self, px_x, px_y):
        h_orig, w_orig = self.debayered_cache.shape[:2]
        img_gray = 0.2126 * self.debayered_cache[:,:,0] + 0.7152 * self.debayered_cache[:,:,1] + 0.0722 * self.debayered_cache[:,:,2]
        flux, cx, cy = self.measure_aperture_flux(img_gray, px_x, px_y)
        
        if flux <= 1.0:
            messagebox.showwarning("Calibration Warning", 
                                   f"Measured net flux is too low or negative ({flux:.1f}). This star cannot be used for calibration.", 
                                   parent=self.root)
            return
        
        matched_star = None
        if self.wcs:
            clicked_sky = self.wcs.pixel_to_world(cx, cy)
            best_dist = 999.0
            for star in self.catalog_stars:
                star_coord = SkyCoord(ra=star['ra']*u.deg, dec=star['dec']*u.deg, frame='icrs')
                dist_arcsec = clicked_sky.separation(star_coord).arcsec
                if dist_arcsec < 15.0 and dist_arcsec < best_dist:
                    best_dist = dist_arcsec
                    matched_star = star
                    
        if matched_star is not None:
            if matched_star.get('is_variable'):
                messagebox.showwarning("Stella Variabile", 
                                       "Questa stella è contrassegnata come variabile e non può essere usata per la calibrazione fotometrica.", 
                                       parent=self.root)
                return
            existing_idx = None
            for idx, s in enumerate(self.calib_stars):
                if np.hypot(s['x'] - cx, s['y'] - cy) < 5:
                    existing_idx = idx
                    break
            if existing_idx is not None:
                self.calib_stars.pop(existing_idx)
                self.render_canvas(is_dragging=False)
                messagebox.showinfo("Photometry Calibration", 
                                    f"Removed calibration star.\n"
                                    f"Total calibration stars: {len(self.calib_stars)}")
                return
            flux_r, _, _ = self.measure_aperture_flux(self.debayered_cache[:,:,0], px_x, px_y)
            flux_b, _, _ = self.measure_aperture_flux(self.debayered_cache[:,:,2], px_x, px_y)
            self.calib_stars.append({
                'x': cx,
                'y': cy,
                'flux': flux,
                'flux_r': flux_r,
                'flux_b': flux_b,
                'mag': matched_star['mag'],
                'color_index': matched_star.get('color_index', 0.0)
            })
            self.render_canvas(is_dragging=False)
            messagebox.showinfo("Photometry Calibration", 
                                f"Added Calibration Star:\n"
                                f"Catalog Magnitude: {matched_star['mag']:.2f}\n"
                                f"Measured Net Flux: {flux:.1f}\n"
                                f"Total calibration stars: {len(self.calib_stars)}")
        else:
            messagebox.showwarning("Calibration Error", 
                                   "No catalog star found near this coordinate to use for calibration.\n"
                                   "Ensure catalog stars overlay is visible and you click close to a catalog star circle.",
                                   parent=self.root)

    def handle_measurement_click(self, px_x, px_y):
        if len(self.calib_stars) == 0:
            messagebox.showwarning("Measurement Error", 
                                   "Please establish the zero-point calibration first.\n"
                                   "Either use 'Auto-Calibrate Photometry' or manually add calibration stars using 'Mark Calib Stars'.",
                                   parent=self.root)
            return

        h_orig, w_orig = self.debayered_cache.shape[:2]
        img_gray = 0.2126 * self.debayered_cache[:,:,0] + 0.7152 * self.debayered_cache[:,:,1] + 0.0722 * self.debayered_cache[:,:,2]
        flux, cx, cy = self.measure_aperture_flux(img_gray, px_x, px_y)
        
        if flux <= 1.0:
            messagebox.showwarning("Measurement Warning", 
                                   f"Measured net flux is too low or negative ({flux:.1f}). Cannot measure magnitude.", 
                                   parent=self.root)
            return
            
        zp_list = []
        for s in self.calib_stars:
            zp_val = s['mag'] + 2.5 * np.log10(s['flux'])
            zp_list.append(zp_val)
            
        num_calib = len(self.calib_stars)
        has_color_correction = False
        zp = np.mean(zp_list)
        c1 = 0.0
        
        if num_calib >= 3:
            colors = np.array([s.get('color_index', 0.0) for s in self.calib_stars])
            zps = np.array(zp_list)
            try:
                slope, intercept = np.polyfit(colors, zps, 1)
                c1 = slope
                zp = intercept
                has_color_correction = True
            except Exception:
                pass
                
        # Look up if this star is in the catalog to get its color index automatically
        matched_star = None
        if self.wcs:
            clicked_sky = self.wcs.pixel_to_world(cx, cy)
            best_dist = 999.0
            for star in self.catalog_stars:
                star_coord = SkyCoord(ra=star['ra']*u.deg, dec=star['dec']*u.deg, frame='icrs')
                dist_arcsec = clicked_sky.separation(star_coord).arcsec
                if dist_arcsec < 15.0 and dist_arcsec < best_dist:
                    best_dist = dist_arcsec
                    matched_star = star
                    
        is_gaia = True
        if self.catalog_stars and 'Vmag' in self.catalog_stars[0]:
            is_gaia = False
            
        default_color = 0.8 if is_gaia else 0.6
        color_name = "BP-RP (Gaia)" if is_gaia else "B-V (APASS)"
        
        color_source_info = ""
        target_color = default_color
        dialog_needed = True
        
        # Check if the image is a color FIT to perform color index measurement & interpolation
        is_color_image = False
        if len(self.debayered_cache.shape) == 3 and self.debayered_cache.shape[2] == 3:
            is_color_image = np.max(np.abs(self.debayered_cache[:,:,0] - self.debayered_cache[:,:,2])) > 0.005
            
        if is_color_image:
            calib_with_colors = []
            for s in self.calib_stars:
                if s.get('flux_r') is not None and s.get('flux_b') is not None:
                    fr = s['flux_r']
                    fb = s['flux_b']
                    if fr > 1.0 and fb > 1.0:
                        instr_ci = -2.5 * np.log10(fb / fr)
                        calib_with_colors.append({
                            'instr_ci': instr_ci,
                            'true_ci': s.get('color_index', 0.0)
                        })
            if len(calib_with_colors) >= 2:
                x_pts = np.array([item['instr_ci'] for item in calib_with_colors])
                y_pts = np.array([item['true_ci'] for item in calib_with_colors])
                try:
                    slope_c, intercept_c = np.polyfit(x_pts, y_pts, 1)
                    flux_r, _, _ = self.measure_aperture_flux(self.debayered_cache[:,:,0], px_x, px_y)
                    flux_b, _, _ = self.measure_aperture_flux(self.debayered_cache[:,:,2], px_x, px_y)
                    if flux_r > 1.0 and flux_b > 1.0:
                        target_instr_ci = -2.5 * np.log10(flux_b / flux_r)
                        target_color = slope_c * target_instr_ci + intercept_c
                        target_color = float(np.clip(target_color, -0.5, 3.0))
                        color_source_info = f"{target_color:.2f} (interpolated from color FIT: B/R flux ratio)"
                        dialog_needed = False
                except Exception:
                    pass
                    
        if dialog_needed:
            # Fallback 1: Silent lookup from catalog if it is a known star
            if matched_star is not None and matched_star.get('color_index') is not None:
                try:
                    target_color = float(matched_star['color_index'])
                    color_source_info = f"{target_color:.2f} (loaded automatically from catalog)"
                except ValueError:
                    dialog = ColorIndexDialog(self.root, is_gaia=is_gaia, default_val=default_color)
                    target_color = dialog.result
                    color_source_info = f"{target_color:.2f} (manually selected)"
            else:
                # Fallback 2: Unknown target and monochrome FIT. Open dialog!
                dialog = ColorIndexDialog(self.root, is_gaia=is_gaia, default_val=default_color)
                target_color = dialog.result
                color_source_info = f"{target_color:.2f} (manually selected, unknown target)"
                
        if has_color_correction:
            mag_u = -2.5 * np.log10(flux) + zp + c1 * target_color
        else:
            mag_u = -2.5 * np.log10(flux) + zp
            
        # Estimate likelihood percentages based on the resolved color index
        c_gaia = target_color if is_gaia else (1.3 * target_color)
        w_ast = np.exp(-0.5 * ((c_gaia - 0.8) / 0.15)**2)
        w_sn = np.exp(-0.5 * ((c_gaia - 0.2) / 0.15)**2)
        w_nova = np.exp(-0.5 * ((c_gaia - 0.6) / 0.25)**2)
        w_red = np.exp(-0.5 * ((c_gaia - 2.2) / 0.40)**2)
        w_blue = np.exp(-0.5 * ((c_gaia - (-0.3)) / 0.25)**2)
        
        total_w = w_ast + w_sn + w_nova + w_red + w_blue
        if total_w < 1e-6:
            total_w = 1e-6
            
        p_ast = (w_ast / total_w) * 100.0
        p_sn = (w_sn / total_w) * 100.0
        p_nova = (w_nova / total_w) * 100.0
        p_red = (w_red / total_w) * 100.0
        p_blue = (w_blue / total_w) * 100.0
        
        candidates = [
            ("Asteroid / Solar-type Yellow Dwarf", p_ast),
            ("Supernova (Typical Ia near Max)", p_sn),
            ("Nova (Typical / Early Outburst)", p_nova),
            ("Red Dwarf / M-Class Star", p_red),
            ("Hot Blue Giant / O-Class Star", p_blue)
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        prob_str = "--- Target Classification Probability Estimate ---\n"
        prob_str += "\n".join([f"• {name}: {prob:.1f}%" for name, prob in candidates if prob >= 1.0])
            
        coord_text = ""
        if self.wcs:
            sky_coord = self.wcs.pixel_to_world(cx, cy)
            j2000 = sky_coord.transform_to(FK5(equinox='J2000'))
            ra_str = j2000.ra.to_string(unit="hour", sep="hms", precision=1)
            dec_str = j2000.dec.to_string(unit="degree", sep="dms", precision=1)
            coord_text = f" (RA:{ra_str} DEC:{dec_str})"
            
        self.push_state()
        self.annotations.append({
            'x': cx / w_orig,
            'y': cy / h_orig,
            'text': f"Target Mag={mag_u:.2f}{coord_text}"
        })
        self.render_canvas(is_dragging=False)
        
        zp_details = "\n".join([f"Star {i+1}: Mag={s['mag']:.2f}, CI={s.get('color_index',0.0):.2f}, ZP={s['mag'] + 2.5 * np.log10(s['flux']):.2f}" for i, s in enumerate(self.calib_stars)])
        
        if has_color_correction:
            calib_msg = (
                f"Photometry Measurement Successful with Color Correction!\n\n"
                f"Fitted Equation: Mag = -2.5*log10(Flux) + {zp:.3f} + ({c1:.3f}) * ColorIndex\n"
                f"Zero-Point (ZP): {zp:.3f}\n"
                f"Color Term (C1): {c1:.3f}\n\n"
                f"Target Star Color Index ({color_name}): {color_source_info}\n"
                f"Target Star Measured Flux: {flux:.1f}\n"
                f"Target Star Calibrated Mag: {mag_u:.2f}\n\n"
                f"{prob_str}\n\n"
                f"Annotation added to image."
            )
        else:
            calib_msg = (
                f"Photometry Measurement Successful!\n\n"
                f"Calibration Zero-Points:\n{zp_details}\n"
                f"Average Zero-Point (ZP): {zp:.3f}\n\n"
                f"Target Star Measured Flux: {flux:.1f}\n"
                f"Target Star Calibrated Mag: {mag_u:.2f}\n\n"
                f"{prob_str}\n\n"
                f"Annotation added to image."
            )
            
        messagebox.showinfo("Measurement Result", calib_msg)
        
    def get_cached_tile(self, ra, dec, size):
        import os
        import numpy as np
        if not os.path.exists("dss_cache"):
            return None
        for filename in os.listdir("dss_cache"):
            if filename.startswith("dss_") and filename.endswith(".fits"):
                parts = filename.replace(".fits", "").split("_")
                if len(parts) == 4:
                    try:
                        file_ra = float(parts[1])
                        file_dec = float(parts[2])
                        file_size = float(parts[3])
                        
                        # High-speed flat-plane angular distance approximation (nanoseconds vs 100ms SkyCoord)
                        d_dec = dec - file_dec
                        d_ra = ((ra - file_ra) * np.cos(np.radians((dec + file_dec) / 2.0))) % 360.0
                        if d_ra > 180.0:
                            d_ra = 360.0 - d_ra
                        dist_arcmin = np.sqrt(d_ra**2 + d_dec**2) * 60.0
                        
                        # Increased matching tolerance to 10 arcminutes to reuse nearby caches
                        if dist_arcmin < 10.0 and file_size >= size - 0.1:
                            return os.path.join("dss_cache", filename)
                    except Exception:
                        pass
        return None

    def is_coord_covered(self, ra, dec):
        import numpy as np
        for tile in self.loaded_dss_tiles:
            tile_ra = tile.get('ra')
            tile_dec = tile.get('dec')
            if tile_ra is not None and tile_dec is not None:
                d_dec = dec - tile_dec
                d_ra = ((ra - tile_ra) * np.cos(np.radians((dec + tile_dec) / 2.0))) % 360.0
                if d_ra > 180.0:
                    d_ra = 360.0 - d_ra
                dist_arcmin = np.sqrt(d_ra**2 + d_dec**2) * 60.0
                # A 45 arcminute tile has a half-width of 22.5 arcminutes.
                # If coordinate is within 20 arcminutes of the center, it is covered!
        return False

    def get_cached_panstarrs_tile(self, ra, dec, size):
        import os
        import numpy as np
        if not os.path.exists("panstarrs_cache"):
            return None
        for filename in os.listdir("panstarrs_cache"):
            if filename.startswith("panstarrs_") and filename.endswith(".fits"):
                parts = filename.replace(".fits", "").split("_")
                if len(parts) == 4:
                    try:
                        file_ra = float(parts[1])
                        file_dec = float(parts[2])
                        file_size = float(parts[3])
                        
                        d_dec = dec - file_dec
                        d_ra = ((ra - file_ra) * np.cos(np.radians((dec + file_dec) / 2.0))) % 360.0
                        if d_ra > 180.0:
                            d_ra = 360.0 - d_ra
                        dist_arcmin = np.sqrt(d_ra**2 + d_dec**2) * 60.0
                        
                        if dist_arcmin < 5.0 and file_size >= size - 1.0:
                            return os.path.join("panstarrs_cache", filename)
                    except Exception:
                        pass
        return None

    def download_panstarrs_background(self):
        if self.fits_data is None or self.wcs is None:
            messagebox.showerror("Error", "A FITS file with valid WCS headers must be loaded first.", parent=self.root)
            return
            
        self.root.config(cursor="watch")
        self.root.update()
        
        log_win = tk.Toplevel(self.root)
        log_win.title("PanSTARRS Cutout Service Monitor")
        log_win.geometry("600x450")
        log_win.configure(bg="#1f2937")
        log_win.transient(self.root)
        
        lbl = tk.Label(log_win, text="Retrieving PanSTARRS DR2 Reference Survey Image...", bg="#1f2937", fg="white", font=("Segoe UI", 10, "bold"))
        lbl.pack(pady=(10, 5))
        
        from tkinter import ttk
        style = ttk.Style(log_win)
        style.theme_use('clam')
        style.configure("PS1.Horizontal.TProgressbar", troughcolor="#111827", background="#10b981", bordercolor="#1f2937", lightcolor="#10b981", darkcolor="#10b981")
        
        progress_bar = ttk.Progressbar(log_win, style="PS1.Horizontal.TProgressbar", orient="horizontal", mode="determinate", length=560)
        progress_bar.pack(pady=5, padx=15, fill="x")
        
        txt_log = tk.Text(log_win, bg="#111827", fg="#10b981", insertbackground="white", font=("Consolas", 9), bd=0, padx=10, pady=10)
        
        self.dss_download_cancelled = False
        
        def log_msg(msg):
            def _log():
                txt_log.insert(tk.END, msg + "\n")
                txt_log.see(tk.END)
            log_win.after(0, _log)
            
        def cancel_download():
            self.dss_download_cancelled = True
            log_msg("[INFO] Cancellation requested by user...")
            
        btn_cancel = tk.Button(log_win, text="Cancel Download", command=cancel_download, bg="#ef4444", fg="white", font=("Segoe UI", 9, "bold"), bd=0, padx=15, pady=6)
        btn_cancel.pack(side="bottom", pady=15)
        
        txt_log.pack(side="top", fill="both", expand=True, padx=15, pady=(0, 10))
            
        def set_progress(value, max_val):
            def _prog():
                progress_bar.config(maximum=max_val, value=value)
            log_win.after(0, _prog)
            
        self.loaded_dss_tiles = []
        
        def add_tile_to_memory(file_path, r_val=None, d_val=None, should_render=True):
            try:
                import os
                if r_val is None or d_val is None:
                    parts = os.path.basename(file_path).replace(".fits", "").split("_")
                    if len(parts) == 4:
                        r_val = float(parts[1])
                        d_val = float(parts[2])
                with fits.open(file_path) as hdul:
                    dss_data = hdul[0].data.astype(np.float32)
                    dss_header = hdul[0].header
                    wcs_obj = WCS(dss_header, naxis=2)
                
                # Background median matching to eliminate seams (stacco) between tiles
                dss_data = np.nan_to_num(dss_data, nan=np.nanmedian(dss_data))
                bg_med = np.nanmedian(dss_data)
                dss_data = dss_data - bg_med
                
                vmax = np.nanpercentile(dss_data, 99.5)
                if np.isnan(vmax) or vmax <= 0:
                    vmax = 1.0
                    
                # Lock background (0) to gray level 15, and stars to 235
                dss_raw = np.clip(15.0 + (dss_data / vmax) * 220.0, 0, 255).astype(np.uint8)
                
                self.loaded_dss_tiles.append({
                    'data': dss_raw,
                    'wcs': wcs_obj,
                    'path': file_path,
                    'ra': r_val,
                    'dec': d_val
                })
                
                if should_render:
                    self.dss_blend_ratio.set(0.50)
                    self.dss_warped_cache = None
                    self.render_canvas(is_dragging=False)
                    self.root.config(cursor="")
            except Exception as e:
                log_msg(f"[ERROR] Failed to load tile {file_path}: {e}")

        import os
        import numpy as np
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        
        h_orig, w_orig = self.debayered_cache.shape[:2]
        center_coord = self.wcs.pixel_to_world(w_orig / 2.0, h_orig / 2.0)
        target_ra = center_coord.ra.deg
        target_dec = center_coord.dec.deg
        
        if os.path.exists("panstarrs_cache"):
            log_msg("[INFO] Preloading overlapping files from cache...")
            for filename in os.listdir("panstarrs_cache"):
                if filename.startswith("panstarrs_") and filename.endswith(".fits"):
                    parts = filename.replace(".fits", "").split("_")
                    if len(parts) == 4:
                        try:
                            file_ra = float(parts[1])
                            file_dec = float(parts[2])
                            file_size = float(parts[3])
                            
                            d_dec = target_dec - file_dec
                            d_ra = ((target_ra - file_ra) * np.cos(np.radians((target_dec + file_dec) / 2.0))) % 360.0
                            if d_ra > 180.0:
                                d_ra = 360.0 - d_ra
                            dist_deg = np.sqrt(d_ra**2 + d_dec**2)
                            
                            max_dist_deg = (h_orig + w_orig)/2.0 * 0.70
                            if dist_deg < max_dist_deg:
                                log_msg(f"[CACHE] Auto-loading cached tile: {filename}")
                                add_tile_to_memory(os.path.join("panstarrs_cache", filename), file_ra, file_dec, should_render=False)
                        except Exception:
                            pass
            if self.loaded_dss_tiles:
                self.dss_blend_ratio.set(0.50)
                self.dss_warped_cache = None
                self.render_canvas(is_dragging=False)
                self.root.config(cursor="")

        def finish_all_done():
            self.root.config(cursor="")
            log_msg("[SUCCESS] All sky tiles retrieved and merged!")
            messagebox.showinfo("PanSTARRS Sky", "Successfully loaded PanSTARRS reference sky.\nAdjust 'DSS Sky Blend' slider below to crossfade.", parent=self.root)

        def finish_error(err):
            self.root.config(cursor="")
            log_msg(f"[ERROR] Failed to download PanSTARRS: {err}")
            messagebox.showerror("Error", f"Failed to retrieve PanSTARRS image: {err}", parent=self.root)

        def worker():
            try:
                if not os.path.exists("panstarrs_cache"):
                    os.makedirs("panstarrs_cache")
                    
                tile_size_arcmin = 15.0
                size_pix = 3600
                step_deg = 0.20
                
                c_center = self.wcs.pixel_to_world(w_orig / 2.0, h_orig / 2.0)
                c_right = self.wcs.pixel_to_world(w_orig, h_orig / 2.0)
                c_top = self.wcs.pixel_to_world(w_orig / 2.0, h_orig)
                width_deg = c_center.separation(c_right).deg * 2.0
                height_deg = c_center.separation(c_top).deg * 2.0
                
                num_steps_x = int(np.ceil(width_deg / step_deg))
                num_steps_y = int(np.ceil(height_deg / step_deg))
                num_steps_x = np.clip(num_steps_x, 1, 5)
                num_steps_y = np.clip(num_steps_y, 1, 5)
                
                half_x = (num_steps_x - 1) / 2.0
                dx_vals = [(i - half_x) * step_deg for i in range(num_steps_x)]
                
                half_y = (num_steps_y - 1) / 2.0
                dy_vals = [(i - half_y) * step_deg for i in range(num_steps_y)]
                
                cos_dec = np.cos(np.radians(target_dec))
                cos_dec_val = max(1e-6, cos_dec)
                
                candidate_coords = []
                for dx in dx_vals:
                    for dy in dy_vals:
                        cand_ra = (target_ra + dx / cos_dec_val) % 360.0
                        cand_dec = target_dec + dy
                        if -90.0 <= cand_dec <= 90.0:
                            candidate_coords.append((cand_ra, cand_dec))
                            
                grid_points = []
                for r, d in candidate_coords:
                    if abs(r - target_ra) < 0.001 and abs(d - target_dec) < 0.001:
                        grid_points.insert(0, (r, d))
                        continue
                    try:
                        px, py = self.wcs.world_to_pixel(SkyCoord(ra=r*u.deg, dec=d*u.deg, frame='icrs'))
                        if -100 <= px < w_orig + 100 and -100 <= py < h_orig + 100:
                            grid_points.append((r, d))
                    except Exception:
                        pass
                        
                remaining_points = []
                for r, d in grid_points:
                    covered = False
                    for tile in self.loaded_dss_tiles:
                        tile_ra = tile.get('ra')
                        tile_dec = tile.get('dec')
                        if tile_ra is not None and tile_dec is not None:
                            d_dec = d - tile_dec
                            d_ra = ((r - tile_ra) * np.cos(np.radians((d + tile_dec) / 2.0))) % 360.0
                            if d_ra > 180.0:
                                d_ra = 360.0 - d_ra
                            dist_arcmin = np.sqrt(d_ra**2 + d_dec**2) * 60.0
                            if dist_arcmin < 10.0:
                                covered = True
                                break
                    if not covered:
                        remaining_points.append((r, d))
                        
                log_msg(f"[INFO] Generated footprint grid: {len(remaining_points)} PanSTARRS tiles to load.")
                
                import urllib.request
                import urllib.parse
                import ssl
                
                ssl_context = ssl._create_unverified_context()
                
                for idx, (tile_ra, tile_dec) in enumerate(remaining_points):
                    if self.dss_download_cancelled:
                        raise Exception("Download cancelled by user.")
                        
                    log_msg(f"\n--- Loading tile {idx+1}/{len(remaining_points)} at RA={tile_ra:.4f}, DEC={tile_dec:.4f} ---")
                    
                    # 1. Resolve filename first using ps1filenames.py
                    filenames_url = f"https://ps1images.stsci.edu/cgi-bin/ps1filenames.py?ra={tile_ra}&dec={tile_dec}&filters=r"
                    log_msg(f"[QUERY] Resolving skycell for RA={tile_ra:.4f}, DEC={tile_dec:.4f}...")
                    
                    req1 = urllib.request.Request(filenames_url, headers={'User-Agent': 'Mozilla/5.0'})
                    filename_path = None
                    try:
                        with urllib.request.urlopen(req1, timeout=20, context=ssl_context) as response1:
                            content = response1.read().decode('utf-8')
                            lines = content.strip().split('\n')
                            if len(lines) > 1:
                                header = lines[0].split()
                                if 'filename' in header:
                                    fn_idx = header.index('filename')
                                    row1 = lines[1].split()
                                    if len(row1) > fn_idx:
                                        filename_path = row1[fn_idx]
                    except Exception:
                        pass
                                        
                    if not filename_path:
                        filenames_url = f"https://ps1images.stsci.edu/cgi-bin/ps1filenames.py?ra={tile_ra}&dec={tile_dec}&filters=g"
                        req1 = urllib.request.Request(filenames_url, headers={'User-Agent': 'Mozilla/5.0'})
                        try:
                            with urllib.request.urlopen(req1, timeout=20, context=ssl_context) as response1:
                                content = response1.read().decode('utf-8')
                                lines = content.strip().split('\n')
                                if len(lines) > 1:
                                    header = lines[0].split()
                                    if 'filename' in header:
                                        fn_idx = header.index('filename')
                                        row1 = lines[1].split()
                                        if len(row1) > fn_idx:
                                            filename_path = row1[fn_idx]
                        except Exception:
                            pass
                                            
                    if not filename_path:
                        log_msg(f"[WARN] No PanSTARRS coverage found for RA={tile_ra:.4f}, DEC={tile_dec:.4f}. Skipping tile.")
                        continue
                        
                    # 2. Fetch a tiny 1-pixel FITS cutout to resolve the WCS and get the exact skycell center
                    log_msg("[QUERY] Fetching skycell WCS header (1-pixel probe)...")
                    probe_url = f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?ra={tile_ra}&dec={tile_dec}&size=1&format=fits&red={filename_path}"
                    req_probe = urllib.request.Request(probe_url, headers={'User-Agent': 'Mozilla/5.0'})
                    
                    skycell_ra, skycell_dec = tile_ra, tile_dec
                    try:
                        import io
                        from astropy.wcs import WCS
                        with urllib.request.urlopen(req_probe, timeout=15, context=ssl_context) as resp_probe:
                            probe_data = resp_probe.read()
                        with fits.open(io.BytesIO(probe_data)) as hdul:
                            wcs_obj = WCS(hdul[0].header)
                            # Convert pixel (3000, 3000) to find the exact center of this skycell
                            sky_center = wcs_obj.pixel_to_world(3000, 3000)
                            skycell_ra = sky_center.ra.deg
                            skycell_dec = sky_center.dec.deg
                        log_msg(f"[SUCCESS] Resolved true skycell center: RA={skycell_ra:.6f}, DEC={skycell_dec:.6f}")
                    except Exception as probe_err:
                        log_msg(f"[WARN] Failed to resolve skycell center: {probe_err}. Using requested coordinates.")
                        
                    # 3. Check if this exact skycell center is already in cache
                    cached_path = self.get_cached_panstarrs_tile(skycell_ra, skycell_dec, tile_size_arcmin)
                    if cached_path:
                        log_msg(f"[CACHE] Found local cache for skycell center: {os.path.basename(cached_path)}")
                        # Check if already loaded in memory to prevent duplicates
                        if not any(abs(t.get('ra', 0.0) - skycell_ra) < 0.001 and abs(t.get('dec', 0.0) - skycell_dec) < 0.001 for t in self.loaded_dss_tiles):
                            log_win.after(0, lambda p=cached_path, r=skycell_ra, d=skycell_dec: add_tile_to_memory(p, r, d))
                        continue
                        
                    # 4. Download full-size cutout centered EXACTLY on the skycell center to prevent black borders
                    cutout_url = f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?ra={skycell_ra}&dec={skycell_dec}&size={size_pix}&format=fits&red={filename_path}"
                    log_msg(f"[QUERY] Fetching PanSTARRS FITS centered on skycell...")
                    
                    req2 = urllib.request.Request(cutout_url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req2, timeout=30, context=ssl_context) as response2:
                        content_length = response2.headers.get('Content-Length')
                        total_bytes = int(content_length) if content_length else 0
                        if total_bytes:
                            log_msg(f"[INFO] File size: {total_bytes / 1024.0 / 1024.0:.2f} MB")
                            set_progress(0, total_bytes)
                        else:
                            set_progress(0, 100)
                            
                        img_data = bytearray()
                        downloaded_bytes = 0
                        chunk_size = 65536
                        last_log_time = 0
                        
                        while True:
                            if self.dss_download_cancelled:
                                raise Exception("Download cancelled by user.")
                            chunk = response2.read(chunk_size)
                            if not chunk:
                                break
                            img_data.extend(chunk)
                            downloaded_bytes += len(chunk)
                            
                            if total_bytes:
                                set_progress(downloaded_bytes, total_bytes)
                                import time
                                current_time = time.time()
                                if current_time - last_log_time > 1.5 or downloaded_bytes == total_bytes:
                                    pct = (downloaded_bytes / total_bytes) * 100
                                    log_msg(f"[INFO] Downloading: {pct:.1f}%")
                                    last_log_time = current_time
                            else:
                                set_progress(downloaded_bytes % 100, 100)
                                
                    if b"html" in img_data[:100].lower() or b"<!doctype" in img_data[:100].lower():
                        raise Exception("STScI fitscut service returned an HTML error page.")
                        
                    cache_filename = f"panstarrs_{skycell_ra:.6f}_{skycell_dec:.6f}_{tile_size_arcmin:.1f}.fits"
                    save_path = os.path.join("panstarrs_cache", cache_filename)
                    with open(save_path, "wb") as f:
                        f.write(img_data)
                    log_msg(f"[CACHE] Saved to cache: {cache_filename}")
                    
                    log_win.after(0, lambda p=save_path, r=skycell_ra, d=skycell_dec: add_tile_to_memory(p, r, d))
                    
                log_win.after(0, finish_all_done)
                
            except Exception as e:
                log_win.after(0, lambda: finish_error(e))
                
        import threading
        download_thread = threading.Thread(target=worker)
        download_thread.daemon = True
        download_thread.start()

    def download_dss_background(self):
        if self.fits_data is None or self.wcs is None:
            messagebox.showerror("Error", "A FITS file with valid WCS headers must be loaded first.")
            return
            
        self.root.config(cursor="watch")
        self.root.update()
        
        # Create a progress log window overlay
        log_win = tk.Toplevel(self.root)
        log_win.title("DSS Cutout Service Monitor")
        log_win.geometry("600x450")
        log_win.configure(bg="#1f2937")
        log_win.transient(self.root)
        
        lbl = tk.Label(log_win, text="Retrieving DSS2 Reference Survey Image...", bg="#1f2937", fg="white", font=("Segoe UI", 10, "bold"))
        lbl.pack(pady=(10, 5))
        
        # Add Progress Bar
        from tkinter import ttk
        style = ttk.Style(log_win)
        style.theme_use('clam')
        style.configure("DSS.Horizontal.TProgressbar", troughcolor="#111827", background="#38bdf8", bordercolor="#1f2937", lightcolor="#38bdf8", darkcolor="#38bdf8")
        
        progress_bar = ttk.Progressbar(log_win, style="DSS.Horizontal.TProgressbar", orient="horizontal", mode="determinate", length=560)
        progress_bar.pack(pady=5, padx=15, fill="x")
        
        txt_log = tk.Text(log_win, bg="#111827", fg="#38bdf8", insertbackground="white", font=("Consolas", 9), bd=0, padx=10, pady=10)
        
        self.dss_download_cancelled = False
        
        def log_msg(msg):
            def _log():
                txt_log.insert(tk.END, msg + "\n")
                txt_log.see(tk.END)
            log_win.after(0, _log)
            
        def cancel_download():
            self.dss_download_cancelled = True
            log_msg("[INFO] Cancellation requested by user...")
            
        btn_cancel = tk.Button(log_win, text="Cancel Download", command=cancel_download, bg="#ef4444", fg="white", font=("Segoe UI", 9, "bold"), bd=0, padx=15, pady=6)
        btn_cancel.pack(side="bottom", pady=15)
        
        txt_log.pack(side="top", fill="both", expand=True, padx=15, pady=(0, 10))
            
        def set_progress(value, max_val):
            def _prog():
                progress_bar.config(maximum=max_val, value=value)
            log_win.after(0, _prog)
            
        # Reset current loaded tiles list
        self.loaded_dss_tiles = []
        
        def add_tile_to_memory(file_path, r_val=None, d_val=None, should_render=True):
            try:
                import os
                # Parse RA/Dec from filename if not explicitly provided
                if r_val is None or d_val is None:
                    parts = os.path.basename(file_path).replace(".fits", "").split("_")
                    if len(parts) == 4:
                        r_val = float(parts[1])
                        d_val = float(parts[2])
                with fits.open(file_path) as hdul:
                    dss_data = hdul[0].data.astype(np.float32)
                    dss_header = hdul[0].header
                    wcs_obj = WCS(dss_header, naxis=2)
                
                # Background median matching to eliminate seams (stacco) between tiles
                dss_data = np.nan_to_num(dss_data, nan=np.nanmedian(dss_data))
                bg_med = np.nanmedian(dss_data)
                dss_data = dss_data - bg_med
                
                vmax = np.nanpercentile(dss_data, 99.5)
                if np.isnan(vmax) or vmax <= 0:
                    vmax = 1.0
                    
                # Lock background (0) to gray level 15, and stars to 235
                dss_raw = np.clip(15.0 + (dss_data / vmax) * 220.0, 0, 255).astype(np.uint8)
                
                # Save parsed tile in list
                self.loaded_dss_tiles.append({
                    'data': dss_raw,
                    'wcs': wcs_obj,
                    'path': file_path,
                    'ra': r_val,
                    'dec': d_val
                })
                
                if should_render:
                    # Set slider and redraw
                    self.dss_blend_ratio.set(0.50)
                    # Reset view cache to force redraw of DSS layer
                    self.dss_warped_cache = None
                    self.render_canvas(is_dragging=False)
                    # Release hourglass/watch cursor as soon as we have at least one tile loaded so the user can interact
                    self.root.config(cursor="")
            except Exception as e:
                log_msg(f"[ERROR] Failed to load tile {file_path}: {e}")

        # Scan and preload all overlapping cached tiles instantly on click using fast angular distance math
        import os
        import numpy as np
        
        h_orig, w_orig = self.debayered_cache.shape[:2]
        center_coord = self.wcs.pixel_to_world(w_orig / 2.0, h_orig / 2.0)
        target_ra = center_coord.ra.deg
        target_dec = center_coord.dec.deg
        
        corner_coord = self.wcs.pixel_to_world(0, 0)
        radius_deg = center_coord.separation(corner_coord).deg
        
        if os.path.exists("dss_cache"):
            log_msg("[INFO] Preloading overlapping files from cache...")
            for filename in os.listdir("dss_cache"):
                if filename.startswith("dss_") and filename.endswith(".fits"):
                    parts = filename.replace(".fits", "").split("_")
                    if len(parts) == 4:
                        try:
                            file_ra = float(parts[1])
                            file_dec = float(parts[2])
                            file_size = float(parts[3])
                            
                            d_dec = target_dec - file_dec
                            d_ra = ((target_ra - file_ra) * np.cos(np.radians((target_dec + file_dec) / 2.0))) % 360.0
                            if d_ra > 180.0:
                                d_ra = 360.0 - d_ra
                            dist_deg = np.sqrt(d_ra**2 + d_dec**2)
                            
                            # Check overlap footprint
                            max_dist_deg = radius_deg + (file_size / 60.0) * 0.70
                            if dist_deg < max_dist_deg:
                                log_msg(f"[CACHE] Auto-loading cached tile: {filename}")
                                add_tile_to_memory(os.path.join("dss_cache", filename), file_ra, file_dec, should_render=False)
                        except Exception:
                            pass
            
            # Perform a single high-quality render after loading all cached tiles to avoid stuttering
            if self.loaded_dss_tiles:
                self.dss_blend_ratio.set(0.50)
                self.dss_warped_cache = None
                self.render_canvas(is_dragging=False)
                self.root.config(cursor="")

        def finish_all_done():
            self.root.config(cursor="")
            log_msg("[SUCCESS] All sky tiles retrieved and merged!")
            messagebox.showinfo("DSS Sky", "Successfully loaded DSS reference sky.\nAdjust 'DSS Sky Blend' slider below to crossfade.", parent=self.root)

        def finish_error(err):
            self.root.config(cursor="")
            log_msg(f"[ERROR] Failed to download DSS: {err}")
            messagebox.showerror("Error", f"Failed to retrieve DSS image: {err}", parent=self.root)

        def worker():
            try:
                h_orig, w_orig = self.debayered_cache.shape[:2]
                
                # 1. Determine primary query coordinate (Annotation RA/Dec or center)
                target_ra, target_dec = None, None
                if getattr(self, 'annotations', None) and self.wcs:
                    ann = self.annotations[0]
                    px = ann['x'] * w_orig
                    py = ann['y'] * h_orig
                    sky = self.wcs.pixel_to_world(px, py)
                    target_ra = sky.ra.deg
                    target_dec = sky.dec.deg
                    log_msg(f"[INFO] Targeting annotation coordinate: RA={target_ra:.6f} deg, DEC={target_dec:.6f} deg")
                else:
                    center_coord = self.wcs.pixel_to_world(w_orig / 2.0, h_orig / 2.0)
                    target_ra = center_coord.ra.deg
                    target_dec = center_coord.dec.deg
                    log_msg(f"[INFO] Targeting center coordinate: RA={target_ra:.6f} deg, DEC={target_dec:.6f} deg")
                
                corner_coord = self.wcs.pixel_to_world(0, 0)
                radius_deg = self.wcs.pixel_to_world(w_orig / 2.0, h_orig / 2.0).separation(corner_coord).deg
                radius_arcmin = radius_deg * 60.0
                
                # We request a tile size of 45 arcminutes (fast query size)
                tile_size_arcmin = 45.0
                
                # 2. Build grid coordinates to cover the FITS footprint dynamically
                # Step size is 30 arcminutes to guarantee substantial overlap
                step_deg = 0.5
                
                # Estimate the width and height of the image in degrees
                c_center = self.wcs.pixel_to_world(w_orig / 2.0, h_orig / 2.0)
                c_right = self.wcs.pixel_to_world(w_orig, h_orig / 2.0)
                c_top = self.wcs.pixel_to_world(w_orig / 2.0, h_orig)
                
                width_deg = c_center.separation(c_right).deg * 2.0
                height_deg = c_center.separation(c_top).deg * 2.0
                
                num_steps_x = int(np.ceil(width_deg / step_deg))
                num_steps_y = int(np.ceil(height_deg / step_deg))
                
                # Limit the steps between 1 and 5 (covers up to a 5x5 grid surface)
                num_steps_x = np.clip(num_steps_x, 1, 5)
                num_steps_y = np.clip(num_steps_y, 1, 5)
                
                half_x = (num_steps_x - 1) / 2.0
                dx_vals = [(i - half_x) * step_deg for i in range(num_steps_x)]
                
                half_y = (num_steps_y - 1) / 2.0
                dy_vals = [(i - half_y) * step_deg for i in range(num_steps_y)]
                
                log_msg(f"[INFO] Calculated dynamic coverage grid: {num_steps_x}x{num_steps_y} surface")
                
                # Cosine factor for RA offset
                cos_dec = np.cos(np.radians(target_dec))
                cos_dec_val = max(1e-6, cos_dec)
                
                candidate_coords = []
                for dx in dx_vals:
                    for dy in dy_vals:
                        cand_ra = (target_ra + dx / cos_dec_val) % 360.0
                        cand_dec = target_dec + dy
                        if -90.0 <= cand_dec <= 90.0:
                            candidate_coords.append((cand_ra, cand_dec))
                
                # Filter grid coordinates that fall inside FITS viewport
                grid_points = []
                for r, d in candidate_coords:
                    # Target center (0.0, 0.0 offset) is always queried first
                    if abs(r - target_ra) < 0.001 and abs(d - target_dec) < 0.001:
                        grid_points.insert(0, (r, d))
                        continue
                        
                    try:
                        px, py = self.wcs.world_to_pixel(SkyCoord(ra=r*u.deg, dec=d*u.deg, frame='icrs'))
                        # Keep coordinates that map close to FITS footprint
                        if -200 <= px < w_orig + 200 and -200 <= py < h_orig + 200:
                            grid_points.append((r, d))
                    except Exception:
                        pass
                
                # Exclude coordinates that are already covered by preloaded/cached tiles
                remaining_points = []
                for r, d in grid_points:
                    if not self.is_coord_covered(r, d):
                        remaining_points.append((r, d))
                    else:
                        log_msg(f"[CACHE] Sky region at RA={r:.4f}, DEC={d:.4f} is already covered by a loaded tile. Skipping query.")
                
                log_msg(f"[INFO] Generated footprint grid: {len(remaining_points)} tiles to load.")
                
                import urllib.request
                import urllib.parse
                import os
                
                # Download / Load tiles sequentially
                for idx, (tile_ra, tile_dec) in enumerate(remaining_points):
                    if self.dss_download_cancelled:
                        raise Exception("Download cancelled by user.")
                        
                    log_msg(f"\n--- Loading tile {idx+1}/{len(remaining_points)} at RA={tile_ra:.4f}, DEC={tile_dec:.4f} ---")
                    
                    # Check cache first
                    cached_path = self.get_cached_tile(tile_ra, tile_dec, tile_size_arcmin)
                    if cached_path:
                        log_msg(f"[CACHE] Found local cache: {os.path.basename(cached_path)}")
                        log_win.after(0, lambda p=cached_path: add_tile_to_memory(p))
                        continue
                        
                    # Request from server
                    coord = SkyCoord(ra=tile_ra*u.deg, dec=tile_dec*u.deg, frame='icrs')
                    ra_str = coord.ra.to_string(unit="hour", sep=" ", precision=2)
                    dec_str = coord.dec.to_string(unit="degree", sep=" ", precision=2)
                    
                    params = {
                        'v': 'poss2ukstu_red',
                        'r': ra_str,
                        'd': dec_str,
                        'e': 'J2000',
                        'h': f"{tile_size_arcmin:.2f}",
                        'w': f"{tile_size_arcmin:.2f}",
                        'f': 'fits'
                    }
                    query_str = urllib.parse.urlencode(params)
                    url = f"http://archive.stsci.edu/cgi-bin/dss_search?{query_str}"
                    log_msg(f"[QUERY] Fetching FITS from: {url}")
                    
                    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, timeout=30) as response:
                        content_length = response.headers.get('Content-Length')
                        total_bytes = int(content_length) if content_length else 0
                        if total_bytes:
                            log_msg(f"[INFO] File size: {total_bytes / 1024.0 / 1024.0:.2f} MB")
                            set_progress(0, total_bytes)
                        else:
                            set_progress(0, 100)
                            
                        img_data = bytearray()
                        downloaded_bytes = 0
                        chunk_size = 65536
                        last_log_time = 0
                        
                        while True:
                            if self.dss_download_cancelled:
                                raise Exception("Download cancelled by user.")
                            chunk = response.read(chunk_size)
                            if not chunk:
                                break
                            img_data.extend(chunk)
                            downloaded_bytes += len(chunk)
                            
                            if total_bytes:
                                set_progress(downloaded_bytes, total_bytes)
                                import time
                                current_time = time.time()
                                if current_time - last_log_time > 1.5 or downloaded_bytes == total_bytes:
                                    pct = (downloaded_bytes / total_bytes) * 100
                                    log_msg(f"[INFO] Downloading: {pct:.1f}%")
                                    last_log_time = current_time
                            else:
                                set_progress(downloaded_bytes % 100, 100)
                                
                    if b"html" in img_data[:100].lower() or b"<!doctype" in img_data[:100].lower():
                        raise Exception("STScI service returned an HTML error page.")
                        
                    # Save to cache folder
                    cache_filename = f"dss_{tile_ra:.6f}_{tile_dec:.6f}_{tile_size_arcmin:.1f}.fits"
                    save_path = os.path.join("dss_cache", cache_filename)
                    with open(save_path, "wb") as f:
                        f.write(img_data)
                    log_msg(f"[CACHE] Saved to cache: {cache_filename}")
                    
                    log_win.after(0, lambda p=save_path: add_tile_to_memory(p))
                    
                log_win.after(0, finish_all_done)
                
            except Exception as e:
                log_win.after(0, lambda: finish_error(e))

        import threading
        download_thread = threading.Thread(target=worker)
        download_thread.daemon = True
        download_thread.start()

    def platesolve_vizier_astroalign(self):
        if self.fits_data is None:
            messagebox.showerror("Error", "Load a FITS file first.")
            return

        # Prefill RA/Dec from header pointing keywords FIRST (most reliable target coordinate)
        ra_val = ""
        dec_val = ""
        for key in ['RA', 'OBJCTRA', 'TELRA']:
            if key in self.fits_header and self.fits_header[key] != "":
                ra_val = self.fits_header[key]
                break
        for key in ['DEC', 'OBJCTDEC', 'TELDEC']:
            if key in self.fits_header and self.fits_header[key] != "":
                dec_val = self.fits_header[key]
                break
                
        if ra_val != "":
            init_ra = str(ra_val)
        if dec_val != "":
            init_dec = str(dec_val)
            
        # If no telescope header coordinates exist, fallback to WCS center
        if not init_ra or not init_dec:
            if self.debayered_cache is not None and self.wcs is not None:
                try:
                    h_orig, w_orig = self.debayered_cache.shape[:2]
                    cx_px = w_orig / 2.0
                    cy_px = h_orig / 2.0
                    
                    if hasattr(self, 'src_left_int') and hasattr(self, 'src_top_int'):
                        w_view = getattr(self, 'src_right_int', w_orig) - self.src_left_int
                        h_view = getattr(self, 'src_bottom_int', h_orig) - self.src_top_int
                        cx_px = self.src_left_int + w_view / 2.0
                        cy_px = self.src_top_int + h_view / 2.0
                        
                    if getattr(self, 'visual_crop_box', None) is not None:
                        start_x, start_y, end_x, end_y = self.visual_crop_box
                        cx_px += start_x
                        cy_px += start_y
                        
                    center_sky = self.wcs.pixel_to_world(cx_px, cy_px)
                    init_ra = f"{center_sky.ra.deg:.6f}"
                    init_dec = f"{center_sky.dec.deg:.6f}"
                except Exception:
                    pass
            
        init_focal = str(self.fits_header.get('FOCALLEN', ''))
        init_pixel = str(self.fits_header.get('XPIXSZ', self.fits_header.get('PIXSIZE', '')))
        
        # Calculate optimal search radius based on image dimensions and pixel scale
        init_radius = "50"
        try:
            focal_val = float(init_focal)
            pixel_val = float(init_pixel)
            if focal_val > 0 and pixel_val > 0:
                h_orig, w_orig = self.fits_data.shape[-2:]
                pixel_scale_deg = (pixel_val / focal_val) * (206.265 / 3600.0)
                diag_pixels = np.sqrt(w_orig**2 + h_orig**2)
                diag_arcmin = diag_pixels * pixel_scale_deg * 60.0
                # Optimal radius is half the diagonal plus a margin of 20%
                opt_rad = (diag_arcmin / 2.0) * 1.2
                init_radius = f"{opt_rad:.1f}"
        except Exception:
            pass

        dialog = tk.Toplevel(self.root)
        dialog.title("Plate Solve (Vizier/Astroalign)")
        dialog.geometry("450x380")
        dialog.configure(bg=self.panel_color)
        dialog.transient(self.root)
        dialog.grab_set()
        
        lbl_title = tk.Label(dialog, text="Plate Solve Parameters", bg=self.panel_color, fg=self.text_color, font=("Segoe UI", 12, "bold"))
        lbl_title.pack(pady=15)
        
        grid_frame = tk.Frame(dialog, bg=self.panel_color)
        grid_frame.pack(padx=20, pady=10, fill="both", expand=True)
        
        lbl_ra = tk.Label(grid_frame, text="Target RA (deg or hms):", bg=self.panel_color, fg=self.text_color, font=("Segoe UI", 9, "bold"), anchor="w")
        lbl_ra.grid(row=0, column=0, sticky="ew", pady=5)
        ent_ra = tk.Entry(grid_frame, bg=self.bg_color, fg=self.text_color, insertbackground="white", bd=0, highlightthickness=1, highlightbackground=self.control_bg)
        ent_ra.insert(0, init_ra)
        ent_ra.grid(row=0, column=1, sticky="ew", pady=5, padx=10)
        
        lbl_dec = tk.Label(grid_frame, text="Target DEC (deg or dms):", bg=self.panel_color, fg=self.text_color, font=("Segoe UI", 9, "bold"), anchor="w")
        lbl_dec.grid(row=1, column=0, sticky="ew", pady=5)
        ent_dec = tk.Entry(grid_frame, bg=self.bg_color, fg=self.text_color, insertbackground="white", bd=0, highlightthickness=1, highlightbackground=self.control_bg)
        ent_dec.insert(0, init_dec)
        ent_dec.grid(row=1, column=1, sticky="ew", pady=5, padx=10)
        
        lbl_focal = tk.Label(grid_frame, text="Focal Length (mm):", bg=self.panel_color, fg=self.text_color, font=("Segoe UI", 9, "bold"), anchor="w")
        lbl_focal.grid(row=2, column=0, sticky="ew", pady=5)
        ent_focal = tk.Entry(grid_frame, bg=self.bg_color, fg=self.text_color, insertbackground="white", bd=0, highlightthickness=1, highlightbackground=self.control_bg)
        ent_focal.insert(0, init_focal)
        ent_focal.grid(row=2, column=1, sticky="ew", pady=5, padx=10)
        
        lbl_pixel = tk.Label(grid_frame, text="Pixel Size (μm):", bg=self.panel_color, fg=self.text_color, font=("Segoe UI", 9, "bold"), anchor="w")
        lbl_pixel.grid(row=3, column=0, sticky="ew", pady=5)
        ent_pixel = tk.Entry(grid_frame, bg=self.bg_color, fg=self.text_color, insertbackground="white", bd=0, highlightthickness=1, highlightbackground=self.control_bg)
        ent_pixel.insert(0, init_pixel)
        ent_pixel.grid(row=3, column=1, sticky="ew", pady=5, padx=10)

        lbl_radius = tk.Label(grid_frame, text="Search Radius (arcmin):", bg=self.panel_color, fg=self.text_color, font=("Segoe UI", 9, "bold"), anchor="w")
        lbl_radius.grid(row=4, column=0, sticky="ew", pady=5)
        ent_radius = tk.Entry(grid_frame, bg=self.bg_color, fg=self.text_color, insertbackground="white", bd=0, highlightthickness=1, highlightbackground=self.control_bg)
        ent_radius.insert(0, init_radius)
        ent_radius.grid(row=4, column=1, sticky="ew", pady=5, padx=10)
        
        grid_frame.columnconfigure(1, weight=1)
        
        btn_frame = tk.Frame(dialog, bg=self.panel_color)
        btn_frame.pack(pady=15, fill="x")
        
        def parse_coordinate(ra_str, dec_str):
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            ra_str = str(ra_str).strip()
            dec_str = str(dec_str).strip()
            try:
                return SkyCoord(ra=float(ra_str), dec=float(dec_str), unit=(u.deg, u.deg), frame='icrs')
            except ValueError:
                pass
            try:
                if 'h' in ra_str or ' ' in ra_str:
                    return SkyCoord(ra=ra_str, dec=dec_str, unit=(u.hourangle, u.deg), frame='icrs')
                else:
                    return SkyCoord(ra=ra_str, dec=dec_str, unit=(u.deg, u.deg), frame='icrs')
            except Exception as e:
                raise ValueError(f"RA={ra_str}, DEC={dec_str}: {e}")

        def on_solve():
            ra_s = ent_ra.get().strip()
            dec_s = ent_dec.get().strip()
            focal_s = ent_focal.get().strip()
            pixel_s = ent_pixel.get().strip()
            radius_s = ent_radius.get().strip()
            
            if not ra_s or not dec_s:
                messagebox.showerror("Error", "Please provide target RA and DEC coordinates.")
                return
            try:
                parse_coordinate(ra_s, dec_s)
            except Exception as ex:
                messagebox.showerror("Error", f"Failed to parse coordinates: {ex}")
                return
                
            try:
                f_val = float(focal_s)
                p_val = float(pixel_s)
                r_val = float(radius_s)
                if f_val <= 0 or p_val <= 0 or r_val <= 0:
                    raise ValueError()
            except ValueError:
                messagebox.showerror("Error", "Focal Length, Pixel Size and Radius must be valid positive numbers.")
                return
                
            dialog.destroy()
            self.run_platesolve_vizier_astroalign(ra_s, dec_s, f_val, p_val, r_val)
            
        btn_solve = tk.Button(btn_frame, text="Solve Image", command=on_solve, bg="#10b981", fg="white", font=("Segoe UI", 10, "bold"), bd=0, padx=20, pady=6)
        btn_solve.pack(side="right", padx=20)
        
        btn_cancel = tk.Button(btn_frame, text="Cancel", command=dialog.destroy, bg="#374151", fg="white", font=("Segoe UI", 10, "bold"), bd=0, padx=20, pady=6)
        btn_cancel.pack(side="right", padx=10)

    def run_platesolve_vizier_astroalign(self, ra_str, dec_str, focal_length, pixel_size, radius_arcmin):
        self.root.config(cursor="watch")
        self.root.update()
        
        log_win = tk.Toplevel(self.root)
        log_win.title("Vizier/Astroalign Plate Solver")
        log_win.geometry("600x450")
        log_win.configure(bg="#1f2937")
        log_win.transient(self.root)
        
        lbl = tk.Label(log_win, text="Resolving Astrometric Solution...", bg="#1f2937", fg="white", font=("Segoe UI", 10, "bold"))
        lbl.pack(pady=(10, 5))
        
        txt_log = tk.Text(log_win, bg="#111827", fg="#10b981", insertbackground="white", font=("Consolas", 9), bd=0, padx=10, pady=10)
        txt_log.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        def log_msg(msg):
            def _log():
                txt_log.insert(tk.END, msg + "\n")
                txt_log.see(tk.END)
            log_win.after(0, _log)
            
        def finish_done(solved_wcs):
            try:
                self.root.config(cursor="")
                log_msg("\n[SUCCESS] Image plate-solved successfully!")
                
                # Cache WCS in jobs database
                wcs_cards = dict(solved_wcs.to_header())
                wcs_cache = {k: v for k, v in wcs_cards.items() if any(x in k for x in ['CRPIX', 'CRVAL', 'CD', 'CTYPE', 'CUNIT', 'PC'])}
                self.jobs[self.fits_path] = {
                    "status": "success",
                    "timestamp": time.time(),
                    "wcs": wcs_cache
                }
                self.save_jobs()
                
                # Merge WCS header cards into in-memory fits_header
                for key, val in wcs_cache.items():
                    self.fits_header[key] = val
                self.fits_header['RADECSYS'] = 'ICRS'
                self.fits_header['EQUINOX'] = 2000.0
                self.wcs_saved_to_disk = False
                
                # If non-FITS file, save to sidecar file: <filename>.info.json
                ext = os.path.splitext(self.fits_path)[1].lower()
                if ext not in ['.fits', '.fit']:
                    sidecar_path = self.fits_path + ".info.json"
                    sidecar_data = {
                        "header": dict(self.fits_header),
                        "annotations": self.annotations
                    }
                    import json
                    try:
                        with open(sidecar_path, "w", encoding="utf-8") as sf:
                            json.dump(sidecar_data, sf, indent=4)
                        log_msg(f"[INFO] Sidecar file saved: {os.path.basename(sidecar_path)}")
                    except Exception as sf_err:
                        log_msg(f"[WARN] Failed to write sidecar file: {sf_err}")
                
                self.wcs = solved_wcs
                self.meta_tree.delete(*self.meta_tree.get_children())
                if ext not in ['.fits', '.fit']:
                    self.meta_tree.insert("", "end", text="IMAGE_TYPE", values=("Non-FITS (Loaded via PIL)",))
                    sidecar_path = self.fits_path + ".info.json"
                    if os.path.exists(sidecar_path):
                        self.meta_tree.insert("", "end", text="SIDECAR_INFO", values=("Loaded from .info.json",))
                important_keys = ["OBJECT", "EXPTIME", "TELESCOP", "INSTRUME", "FILTER", "DATE-OBS", "BAYERPAT", "EQUINOX"]
                for key in important_keys:
                    if key in self.fits_header:
                        self.meta_tree.insert("", "end", text=key, values=(str(self.fits_header[key]),))
                for key, val in self.fits_header.items():
                    if key not in important_keys and key.strip() != "":
                        self.meta_tree.insert("", "end", text=key, values=(str(val),))
                        
                self.render_canvas(is_dragging=False)
                self.update_wcs_hint_visibility()
                log_win.destroy()
                messagebox.showinfo("Plate Solve", "Plate solve successful! WCS coordinates loaded in memory.", parent=self.root)
            except Exception as e:
                log_msg(f"\n[FATAL ERROR] finish_done failed: {e}")
                import traceback
                log_msg(traceback.format_exc())
            
        def finish_error(err):
            self.root.config(cursor="")
            log_msg(f"\n[ERROR] Plate solve failed: {err}")
            messagebox.showerror("Error", f"Plate solve failed: {err}", parent=self.root)

        def parse_coordinate(ra_str, dec_str):
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            ra_str = str(ra_str).strip()
            dec_str = str(dec_str).strip()
            try:
                return SkyCoord(ra=float(ra_str), dec=float(dec_str), unit=(u.deg, u.deg), frame='icrs')
            except ValueError:
                pass
            try:
                if 'h' in ra_str or ' ' in ra_str:
                    return SkyCoord(ra=ra_str, dec=dec_str, unit=(u.hourangle, u.deg), frame='icrs')
                else:
                    return SkyCoord(ra=ra_str, dec=dec_str, unit=(u.deg, u.deg), frame='icrs')
            except Exception as e:
                raise ValueError(f"RA={ra_str}, DEC={dec_str}: {e}")

        def worker():
            try:
                import numpy as np
                import cv2
                import urllib.request
                import urllib.parse
                import ssl
                import astroalign as aa
                from astropy.wcs import WCS
                from astropy.coordinates import SkyCoord
                import astropy.units as u
                
                log_msg("[INFO] Extracting stars from FITS image...")
                h_orig, w_orig = self.fits_data.shape[-2:]
                
                if len(self.fits_data.shape) == 3:
                    gray = np.mean(self.fits_data, axis=0).astype(np.float32)
                else:
                    gray = self.fits_data.astype(np.float32)
                    
                percentiles_to_try = [99.2, 99.8, 99.9, 99.95, 99.98]
                thresh_val = None
                contours = []
                
                for pct in percentiles_to_try:
                    thresh_val = np.percentile(gray, pct)
                    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
                    thresh = thresh.astype(np.uint8)
                    contours_candidate, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    valid_candidates = 0
                    for cnt in contours_candidate:
                        if cv2.contourArea(cnt) > 1.0:
                            valid_candidates += 1
                            
                    if 80 <= valid_candidates <= 900:
                        contours = contours_candidate
                        log_msg(f"[INFO] Selected adaptive threshold at {pct}% (Thresh={thresh_val:.1f}, {valid_candidates} candidates)")
                        break
                
                if len(contours) == 0:
                    pct = 99.95 if np.max(gray) > 5000 else 99.2
                    thresh_val = np.percentile(gray, pct)
                    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
                    thresh = thresh.astype(np.uint8)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    log_msg(f"[INFO] Using fallback threshold at {pct}% (Thresh={thresh_val:.1f})")
                    
                img_stars = []
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area <= 1.0:
                        continue
                    M = cv2.moments(cnt)
                    if M["m00"] > 0:
                        cX = M["m10"] / M["m00"]
                        cY = M["m01"] / M["m00"]
                        img_stars.append((cX, cY, M["m00"]))
                        
                img_stars = sorted(img_stars, key=lambda s: s[2], reverse=True)[:150]
                img_pts = np.array([[s[0], s[1]] for s in img_stars], dtype=np.float32)
                log_msg(f"[INFO] Extracted {len(img_pts)} star candidates from FITS (hot-pixel filter applied).")
                
                if len(img_pts) < 10:
                    raise Exception("Too few stars detected in FITS image.")
                    
                target_coord = parse_coordinate(ra_str, dec_str)
                ra_deg = target_coord.ra.deg
                dec_deg = target_coord.dec.deg
                
                log_msg(f"[INFO] Coordinates: RA={ra_deg:.6f} deg, DEC={dec_deg:.6f} deg")
                pixel_scale_deg = (pixel_size / focal_length) * (206.265 / 3600.0)
                log_msg(f"[INFO] Calculated scale: {pixel_scale_deg*3600.0:.3f} arcsec/pixel")
                
                log_msg(f"[QUERY] Querying Vizier Gaia DR3 (Radius: {radius_arcmin} arcmin)...")
                c_val = f"{ra_deg:.6f} {dec_deg:.6f}".replace(',', '.')
                c_str = urllib.parse.quote(c_val)
                
                url = f"https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=I/355/gaiadr3&-c={c_str}&-c.r={radius_arcmin:f}&-c.u=arcmin&-out.form=|&-out.add=RA_ICRS,DE_ICRS&-out=Gmag&-sort=Gmag&-out.max=500"
                
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                ssl_context = ssl._create_unverified_context()
                
                lines = []
                try:
                    with urllib.request.urlopen(req, timeout=20, context=ssl_context) as response:
                        lines = response.read().decode('utf-8').split('\n')
                except Exception as ex:
                    log_msg(f"[WARN] Primary mirror failed: {ex}. Trying Harvard...")
                    url_harv = url.replace("vizier.cds.unistra.fr", "vizier.cfa.harvard.edu")
                    req = urllib.request.Request(url_harv, headers={'User-Agent': 'Mozilla/5.0'})
                    try:
                        with urllib.request.urlopen(req, timeout=20, context=ssl_context) as response:
                            lines = response.read().decode('utf-8').split('\n')
                    except Exception as harv_ex:
                        log_msg(f"[ERROR] Harvard mirror also failed: {harv_ex}")
                        raise Exception(f"All Vizier mirrors failed (Primary: {ex}, Secondary: {harv_ex}). Check your connection.")
                
                header_found = False
                cols = {}
                catalog_stars = []
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split('|')
                    if not header_found:
                        if any("RA" in p or "coord" in p or "_RAJ" in p for p in parts):
                            header_found = True
                            cols = {p.strip(): i for i, p in enumerate(parts)}
                        continue
                    if line.startswith("-"):
                        continue
                    if len(parts) >= 3:
                        try:
                            ra_idx = cols.get("RA_ICRS", cols.get("_RA_ICRS", 0))
                            dec_idx = cols.get("DE_ICRS", cols.get("_DE_ICRS", 1))
                            mag_idx = cols.get("Gmag", 2)
                            
                            ra_s = float(parts[ra_idx])
                            dec_s = float(parts[dec_idx])
                            mag_s = float(parts[mag_idx])
                            catalog_stars.append({'ra': ra_s, 'dec': dec_s, 'mag': mag_s})
                        except Exception:
                            continue
                            
                catalog_stars = sorted(catalog_stars, key=lambda s: s['mag'])[:150]
                log_msg(f"[INFO] Fetched {len(catalog_stars)} Gaia catalog stars.")
                
                if len(catalog_stars) < 10:
                    raise Exception("Too few stars in Gaia catalog search region.")
                    
                guess_wcs = WCS(naxis=2)
                guess_wcs.wcs.crpix = [w_orig / 2.0, h_orig / 2.0]
                guess_wcs.wcs.crval = [ra_deg, dec_deg]
                guess_wcs.wcs.cdelt = [-pixel_scale_deg, pixel_scale_deg]
                guess_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
                
                solved = False
                solved_wcs = None
                for parity in ["normal", "mirrored"]:
                    log_msg(f"[ALIGN] Running astroalign ({parity} parity)...")
                    cat_pts = []
                    for star in catalog_stars:
                        px, py = guess_wcs.world_to_pixel(SkyCoord(star['ra'], star['dec'], unit='deg', frame='icrs'))
                        if parity == "mirrored":
                            px = w_orig - px
                        if -100 <= px < w_orig + 100 and -100 <= py < h_orig + 100:
                            cat_pts.append((px, py))
                            
                    cat_pts = np.array(cat_pts, dtype=np.float32)
                    if len(cat_pts) < 10:
                        continue
                        
                    try:
                        transf, (s_list, t_list) = aa.find_transform(img_pts, cat_pts)
                        log_msg(f"[ALIGN] Solved! Scale: {transf.scale:.4f}, Rotation: {np.degrees(transf.rotation):.2f}°")
                        
                        S = transf.scale
                        theta = transf.rotation
                        cos_t = np.cos(theta)
                        sin_t = np.sin(theta)
                        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
                        T = transf.translation
                        
                        CD_guess = np.diag(guess_wcs.wcs.cdelt)
                        crpix_guess = np.array(guess_wcs.wcs.crpix)
                        
                        if parity == "mirrored":
                            M = np.array([[-S * R[0,0], -S * R[0,1]],
                                          [ S * R[1,0],  S * R[1,1]]])
                            K = np.array([w_orig - T[0], T[1]])
                        else:
                            M = S * R
                            K = T
                            
                        M_inv = np.linalg.inv(M)
                        crpix_new = M_inv @ (crpix_guess - K)
                        CD_new = CD_guess @ M
                            
                        solved_wcs = WCS(naxis=2)
                        solved_wcs.wcs.crval = guess_wcs.wcs.crval
                        solved_wcs.wcs.crpix = crpix_new
                        solved_wcs.wcs.cd = CD_new
                        solved_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
                        solved = True
                        break
                    except Exception as match_ex:
                        log_msg(f"[ALIGN] Failed for {parity} parity: {match_ex}")
                        
                if not solved:
                    raise Exception("Astroalign could not match the patterns. Verify focal length, pixel size, or RA/Dec coordinates.")
                    
                log_win.after(0, lambda: finish_done(solved_wcs))
            except Exception as e:
                log_win.after(0, lambda: finish_error(e))
                
        import threading
        t = threading.Thread(target=worker)
        t.daemon = True
        t.start()

    def update_wcs_hint_visibility(self):
        if self.fits_data is not None:
            job_info = self.jobs.get(self.fits_path, {})
            status = job_info.get("status", "")
            
            if self.wcs is None or not getattr(self.wcs, 'has_celestial', False):
                if status == "pending":
                    self.lbl_wcs_hint.config(
                        text=f"⏳ Online job pending (Job ID: {job_info.get('job_id')})\nUse Astrometry -> Check Status...",
                        bg="black", fg="#38bdf8", font=("Segoe UI", 9, "bold"), cursor=""
                    )
                    self.lbl_wcs_hint.pack(fill="x", padx=5, pady=5)
                else:
                    self.lbl_wcs_hint.config(
                        text="⚠️ No WCS projection found\nClick here to Plate Solve (Vizier)",
                        bg="black", fg="#facc15", font=("Segoe UI", 9, "bold", "underline"), cursor="hand2"
                    )
                    self.lbl_wcs_hint.pack(fill="x", padx=5, pady=5)
                self.btn_save_wcs_header.pack_forget()
            else:
                self.lbl_wcs_hint.pack_forget()
                if not self.wcs_saved_to_disk:
                    self.btn_save_wcs_header.pack(fill="x", padx=5, pady=5)
                else:
                    self.btn_save_wcs_header.pack_forget()

    def on_wcs_hint_click(self, event):
        job_info = self.jobs.get(self.fits_path, {}) if self.fits_path else {}
        status = job_info.get("status", "")
        if status != "pending" and self.wcs is None:
            self.platesolve_vizier_astroalign()

    def load_settings(self):
        import json
        import os
        if os.path.exists(self.settings_path):
            try:
                with open(self.settings_path, "r", encoding="utf-8") as f:
                    self.settings = json.load(f)
            except Exception:
                self.settings = {}
        else:
            self.settings = {}

    def save_settings(self):
        import json
        try:
            with open(self.settings_path, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=4)
        except Exception:
            pass

    def load_jobs(self):
        import json
        import os
        if os.path.exists(self.jobs_path):
            try:
                with open(self.jobs_path, "r", encoding="utf-8") as f:
                    self.jobs = json.load(f)
            except Exception:
                self.jobs = {}
        else:
            self.jobs = {}

    def save_jobs(self):
        import json
        try:
            with open(self.jobs_path, "w", encoding="utf-8") as f:
                json.dump(self.jobs, f, indent=4)
        except Exception:
            pass

    def configure_astrometry_api_key(self):
        import tkinter.simpledialog as sd
        current_key = self.settings.get("astrometry_net_api_key", "")
        
        msg = (
            "Inserisci la tua API Key di Nova.astrometry.net:\n\n"
            "Come ottenerla:\n"
            "1. Accedi o registrati su: https://nova.astrometry.net/\n"
            "2. Vai nel tuo 'Profile' (in alto a destra)\n"
            "3. Troverai la tua API Key nella sezione 'My API Key'\n\n"
            "La chiave verrà salvata localmente nel file 'settings.json'."
        )
        
        new_key = sd.askstring("Configura API Key Astrometry.net", msg, initialvalue=current_key, parent=self.root)
        if new_key is not None:
            self.settings["astrometry_net_api_key"] = new_key.strip()
            self.save_settings()
            messagebox.showinfo("Configurazione", "API Key salvata con successo in settings.json!", parent=self.root)

    def platesolve_nova_astrometry(self):
        if self.fits_data is None:
            messagebox.showerror("Error", "Load a FITS file first.")
            return

        api_key = self.settings.get("astrometry_net_api_key", "").strip()
        if not api_key:
            messagebox.showerror("API Key mancante", 
                                 "Configura prima la tua API Key di Astrometry.net dal menu:\n"
                                 "Astrometria -> Configura Chiave API Astrometry.net...", 
                                 parent=self.root)
            return

        self.run_platesolve_nova_astrometry(api_key)

    def run_platesolve_nova_astrometry(self, api_key):
        self.root.update()
        
        log_win = tk.Toplevel(self.root)
        log_win.title("Nova.astrometry.net Online Solver")
        log_win.geometry("600x450")
        log_win.configure(bg="#1f2937")
        log_win.transient(self.root)
        
        lbl = tk.Label(log_win, text="Uploading and Solving via Astrometry.net...", bg="#1f2937", fg="white", font=("Segoe UI", 10, "bold"))
        lbl.pack(pady=(10, 5))
        
        txt_log = tk.Text(log_win, bg="#111827", fg="#38bdf8", insertbackground="white", font=("Consolas", 9), bd=0, padx=10, pady=10)
        txt_log.pack(fill="both", expand=True, padx=15, pady=(0, 5))
        
        btn_bg = tk.Button(log_win, text="Esegui in Background (Chiudi)", command=log_win.destroy, bg="#374151", fg="white", font=("Segoe UI", 9, "bold"), bd=0, padx=10, pady=5)
        btn_bg.pack(pady=(5, 10))
        
        def log_msg(msg):
            def _log():
                if log_win.winfo_exists():
                    txt_log.insert(tk.END, msg + "\n")
                    txt_log.see(tk.END)
            if log_win.winfo_exists():
                log_win.after(0, _log)
            
        def finish_done(solved_wcs, resolved_job_id):
            try:
                log_msg("\n[SUCCESS] Image plate-solved successfully!")
                
                wcs_cards = dict(solved_wcs.to_header())
                wcs_cache = {k: v for k, v in wcs_cards.items() if any(x in k for x in ['CRPIX', 'CRVAL', 'CD', 'CTYPE', 'CUNIT', 'PC'])}
                self.jobs[self.fits_path] = {
                    "status": "success",
                    "timestamp": time.time(),
                    "wcs": wcs_cache,
                    "job_id": resolved_job_id
                }
                self.save_jobs()
                
                # Merge WCS header cards into in-memory fits_header
                for key, val in wcs_cache.items():
                    self.fits_header[key] = val
                self.fits_header['RADECSYS'] = 'ICRS'
                self.fits_header['EQUINOX'] = 2000.0
                self.wcs_saved_to_disk = False
                
                self.wcs = solved_wcs
                self.meta_tree.delete(*self.meta_tree.get_children())
                important_keys = ["OBJECT", "EXPTIME", "TELESCOP", "INSTRUME", "FILTER", "DATE-OBS", "BAYERPAT", "EQUINOX"]
                for key in important_keys:
                    if key in self.fits_header:
                        self.meta_tree.insert("", "end", text=key, values=(str(self.fits_header[key]),))
                for key, val in self.fits_header.items():
                    if key not in important_keys and key.strip() != "":
                        self.meta_tree.insert("", "end", text=key, values=(str(val),))
                        
                self.render_canvas(is_dragging=False)
                self.update_wcs_hint_visibility()
                if log_win.winfo_exists():
                    log_win.destroy()
                messagebox.showinfo("Plate Solve", "Plate solve successful! WCS coordinates loaded in memory.", parent=self.root)
            except Exception as e:
                log_msg(f"\n[FATAL ERROR] finish_done failed: {e}")
                import traceback
                log_msg(traceback.format_exc())
            
        def finish_error(err):
            log_msg(f"\n[ERROR] Plate solve failed: {err}")
            if log_win.winfo_exists():
                messagebox.showerror("Error", f"Plate solve failed: {err}", parent=log_win)

        def worker():
            try:
                import numpy as np
                import cv2
                import urllib.request
                import urllib.parse
                import ssl
                import json
                import time
                from astropy.wcs import WCS
                
                log_msg("[INFO] Compressing FITS to PNG for upload...")
                h_orig, w_orig = self.fits_data.shape[-2:]
                
                if len(self.fits_data.shape) == 3:
                    gray = np.mean(self.fits_data, axis=0).astype(np.float32)
                else:
                    gray = self.fits_data.astype(np.float32)
                    
                vmin, vmax = np.percentile(gray, [1.0, 99.5])
                stretched = np.clip((gray - vmin) / (vmax - vmin) * 255.0, 0, 255).astype(np.uint8)
                
                scale_factor = 4
                new_w = w_orig // scale_factor
                new_h = h_orig // scale_factor
                downsampled = cv2.resize(stretched, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                _, png_bytes = cv2.imencode(".png", downsampled)
                png_data = png_bytes.tobytes()
                log_msg(f"[INFO] Image compressed to {len(png_data)/1024:.1f} KB.")
                
                ssl_context = ssl._create_unverified_context()
                
                # Step 1: Login
                log_msg("[INFO] Logging in to Astrometry.net API...")
                login_url = "https://nova.astrometry.net/api/login"
                payload = {"request-json": json.dumps({"apikey": api_key})}
                data_encoded = urllib.parse.urlencode(payload).encode('utf-8')
                
                req = urllib.request.Request(login_url, data=data_encoded, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, context=ssl_context) as response:
                    resp = json.loads(response.read().decode('utf-8'))
                    session = resp.get("session")
                    
                if not session:
                    raise Exception(f"Login failed: {resp}")
                    
                log_msg("[INFO] Login successful. Session created.")
                
                # Step 2: Upload
                log_msg("[INFO] Uploading image to Astrometry.net queue...")
                upload_url = "https://nova.astrometry.net/api/upload"
                req_json = {
                    "session": session,
                    "allow_commercial_use": "d",
                    "allow_modifications": "d",
                    "publicly_visible": "n"
                }
                
                boundary = b'----Boundary' + bytes(str(time.time()), 'utf-8')
                body = bytearray()
                body.extend(b'--' + boundary + b'\r\n')
                body.extend(b'Content-Disposition: form-data; name="request-json"\r\n\r\n')
                body.extend(json.dumps(req_json).encode('utf-8') + b'\r\n')
                body.extend(b'--' + boundary + b'\r\n')
                body.extend(b'Content-Disposition: form-data; name="file"; filename="image.png"\r\n')
                body.extend(b'Content-Type: image/png\r\n\r\n')
                body.extend(png_data + b'\r\n')
                body.extend(b'--' + boundary + b'--\r\n')
                
                content_type = f'multipart/form-data; boundary={boundary.decode("utf-8")}'
                
                req = urllib.request.Request(upload_url, data=body, headers={'Content-Type': content_type, 'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, context=ssl_context) as response:
                    resp = json.loads(response.read().decode('utf-8'))
                    
                subid = resp.get("subid")
                if not subid:
                    raise Exception(f"Upload failed: {resp}")
                
                self.jobs[self.fits_path] = {
                    "subid": subid,
                    "status": "pending",
                    "timestamp": time.time()
                }
                self.save_jobs()
                self.root.after(0, self.update_wcs_hint_visibility)
                
                log_msg(f"[INFO] Upload success! Submission ID: {subid}. Waiting for job creation...")
                
                # Step 3: Poll Submission for Job ID
                job_id = None
                for _ in range(40):
                    time.sleep(3)
                    status_url = f"https://nova.astrometry.net/api/submissions/{subid}"
                    req = urllib.request.Request(status_url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, context=ssl_context) as response:
                        resp = json.loads(response.read().decode('utf-8'))
                    jobs = resp.get("jobs", [])
                    if jobs and jobs[0] is not None:
                        job_id = jobs[0]
                        break
                        
                if not job_id:
                    raise Exception("Astrometry.net took too long to create a solving job.")
                
                self.jobs[self.fits_path]["job_id"] = job_id
                self.save_jobs()
                self.root.after(0, self.update_wcs_hint_visibility)
                log_msg(f"[INFO] Job created! Job ID: {job_id}. Running online plate-solve...")
                log_msg("[INFO] Puoi chiudere questa finestra se desideri; l'elaborazione continuerà sul server.")
                
                # Step 4: Poll Job for Success
                solved = False
                for _ in range(90):
                    time.sleep(4)
                    job_url = f"https://nova.astrometry.net/api/jobs/{job_id}"
                    req = urllib.request.Request(job_url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, context=ssl_context) as response:
                        resp = json.loads(response.read().decode('utf-8'))
                    status = resp.get("status")
                    log_msg(f"[STATUS] Job status: {status}")
                    if status == "success":
                        solved = True
                        break
                    elif status == "failure":
                        self.jobs[self.fits_path]["status"] = "failed"
                        self.save_jobs()
                        self.root.after(0, self.update_wcs_hint_visibility)
                        raise Exception("Astrometry.net failed to solve this image.")
                        
                if not solved:
                    raise Exception("Plate-solving is taking a long time. You can check status later via Astrometria -> Verifica stato.")
                    
                # Step 5: Download and Scale WCS
                log_msg("[INFO] Job succeeded! Downloading solved WCS headers...")
                wcs_url = f"https://nova.astrometry.net/wcs_file/{job_id}"
                req = urllib.request.Request(wcs_url, headers={'User-Agent': 'Mozilla/5.0'})
                
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tmp_f:
                    tmp_path = tmp_f.name
                    
                try:
                    with urllib.request.urlopen(req, context=ssl_context) as response:
                        with open(tmp_path, "wb") as f:
                            f.write(response.read())
                            
                    solved_wcs = WCS(tmp_path)
                    
                    solved_wcs.wcs.crpix = [pix * scale_factor for pix in solved_wcs.wcs.crpix]
                    if solved_wcs.wcs.has_cd():
                        solved_wcs.wcs.cd = solved_wcs.wcs.cd / scale_factor
                    elif solved_wcs.wcs.cdelt is not None:
                        solved_wcs.wcs.cdelt = [delt / scale_factor for delt in solved_wcs.wcs.cdelt]
                        
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                        
                log_win.after(0, lambda: finish_done(solved_wcs, job_id))
            except Exception as e:
                log_win.after(0, lambda: finish_error(e))
                
        import threading
        t = threading.Thread(target=worker)
        t.daemon = True
        t.start()

    def check_astrometry_job_status(self):
        if self.fits_data is None:
            messagebox.showerror("Error", "Load a FITS file first.")
            return
            
        job_info = self.jobs.get(self.fits_path, {})
        status = job_info.get("status")
        if not status:
            messagebox.showinfo("Nessun Job", "Nessun lavoro pendente o salvato per questo file FITS.", parent=self.root)
            return
            
        if status == "success":
            messagebox.showinfo("Risolto", "Questo file è già stato risolto con successo. Il WCS è caricato in memoria.", parent=self.root)
            return
            
        if status == "failed":
            retry = messagebox.askyesno("Job Fallito", "Il lavoro precedente per questo file è fallito. Vuoi riprovare a risolverlo?", parent=self.root)
            if retry:
                self.platesolve_nova_astrometry()
            return
            
        subid = job_info.get("subid")
        job_id = job_info.get("job_id")
        
        self.root.update()
        
        def checker_worker():
            try:
                import urllib.request
                import urllib.parse
                import ssl
                import json
                import os
                import tempfile
                import time
                from astropy.wcs import WCS
                
                ssl_context = ssl._create_unverified_context()
                resolved_job_id = job_id
                
                if not resolved_job_id and subid:
                    status_url = f"https://nova.astrometry.net/api/submissions/{subid}"
                    req = urllib.request.Request(status_url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, context=ssl_context) as response:
                        resp = json.loads(response.read().decode('utf-8'))
                    jobs = resp.get("jobs", [])
                    if jobs and jobs[0] is not None:
                        resolved_job_id = jobs[0]
                        self.jobs[self.fits_path]["job_id"] = resolved_job_id
                        self.save_jobs()
                        
                if not resolved_job_id:
                    messagebox.showinfo("In attesa", "Astrometry.net non ha ancora creato un job di risoluzione per questo caricamento. Riprova tra poco.", parent=self.root)
                    self.root.after(0, self.update_wcs_hint_visibility)
                    return
                    
                job_url = f"https://nova.astrometry.net/api/jobs/{resolved_job_id}"
                req = urllib.request.Request(job_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, context=ssl_context) as response:
                    resp = json.loads(response.read().decode('utf-8'))
                
                job_status = resp.get("status")
                
                if job_status in ["pending", "solving"]:
                    messagebox.showinfo("In elaborazione", f"Il server sta ancora elaborando il lavoro (Stato: {job_status}).\nSi prega di attendere ed eseguire la verifica più tardi.", parent=self.root)
                    return
                elif job_status == "failure":
                    self.jobs[self.fits_path]["status"] = "failed"
                    self.save_jobs()
                    messagebox.showerror("Fallito", "Il server Astrometry.net non è riuscito a risolvere l'immagine.", parent=self.root)
                    self.root.after(0, self.update_wcs_hint_visibility)
                    return
                elif job_status == "success":
                    wcs_url = f"https://nova.astrometry.net/wcs_file/{resolved_job_id}"
                    req = urllib.request.Request(wcs_url, headers={'User-Agent': 'Mozilla/5.0'})
                    
                    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tmp_f:
                        tmp_path = tmp_f.name
                        
                    try:
                        with urllib.request.urlopen(req, context=ssl_context) as response:
                            with open(tmp_path, "wb") as f:
                                f.write(response.read())
                                
                        solved_wcs = WCS(tmp_path)
                        
                        scale_factor = 4
                        solved_wcs.wcs.crpix = [pix * scale_factor for pix in solved_wcs.wcs.crpix]
                        if solved_wcs.wcs.has_cd():
                            solved_wcs.wcs.cd = solved_wcs.wcs.cd / scale_factor
                        elif solved_wcs.wcs.cdelt is not None:
                            solved_wcs.wcs.cdelt = [delt / scale_factor for delt in solved_wcs.wcs.cdelt]
                            
                        self.wcs = solved_wcs
                        
                        wcs_cards = dict(solved_wcs.to_header())
                        wcs_cache = {k: v for k, v in wcs_cards.items() if any(x in k for x in ['CRPIX', 'CRVAL', 'CD', 'CTYPE', 'CUNIT', 'PC'])}
                        self.jobs[self.fits_path] = {
                            "status": "success",
                            "timestamp": time.time(),
                            "wcs": wcs_cache,
                            "job_id": resolved_job_id
                        }
                        self.save_jobs()
                        
                        # Merge WCS header cards into in-memory fits_header
                        for key, val in wcs_cache.items():
                            self.fits_header[key] = val
                        self.fits_header['RADECSYS'] = 'ICRS'
                        self.fits_header['EQUINOX'] = 2000.0
                        self.wcs_saved_to_disk = False
                        
                        self.meta_tree.delete(*self.meta_tree.get_children())
                        important_keys = ["OBJECT", "EXPTIME", "TELESCOP", "INSTRUME", "FILTER", "DATE-OBS", "BAYERPAT", "EQUINOX"]
                        for key in important_keys:
                            if key in self.fits_header:
                                self.meta_tree.insert("", "end", text=key, values=(str(self.fits_header[key]),))
                        for key, val in self.fits_header.items():
                            if key not in important_keys and key.strip() != "":
                                self.meta_tree.insert("", "end", text=key, values=(str(val),))
                                
                        self.root.after(0, lambda: self.render_canvas(is_dragging=False))
                        self.root.after(0, self.update_wcs_hint_visibility)
                        messagebox.showinfo("Plate Solve", "Plate solve completato con successo! Coordinate WCS caricate in memoria.", parent=self.root)
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                else:
                    messagebox.showinfo("Info", f"Risposta del server sconosciuta: {job_status}", parent=self.root)
            except Exception as checker_ex:
                messagebox.showerror("Errore durante la verifica", f"Errore: {checker_ex}", parent=self.root)
                
        import threading
        t = threading.Thread(target=checker_worker)
        t.daemon = True
        t.start()

    def save_wcs_to_fits_file(self):
        if self.wcs is None:
            messagebox.showerror("Error", "No solved WCS coordinates available to save.", parent=self.root)
            return
        if not self.fits_path:
            messagebox.showerror("Error", "No FITS file path defined.", parent=self.root)
            return
            
        try:
            from astropy.io import fits
            with fits.open(self.fits_path, mode='update') as hdul:
                header = hdul[0].header
                for key in ['CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 
                            'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 
                            'CDELT1', 'CDELT2', 'CTYPE1', 'CTYPE2',
                            'CUNIT1', 'CUNIT2', 'RADECSYS', 'EQUINOX']:
                    if key in header:
                        del header[key]
                
                for key, val in self.wcs.to_header().items():
                    if any(k in key for k in ['CRPIX', 'CRVAL', 'CD', 'CTYPE', 'CUNIT', 'PC']):
                        header[key] = val
                header['RADECSYS'] = 'ICRS'
                header['EQUINOX'] = 2000.0
                self.fits_header = header.copy()
                self.wcs_saved_to_disk = True
                
            self.meta_tree.delete(*self.meta_tree.get_children())
            important_keys = ["OBJECT", "EXPTIME", "TELESCOP", "INSTRUME", "FILTER", "DATE-OBS", "BAYERPAT", "EQUINOX"]
            for key in important_keys:
                if key in self.fits_header:
                    self.meta_tree.insert("", "end", text=key, values=(str(self.fits_header[key]),))
            for key, val in self.fits_header.items():
                if key not in important_keys and key.strip() != "":
                    self.meta_tree.insert("", "end", text=key, values=(str(val),))
                    
            messagebox.showinfo("Successo", "WCS salvato con successo nell'header del file FITS su disco!", parent=self.root)
            self.btn_save_wcs_header.pack_forget()
        except Exception as ex:
            messagebox.showerror("Save Error", f"Impossibile salvare sul file FITS: {ex}", parent=self.root)

    def clear_annotations(self):
        self.push_state()
        self.annotations = []
        self.temp_marker = None
        self.render_canvas(is_dragging=False)
        messagebox.showinfo("Clear Target", "All annotations and target markers cleared.", parent=self.root)

    def save_annotations_to_fits(self):
        if not self.fits_path:
            messagebox.showerror("Error", "No FITS file open.", parent=self.root)
            return
            
        if not self.annotations:
            messagebox.showwarning("Save Annotations", "No annotations to save.", parent=self.root)
            return
            
        try:
            from astropy.io import fits
            import numpy as np
            import re
            from astropy.coordinates import FK5
            
            # Resolve image shape for WCS conversions
            h_orig, w_orig = self.debayered_cache.shape[:2] if self.debayered_cache is not None else (1, 1)
            
            # Build arrays for columns
            x_ratios = []
            y_ratios = []
            texts = []
            ras = []
            decs = []
            mags = []
            
            # Strip null characters and pad strings with spaces manually to prevent trailing null byte issues in fits viewers
            texts_raw = [ann['text'].strip() for ann in self.annotations]
            max_len = max(1, max(len(t) for t in texts_raw))
            
            for ann in self.annotations:
                x_r = float(ann['x'])
                y_r = float(ann['y'])
                x_ratios.append(x_r)
                y_ratios.append(y_r)
                texts.append(ann['text'].strip().ljust(max_len))
                
                # Resolve RA/DEC
                ra_val = np.nan
                dec_val = np.nan
                if 'ra' in ann and ann['ra'] is not None:
                    ra_val = float(ann['ra'])
                    dec_val = float(ann['dec'])
                elif self.wcs and h_orig > 1:
                    try:
                        px = int(round(x_r * w_orig))
                        py = int(round(y_r * h_orig))
                        sky_coord = self.wcs.pixel_to_world(px, py)
                        j2000 = sky_coord.transform_to(FK5(equinox='J2000'))
                        ra_val = float(j2000.ra.deg)
                        dec_val = float(j2000.dec.deg)
                    except Exception:
                        pass
                ras.append(ra_val)
                decs.append(dec_val)
                
                # Resolve MAG
                mag_val = np.nan
                if 'mag' in ann and ann['mag'] is not None:
                    mag_val = float(ann['mag'])
                else:
                    match = re.search(r'(?:Mag|V|mag)\s*=\s*([-+]?\d*\.\d+|\d+)', ann['text'])
                    if match:
                        try:
                            mag_val = float(match.group(1))
                        except ValueError:
                            pass
                mags.append(mag_val)
                
            ext = os.path.splitext(self.fits_path)[1].lower()
            is_fits = ext in ['.fits', '.fit']
            
            if not is_fits:
                # For non-FITS files, save to sidecar file
                sidecar_path = self.fits_path + ".info.json"
                import json
                try:
                    # Load existing sidecar data if present
                    sidecar_data = {}
                    if os.path.exists(sidecar_path):
                        try:
                            with open(sidecar_path, "r", encoding="utf-8") as sf:
                                sidecar_data = json.load(sf)
                        except Exception:
                            pass
                    
                    # Update annotations and header
                    sidecar_data["annotations"] = self.annotations
                    sidecar_data["header"] = dict(self.fits_header)
                    
                    with open(sidecar_path, "w", encoding="utf-8") as sf:
                        json.dump(sidecar_data, sf, indent=4)
                    messagebox.showinfo("Success", f"Saved {len(self.annotations)} annotations to sidecar file:\n{os.path.basename(sidecar_path)}", parent=self.root)
                except Exception as sf_err:
                    messagebox.showerror("Error", f"Failed to save annotations to sidecar:\n{sf_err}", parent=self.root)
                return

            # Create columns
            col_x = fits.Column(name='X_RATIO', format='D', array=x_ratios)
            col_y = fits.Column(name='Y_RATIO', format='D', array=y_ratios)
            col_text = fits.Column(name='TEXT', format=f'{max_len}A', array=texts)
            col_ra = fits.Column(name='RA', format='D', array=ras)
            col_dec = fits.Column(name='DEC', format='D', array=decs)
            col_mag = fits.Column(name='MAG', format='D', array=mags)
            
            tb_hdu = fits.BinTableHDU.from_columns([col_x, col_y, col_text, col_ra, col_dec, col_mag], name='ANNOTATIONS')
            
            with fits.open(self.fits_path, mode='update') as hdul:
                if 'ANNOTATIONS' in hdul:
                    del hdul['ANNOTATIONS']
                hdul.append(tb_hdu)
                hdul.flush()
                
            messagebox.showinfo("Success", f"Saved {len(self.annotations)} annotations with WCS coords and magnitudes to FITS table extension.", parent=self.root)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save annotations to FITS:\n{str(e)}", parent=self.root)

    def import_annotations_from_fits(self):
        if not self.fits_path:
            messagebox.showerror("Error", "No image open.", parent=self.root)
            return
            
        ext = os.path.splitext(self.fits_path)[1].lower()
        is_fits = ext in ['.fits', '.fit']
        
        if not is_fits:
            sidecar_path = self.fits_path + ".info.json"
            if not os.path.exists(sidecar_path):
                messagebox.showinfo("Import Annotations", f"No sidecar file found at:\n{os.path.basename(sidecar_path)}", parent=self.root)
                return
            try:
                import json
                with open(sidecar_path, "r", encoding="utf-8") as sf:
                    sidecar_data = json.load(sf)
                imported = sidecar_data.get("annotations", [])
                if imported:
                    self.push_state()
                    existing_coords = {(ann['x'], ann['y']) for ann in self.annotations}
                    added_count = 0
                    for item in imported:
                        if (item['x'], item['y']) not in existing_coords:
                            self.annotations.append(item)
                            added_count += 1
                    self.render_canvas(is_dragging=False)
                    messagebox.showinfo("Import Success", f"Imported {added_count} new annotations from sidecar (total annotations: {len(self.annotations)}).", parent=self.root)
                else:
                    messagebox.showinfo("Import Annotations", "No annotations records found in the sidecar file.", parent=self.root)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to import from sidecar:\n{str(e)}", parent=self.root)
            return
            
        try:
            from astropy.io import fits
            with fits.open(self.fits_path) as hdul:
                if 'ANNOTATIONS' not in hdul:
                    messagebox.showinfo("Import Annotations", "No annotations table extension found in this FITS file.", parent=self.root)
                    return
                
                table = hdul['ANNOTATIONS'].data
                imported = []
                # Check for columns
                col_names = table.columns.names
                
                for row in table:
                    x_val = float(row['X_RATIO'])
                    y_val = float(row['Y_RATIO'])
                    
                    text_val = row['TEXT']
                    if isinstance(text_val, bytes):
                        text_val = text_val.decode('utf-8', errors='ignore')
                    text_val = str(text_val).strip()
                    
                    # Read optional columns
                    ra_val = float(row['RA']) if 'RA' in col_names else None
                    dec_val = float(row['DEC']) if 'DEC' in col_names else None
                    mag_val = float(row['MAG']) if 'MAG' in col_names else None
                    
                    if ra_val is not None and np.isnan(ra_val): ra_val = None
                    if dec_val is not None and np.isnan(dec_val): dec_val = None
                    if mag_val is not None and np.isnan(mag_val): mag_val = None
                    
                    imported.append({
                        'x': x_val,
                        'y': y_val,
                        'text': text_val,
                        'ra': ra_val,
                        'dec': dec_val,
                        'mag': mag_val
                    })
                
                if imported:
                    self.push_state()
                    existing_coords = {(ann['x'], ann['y']) for ann in self.annotations}
                    added_count = 0
                    for item in imported:
                        if (item['x'], item['y']) not in existing_coords:
                            self.annotations.append(item)
                            added_count += 1
                            
                    self.render_canvas(is_dragging=False)
                    messagebox.showinfo("Import Success", f"Imported {added_count} new annotations (total annotations: {len(self.annotations)}).", parent=self.root)
                else:
                    messagebox.showinfo("Import Annotations", "No annotations records found in the FITS extension table.", parent=self.root)
                    
        except Exception as e:
            messagebox.showerror("Error", f"Failed to import annotations:\n{str(e)}", parent=self.root)

    def toggle_gaia_search_mode(self):
        self.gaia_search_mode = not getattr(self, 'gaia_search_mode', False)
        self.gaia_search_var.set(self.gaia_search_mode)
        if self.gaia_search_mode:
            self.calibration_mode = False
            self.measurement_mode = False
            self.crop_mode = False
            self.annotation_mode = False
            self.balance_mode = "None"
            self.btn_calibration.config(text="Mark Calib Stars: Off", bg="#374151")
            self.btn_measurement.config(text="Measure Target: Off", bg="#374151")
            self.btn_crop.config(text="Crop Mode: Off", bg="#374151")
            self.btn_annotate.config(text="Annotate Mode: Off", bg="#374151")
            self.canvas.config(cursor="question_arrow")
            messagebox.showinfo("Gaia Search", "Gaia Search Mode enabled.\nClick on any star to query Gaia DR3 catalog details.", parent=self.root)
        else:
            self.canvas.config(cursor="")
            
    def handle_gaia_search_click(self, px_x, px_y):
        if not self.wcs:
            messagebox.showwarning("Gaia Search", "Gaia queries require a valid plate-solved WCS model. Please solve the FITS image first.", parent=self.root)
            return
            
        clicked_sky = self.wcs.pixel_to_world(px_x, px_y)
        ra_deg = clicked_sky.ra.deg
        dec_deg = clicked_sky.dec.deg
        
        h_orig, w_orig = self.debayered_cache.shape[:2]
        self.temp_marker = {
            'ratio_x': px_x / w_orig,
            'ratio_y': px_y / h_orig,
            'ra': ra_deg,
            'dec': dec_deg,
            'type': 'gaia_query'
        }
        self.render_canvas(is_dragging=False)
        
        # Open a loading/waiting window
        info_win = tk.Toplevel(self.root)
        info_win.title("Gaia Object Query")
        info_win.geometry("550x500")
        info_win.configure(bg=self.panel_color)
        info_win.transient(self.root)
        
        lbl_loading = tk.Label(info_win, text=f"Querying Gaia DR3 database...\nRA: {ra_deg:.6f}°, DEC: {dec_deg:.6f}°", bg=self.panel_color, fg="#38bdf8", font=("Segoe UI", 11, "bold"))
        lbl_loading.pack(pady=40)
        
        txt_info = tk.Text(info_win, bg=self.bg_color, fg=self.text_color, insertbackground="white", bd=0, highlightthickness=1, highlightbackground=self.control_bg, font=("Consolas", 10))
        import webbrowser
        txt_info.tag_configure("link", foreground="#38bdf8", underline=1)
        txt_info.tag_bind("link", "<Enter>", lambda e: txt_info.config(cursor="hand2"))
        txt_info.tag_bind("link", "<Leave>", lambda e: txt_info.config(cursor="xterm"))
        
        def run_query():
            try:
                import urllib.request
                import urllib.parse
                import ssl
                
                c_str = urllib.parse.quote(f"{ra_deg:.6f} {dec_deg:.6f}")
                radius_arcmin = 0.16667 # 10 arcseconds
                cols = "Source,RA_ICRS,DE_ICRS,Plx,e_Plx,pmRA,pmDE,Gmag,BPmag,RPmag,RV,Teff,phot_variable_flag"
                url = f"https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=I/355/gaiadr3&-c={c_str}&-c.r={radius_arcmin:f}&-c.u=arcmin&-out.form=|&-out.max=1&-out={cols}"
                
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                ssl_context = ssl._create_unverified_context()
                
                content = ""
                try:
                    with urllib.request.urlopen(req, timeout=15, context=ssl_context) as response:
                        content = response.read().decode('utf-8')
                except Exception as ex:
                    # Try Harvard mirror
                    url_harv = url.replace("vizier.cds.unistra.fr", "vizier.cfa.harvard.edu")
                    req_harv = urllib.request.Request(url_harv, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req_harv, timeout=15, context=ssl_context) as response:
                        content = response.read().decode('utf-8')
                        
                lines = content.split('\n')
                data_line = None
                header_line = None
                
                for line in lines:
                    if line.startswith("#"):
                        continue
                    if line.strip() == "":
                        continue
                    if "Source" in line and "RA_ICRS" in line:
                        header_line = line.strip()
                        continue
                    if line.startswith("-"):
                        continue
                    
                    # Ensure this is a real data line by checking that the first part is a Source ID (integer digits)
                    parts = line.strip().split('|')
                    if len(parts) > 0 and parts[0].strip().isdigit():
                        data_line = line.strip()
                        break
                    
                if not data_line:
                    def _show_no_star():
                        lbl_loading.pack_forget()
                        lbl_err = tk.Label(info_win, text="No Gaia DR3 star found near this coordinate\n(within 10 arcseconds radius).", bg=self.panel_color, fg="#facc15", font=("Segoe UI", 11, "bold"))
                        lbl_err.pack(pady=100)
                    info_win.after(0, _show_no_star)
                    return
                    
                headers = [h.strip() for h in header_line.split('|')]
                values = [v.strip() for v in data_line.split('|')]
                star_data = dict(zip(headers, values))
                
                cat_ra = ra_deg
                cat_dec = dec_deg
                # Calculate and project actual Gaia star position back to FITS pixels
                try:
                    cat_ra = float(star_data.get('RA_ICRS'))
                    cat_dec = float(star_data.get('DE_ICRS'))
                    c_px, c_py = self.wcs.world_to_pixel(SkyCoord(cat_ra, cat_dec, unit='deg', frame='icrs'))
                    c_px = float(np.atleast_1d(c_px)[0])
                    c_py = float(np.atleast_1d(c_py)[0])
                    self.temp_marker['catalog_ratio_x'] = c_px / w_orig
                    self.temp_marker['catalog_ratio_y'] = c_py / h_orig
                except Exception:
                    pass
                
                # Build beautiful report
                report = []
                report.append("==================================================")
                report.append("            GAIA DR3 OBJECT METADATA             ")
                report.append("==================================================")
                report.append(f"Gaia Source ID : {star_data.get('Source', 'N/A')}")
                report.append(f"Right Ascension: {star_data.get('RA_ICRS', 'N/A')} deg")
                report.append(f"Declination    : {star_data.get('DE_ICRS', 'N/A')} deg")
                report.append("")
                
                report.append("--- Astrometric Parameters ---")
                plx_val = star_data.get('Plx', '')
                e_plx_val = star_data.get('e_Plx', '')
                if plx_val:
                    try:
                        plx = float(plx_val)
                        e_plx = float(e_plx_val) if e_plx_val else 0.0
                        report.append(f"Parallax       : {plx:.4f} +/- {e_plx:.4f} mas")
                        if plx > 0:
                            dist_pc = 1000.0 / plx
                            dist_ly = dist_pc * 3.26156
                            report.append(f"Distance       : {dist_pc:.1f} pc ({dist_ly:.1f} light years)")
                        else:
                            report.append("Distance       : Infinite (Negative or zero parallax)")
                    except ValueError:
                        report.append(f"Parallax       : {plx_val} mas")
                else:
                    report.append("Parallax       : N/A")
                    
                report.append(f"Proper Motion RA: {star_data.get('pmRA', 'N/A')} mas/yr")
                report.append(f"Proper Motion DEC: {star_data.get('pmDE', 'N/A')} mas/yr")
                report.append(f"Radial Velocity: {star_data.get('RV', 'N/A')} km/s")
                report.append("")
                
                report.append("--- Photometric Magnitudes ---")
                report.append(f"G Magnitude    : {star_data.get('Gmag', 'N/A')}")
                bp = star_data.get('BPmag', '')
                rp = star_data.get('RPmag', '')
                report.append(f"BP Magnitude   : {bp if bp else 'N/A'}")
                report.append(f"RP Magnitude   : {rp if rp else 'N/A'}")
                if bp and rp:
                    try:
                        color_idx = float(bp) - float(rp)
                        report.append(f"BP - RP Color  : {color_idx:.3f}")
                    except ValueError:
                        pass
                report.append(f"Variable Star? : {'YES' if star_data.get('phot_variable_flag') == 'VARIABLE' else 'NO'}")
                report.append("")
                
                report.append("--- Physical Properties ---")
                teff_val = star_data.get('Teff', '')
                if teff_val:
                    try:
                        teff = float(teff_val)
                        report.append(f"Effective Temp : {teff:.1f} K")
                        # Spectral class classification
                        if teff >= 30000:
                            sp_class = "O (Hot Blue Giant)"
                        elif teff >= 10000:
                            sp_class = "B (Blue-White Star)"
                        elif teff >= 7500:
                            sp_class = "A (White Main Sequence Star)"
                        elif teff >= 6000:
                            sp_class = "F (Yellow-White Star)"
                        elif teff >= 5200:
                            sp_class = "G (Yellow Dwarf - Sun-like)"
                        elif teff >= 3700:
                            sp_class = "K (Orange Dwarf)"
                        else:
                            sp_class = "M (Red Dwarf / Red Giant)"
                        report.append(f"Spectral Class : Class {sp_class}")
                    except ValueError:
                        report.append(f"Effective Temp : {teff_val} K")
                else:
                    report.append("Effective Temp : N/A")
                    
                report.append("==================================================")
                
                final_text = "\n".join(report)
                
                def _show_result():
                    lbl_loading.pack_forget()
                    txt_info.pack(padx=10, pady=10, fill="both", expand=True)
                    txt_info.insert(tk.END, final_text)
                    
                    url_aladin = f"https://aladin.cds.unistra.fr/AladinLite/?target={cat_ra:.6f}%20{cat_dec:.6f}&fov=0.15"
                    txt_info.tag_bind("link", "<Button-1>", lambda e: webbrowser.open_new_tab(url_aladin))
                    
                    txt_info.insert(tk.END, "\n\n🔗 View on Aladin Lite (interactive sky map)", "link")
                    txt_info.insert(tk.END, "\n==================================================")
                    
                    txt_info.config(state=tk.DISABLED)
                    self.render_canvas(is_dragging=False)
                info_win.after(0, _show_result)
                
            except Exception as query_ex:
                def _show_err():
                    lbl_loading.pack_forget()
                    lbl_err = tk.Label(info_win, text=f"Query failed:\n{query_ex}", bg=self.panel_color, fg="#ef4444", font=("Segoe UI", 11, "bold"))
                    lbl_err.pack(pady=100)
                info_win.after(0, _show_err)
                
        import threading
        t = threading.Thread(target=run_query)
        t.daemon = True
        t.start()
    def clear_calibration_stars(self):
        self.calib_stars = []
        self.render_canvas(is_dragging=False)
        messagebox.showinfo("Photometry", "Calibration stars list cleared.", parent=self.root)

    def auto_calibrate_photometry(self):
        if self.fits_data is None:
            messagebox.showerror("Error", "Load a FITS file first.", parent=self.root)
            return
        if not self.wcs:
            messagebox.showerror("Error", "WCS solution is required. Please plate-solve the image first.", parent=self.root)
            return
        if not self.catalog_stars:
            messagebox.showerror("Error", "No catalog stars loaded. Please download Vizier Calibration Stars first.", parent=self.root)
            return
            
        h_orig, w_orig = self.debayered_cache.shape[:2]
        img_gray = 0.2126 * self.debayered_cache[:,:,0] + 0.7152 * self.debayered_cache[:,:,1] + 0.0722 * self.debayered_cache[:,:,2]
        
        self.calib_stars = []
        matched_count = 0
        
        for star in self.catalog_stars:
            if star.get('is_variable'):
                continue
                
            # Project catalog star to pixels
            try:
                px_x, px_y = self.wcs.world_to_pixel(SkyCoord(star['ra'], star['dec'], unit='deg', frame='icrs'))
                px_x = float(np.atleast_1d(px_x)[0])
                px_y = float(np.atleast_1d(px_y)[0])
            except Exception:
                continue
                
            if 0 <= px_x < w_orig and 0 <= px_y < h_orig:
                # Measure aperture flux at this catalog pixel coordinate
                flux, cx, cy = self.measure_aperture_flux(img_gray, px_x, px_y)
                
                # Verify that:
                # 1. Net flux is strong enough (>15.0) to avoid using weak background noise fluctuations
                # 2. The centroid didn't drift more than 5.0 pixels from the projected catalog star coordinate
                if flux > 15.0 and np.hypot(cx - px_x, cy - px_y) < 5.0:
                    flux_r, _, _ = self.measure_aperture_flux(self.debayered_cache[:,:,0], px_x, px_y)
                    flux_b, _, _ = self.measure_aperture_flux(self.debayered_cache[:,:,2], px_x, px_y)
                    self.calib_stars.append({
                        'x': cx,
                        'y': cy,
                        'flux': flux,
                        'flux_r': flux_r,
                        'flux_b': flux_b,
                        'mag': star['mag'],
                        'color_index': star.get('color_index', 0.0)
                    })
                    matched_count += 1
                    
        self.show_catalog_stars.set(True)
        self.render_canvas(is_dragging=False)
        
        if matched_count > 0:
            # Report the automated zero point
            zp_list = [s['mag'] + 2.5 * np.log10(s['flux']) for s in self.calib_stars]
            avg_zp = np.mean(zp_list)
            std_zp = np.std(zp_list) if len(zp_list) > 1 else 0.0
            
            messagebox.showinfo("Auto-Calibration Success", 
                                f"Photometry auto-calibrated successfully!\n\n"
                                f"Matched stars: {matched_count}\n"
                                f"Average Zero Point: {avg_zp:.3f} +/- {std_zp:.3f}\n"
                                f"All {matched_count} stars are now selected as calibration stars (marked in green).",
                                parent=self.root)
        else:
            messagebox.showwarning("Auto-Calibration Failure", 
                                   "Could not match any catalog stars with sufficient signal-to-noise ratio in the image.\n"
                                   "Ensure the image is plate-solved and stars are well-focused.",
                                   parent=self.root)

    def estimate_limiting_magnitude(self):
        if self.fits_data is None:
            messagebox.showerror("Error", "Load an image first.", parent=self.root)
            return
        if not self.wcs:
            messagebox.showerror("Error", "WCS solution is required. Please plate-solve the image first.", parent=self.root)
            return
            
        self.root.config(cursor="watch")
        self.root.update()
        
        log_win = tk.Toplevel(self.root)
        log_win.title("Limiting Magnitude Query Monitor")
        log_win.geometry("600x380")
        log_win.configure(bg="#1f2937")
        log_win.transient(self.root)
        
        lbl = tk.Label(log_win, text="Querying Vizier Gaia DR3 for deep calibration stars...", bg="#1f2937", fg="white", font=("Segoe UI", 10, "bold"))
        lbl.pack(pady=(10, 5))
        
        txt_log = tk.Text(log_win, bg="#111827", fg="#eab308", insertbackground="white", font=("Consolas", 9), bd=0, padx=10, pady=10)
        txt_log.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        def log_msg(msg):
            txt_log.insert(tk.END, msg + "\n")
            txt_log.see(tk.END)
            log_win.update()
            
        try:
            h_orig, w_orig = self.debayered_cache.shape[:2]
            img_gray = 0.2126 * self.debayered_cache[:,:,0] + 0.7152 * self.debayered_cache[:,:,1] + 0.0722 * self.debayered_cache[:,:,2]
            
            # 1. Target center coordinate and image diagonal radius
            center_coord = self.wcs.pixel_to_world(w_orig / 2.0, h_orig / 2.0)
            ra_deg = center_coord.ra.deg
            dec_deg = center_coord.dec.deg
            log_msg(f"[INFO] Targeting image center: RA={ra_deg:.6f} deg, DEC={dec_deg:.6f} deg")
            
            corner_coord = self.wcs.pixel_to_world(0, 0)
            radius_deg = center_coord.separation(corner_coord).deg
            radius_arcmin = radius_deg * 60.0
            radius_arcmin = np.clip(radius_arcmin, 5.0, 45.0)
            log_msg(f"[INFO] Search radius: {radius_arcmin:.2f} arcminutes")
            
            # 2. Download from Vizier Gaia DR3
            import urllib.request
            import urllib.parse
            import ssl
            
            ssl_context = ssl._create_unverified_context()
            c_val = f"{ra_deg:.6f} {dec_deg:.6f}".replace(',', '.')
            c_str = urllib.parse.quote(c_val)
            
            hosts = ["vizier.cds.unistra.fr", "vizier.cfa.harvard.edu"]
            lines = []
            for host in hosts:
                url = f"https://{host}/viz-bin/asu-tsv?-source=I/355/gaiadr3&-c={c_str}&-c.r={radius_arcmin:f}&-c.u=arcmin&-out.form=|&-out.add=RA_ICRS,DE_ICRS&-out=Gmag,bp_rp,phot_variable_flag&-sort=_r&-out.max=4000"
                log_msg(f"[QUERY] Fetching deep Gaia catalog from: {host}...")
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                try:
                    with urllib.request.urlopen(req, timeout=12, context=ssl_context) as response:
                        lines = response.read().decode('utf-8').split('\n')
                    if len(lines) > 50:
                        log_msg(f"[SUCCESS] Retrieved {len(lines)} raw lines from {host}")
                        break
                except Exception as query_err:
                    log_msg(f"[WARN] Failed mirror {host}: {query_err}")
                    
            if not lines:
                raise Exception("Failed to download Gaia DR3 catalog from Vizier mirrors.")
                
            # 3. Parse Gaia stars
            header_found = False
            cols = {}
            local_catalog = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split('|')
                if not header_found:
                    if any("RA" in p or "coord" in p or "_RAJ" in p for p in parts):
                        header_found = True
                        cols = {p.strip(): i for i, p in enumerate(parts)}
                    continue
                if line.startswith("-"):
                    continue
                if len(parts) >= 3:
                    try:
                        ra_idx = cols.get("RA_ICRS", cols.get("_RA_ICRS", 0))
                        dec_idx = cols.get("DE_ICRS", cols.get("_DE_ICRS", 1))
                        ra_star = float(parts[ra_idx])
                        dec_star = float(parts[dec_idx])
                        
                        gmag_idx = cols.get("Gmag", cols.get("_Gmag", -1))
                        if gmag_idx == -1 or parts[gmag_idx].strip() == "":
                            continue
                        mag = float(parts[gmag_idx])
                        
                        var_flag = ""
                        var_idx = cols.get("phot_variable_flag", -1)
                        if var_idx != -1 and var_idx < len(parts):
                            var_flag = parts[var_idx].strip()
                        if var_flag == "VARIABLE":
                            continue
                            
                        local_catalog.append({
                            'ra': ra_star,
                            'dec': dec_star,
                            'mag': mag
                        })
                    except Exception:
                        pass
                        
            log_msg(f"[INFO] Parsed {len(local_catalog)} Gaia stars from catalog.")
            if not local_catalog:
                raise Exception("No usable stars parsed from Gaia DR3 catalog.")
                
            # 4. Perform automatic calibration internally
            log_msg("[CALIB] Autocalibrating zero point internally...")
            zp_values = []
            for star in local_catalog:
                # Use stars between mag 12 and 18 for high reliability
                if 12.0 <= star['mag'] <= 18.0:
                    try:
                        px_x, px_y = self.wcs.world_to_pixel(SkyCoord(star['ra'], star['dec'], unit='deg', frame='icrs'))
                        px_x = float(np.atleast_1d(px_x)[0])
                        px_y = float(np.atleast_1d(px_y)[0])
                    except Exception:
                        continue
                    if 0 <= px_x < w_orig and 0 <= px_y < h_orig:
                        flux, cx, cy = self.measure_aperture_flux(img_gray, px_x, px_y)
                        if flux > 15.0 and np.hypot(cx - px_x, cy - px_y) < 4.0:
                            zp = star['mag'] + 2.5 * np.log10(flux)
                            zp_values.append(zp)
                            
            if len(zp_values) < 5:
                # Fallback to broader range
                for star in local_catalog:
                    if 10.0 <= star['mag'] <= 19.5:
                        try:
                            px_x, px_y = self.wcs.world_to_pixel(SkyCoord(star['ra'], star['dec'], unit='deg', frame='icrs'))
                            px_x = float(np.atleast_1d(px_x)[0])
                            px_y = float(np.atleast_1d(px_y)[0])
                        except Exception:
                            continue
                        if 0 <= px_x < w_orig and 0 <= px_y < h_orig:
                            flux, cx, cy = self.measure_aperture_flux(img_gray, px_x, px_y)
                            if flux > 5.0 and np.hypot(cx - px_x, cy - px_y) < 4.0:
                                zp = star['mag'] + 2.5 * np.log10(flux)
                                zp_values.append(zp)
                                
            if not zp_values:
                raise Exception("Could not calibrate photometry: no matching catalog stars detected in the image.")
                
            median_zp = np.median(zp_values)
            mad = np.median(np.abs(zp_values - median_zp))
            valid_zps = [zp for zp in zp_values if abs(zp - median_zp) <= max(0.5, 3.0 * mad)]
            zero_point = np.mean(valid_zps) if valid_zps else median_zp
            log_msg(f"[CALIB] Internal Zero Point determined: {zero_point:.3f} (using {len(valid_zps)} reference stars)")
            
            # 5. Measure detection rates in 1-mag bins
            log_msg("[ANALYSIS] Estimating detection statistics for all catalog stars...")
            bin_edges = np.arange(8.0, 23.0, 1.0)
            bins_data = {int(b): {'total': 0, 'detected': 0, 'snrs': [], 'stars': []} for b in bin_edges[:-1]}
            
            for star in local_catalog:
                try:
                    px_x, px_y = self.wcs.world_to_pixel(SkyCoord(star['ra'], star['dec'], unit='deg', frame='icrs'))
                    px_x = float(np.atleast_1d(px_x)[0])
                    px_y = float(np.atleast_1d(px_y)[0])
                except Exception:
                    continue
                    
                if 0 <= px_x < w_orig and 0 <= px_y < h_orig:
                    mag = star['mag']
                    bin_idx = int(np.floor(mag))
                    if bin_idx not in bins_data:
                        continue
                        
                    bins_data[bin_idx]['total'] += 1
                    
                    flux, cx, cy = self.measure_aperture_flux(img_gray, px_x, px_y)
                    
                    iy, ix = int(round(cy)), int(round(cx))
                    crop_size = 15
                    y_min, y_max = max(0, iy - crop_size), min(h_orig, iy + crop_size + 1)
                    x_min, x_max = max(0, ix - crop_size), min(w_orig, ix + crop_size + 1)
                    patch = img_gray[y_min:y_max, x_min:x_max]
                    yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
                    dists = np.hypot(xx - cx, yy - cy)
                    bg_mask = (dists >= 10) & (dists <= 15)
                    bg_pixels = patch[bg_mask]
                    bg_std = np.std(bg_pixels) if len(bg_pixels) > 0 else 1.0
                    if bg_std <= 0.0:
                        bg_std = 1.0
                        
                    num_source_pixels = (dists <= 6).sum()
                    snr = flux / (np.sqrt(num_source_pixels) * bg_std)
                    
                    is_detected = flux > 8.0 and snr >= 2.5 and np.hypot(cx - px_x, cy - px_y) < 4.0
                    
                    if is_detected:
                        bins_data[bin_idx]['detected'] += 1
                        bins_data[bin_idx]['snrs'].append(snr)
                        bins_data[bin_idx]['stars'].append({'x': cx, 'y': cy, 'mag': mag})
            
            # Select check stars: 2 per bin, including only the first red bin, skipping steps with < 2 stars
            self.limiting_mag_check_stars = []
            red_bin_included = False
            
            for b in sorted(bins_data.keys()):
                d = bins_data[b]
                if len(d['stars']) < 2:
                    continue
                    
                ratio = (d['detected'] / d['total']) * 100.0 if d['total'] > 0 else 0.0
                med_snr = np.median(d['snrs']) if d['snrs'] else 0.0
                
                # Determine category and color
                if ratio < 15.0 or med_snr < 5.0:
                    category = "red"
                    color_hex = "#ef4444"
                elif ratio < 50.0 or med_snr < 10.0:
                    category = "yellow"
                    color_hex = "#f59e0b"
                else:
                    category = "green"
                    color_hex = "#10b981"
                    
                if category == "red":
                    if red_bin_included:
                        continue
                    red_bin_included = True
                
                # Select 2 stars with spatial separation (at least 150px apart if possible)
                s1 = d['stars'][0]
                s2 = None
                for candidate in d['stars'][1:]:
                    dist = np.hypot(candidate['x'] - s1['x'], candidate['y'] - s1['y'])
                    if dist >= 150.0:
                        s2 = candidate
                        break
                if s2 is None:
                    # Fallback to the star furthest from s1
                    s2 = max(d['stars'][1:], key=lambda s: np.hypot(s['x'] - s1['x'], s['y'] - s1['y']))
                    
                self.limiting_mag_check_stars.append({'x': s1['x'], 'y': s1['y'], 'mag': s1['mag'], 'color': color_hex})
                self.limiting_mag_check_stars.append({'x': s2['x'], 'y': s2['y'], 'mag': s2['mag'], 'color': color_hex})
            
            self.render_canvas(is_dragging=False)
            log_win.destroy()
            
            report_win = tk.Toplevel(self.root)
            report_win.title("Limiting Magnitude Estimate")
            report_win.geometry("650x450")
            report_win.configure(bg="#1f2937")
            report_win.transient(self.root)
            
            lbl_title = tk.Label(report_win, text="Vizier Gaia DR3 Calibration Stars detection report:", bg="#1f2937", fg="white", font=("Segoe UI", 10, "bold"))
            lbl_title.pack(pady=10)
            
            txt_report = tk.Text(report_win, bg="#111827", insertbackground="white", font=("Consolas", 10), bd=0, padx=15, pady=15)
            
            # Configure Tkinter tags for color-coding
            txt_report.tag_configure("green", foreground="#10b981")
            txt_report.tag_configure("yellow", foreground="#f59e0b")
            txt_report.tag_configure("red", foreground="#ef4444")
            txt_report.tag_configure("white", foreground="white")
            
            txt_report.insert(tk.END, "=============================================================================\n", "white")
            txt_report.insert(tk.END, f"{'Magnitude Step':<16} | {'Total Stars':<12} | {'Detected':<10} | {'Ratio (%)':<10} | {'Median SNR':<10}\n", "white")
            txt_report.insert(tk.END, "=============================================================================\n", "white")
            
            for b in sorted(bins_data.keys()):
                d = bins_data[b]
                if d['total'] > 0:
                    ratio = (d['detected'] / d['total']) * 100.0
                    med_snr = np.median(d['snrs']) if d['snrs'] else 0.0
                    line_str = f"[{b:2d}.0 - {b+1:2d}.0)     | {d['total']:<12d} | {d['detected']:<10d} | {ratio:<10.1f} | {med_snr:<10.2f}\n"
                    
                    # Categorize row
                    if ratio < 15.0 or med_snr < 5.0:
                        tag = "red"
                    elif ratio < 50.0 or med_snr < 10.0:
                        tag = "yellow"
                    else:
                        tag = "green"
                else:
                    line_str = f"[{b:2d}.0 - {b+1:2d}.0)     | {'0':<12} | {'0':<10} | {'0.0%':<10} | {'N/A':<10}\n"
                    tag = "white"
                    
                txt_report.insert(tk.END, line_str, tag)
                
            txt_report.insert(tk.END, "=============================================================================\n", "white")
            txt_report.config(state="disabled")
            txt_report.pack(fill="both", expand=True, padx=15, pady=5)
            
            btn_close = tk.Button(report_win, text="Close", command=report_win.destroy, bg="#374151", fg="white", font=("Segoe UI", 9, "bold"), bd=0, padx=20, pady=6)
            btn_close.pack(pady=10)
            
        except Exception as ex:
            messagebox.showerror("Error", f"Failed to estimate limiting magnitude:\n{ex}", parent=self.root)
        finally:
            self.root.config(cursor="")

    def cleanup_caches_startup(self):
        import os
        for cache_dir in ["dss_cache", "panstarrs_cache"]:
            os.makedirs(cache_dir, exist_ok=True)
            try:
                files = []
                for entry in os.scandir(cache_dir):
                    if entry.is_file() and entry.name.endswith(".fits"):
                        files.append((entry.path, entry.stat().st_mtime))
                
                if len(files) > 300:
                    # Sort by modification time (oldest first)
                    files.sort(key=lambda x: x[1])
                    excess = len(files) - 300
                    for i in range(excess):
                        try:
                            os.remove(files[i][0])
                        except Exception:
                            pass
            except Exception:
                pass
from html.parser import HTMLParser

class TNSHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results = []
        self.current_row = {}
        self.in_tbody = False
        self.current_tag = None
        self.current_class = None
        self.current_data = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        self.current_tag = tag
        
        if tag == 'tbody':
            self.in_tbody = True
            
        if self.in_tbody:
            if tag == 'tr':
                cls = attrs_dict.get('class', '')
                if 'atreps' in cls or 'spectra' in cls:
                    return
                self.current_row = {}
            elif tag == 'td':
                self.current_class = attrs_dict.get('class', '')
                self.current_data = []

    def handle_data(self, data):
        if self.in_tbody and self.current_class:
            self.current_data.append(data)

    def handle_endtag(self, tag):
        if tag == 'tbody':
            self.in_tbody = False
        elif self.in_tbody:
            if tag == 'tr':
                if self.current_row and 'name' in self.current_row:
                    self.results.append(self.current_row)
                    self.current_row = {}
            elif tag == 'td':
                if self.current_class:
                    val = "".join(self.current_data).strip()
                    if 'cell-name' in self.current_class:
                        self.current_row['name'] = val
                    elif 'cell-ra' in self.current_class:
                        self.current_row['ra'] = val
                    elif 'cell-decl' in self.current_class:
                        self.current_row['dec'] = val
                    elif 'cell-objtype_name' in self.current_class:
                        self.current_row['type'] = val
                    elif 'cell-discoverymag' in self.current_class:
                        self.current_row['mag'] = val
                    elif 'cell-discoverydate' in self.current_class:
                        self.current_row['discovery_date'] = val
                self.current_class = None

    def query_tns_transients(self):
        if self.fits_data is None:
            messagebox.showerror("Error", "Load an image first.", parent=self.root)
            return
        if not self.wcs:
            messagebox.showerror("Error", "WCS solution is required. Please plate-solve the image first.", parent=self.root)
            return
            
        # Open Observation Time Dialog
        default_time_str = ""
        if 'DATE-OBS' in self.fits_header:
            default_time_str = str(self.fits_header['DATE-OBS'])
        elif getattr(self, 'observation_time', None) is not None:
            default_time_str = self.observation_time.isot
        else:
            from astropy.time import Time
            default_time_str = Time.now().isot
            
        dialog = tk.Toplevel(self.root)
        dialog.title("Observation Time")
        dialog.geometry("380x180")
        dialog.configure(bg=self.panel_color)
        dialog.transient(self.root)
        dialog.grab_set()
        
        lbl_info = tk.Label(dialog, text="Verify Observation Time (UTC ISO format):\nYYYY-MM-DDTHH:MM:SS", bg=self.panel_color, fg=self.text_color, font=("Segoe UI", 9, "bold"))
        lbl_info.pack(pady=15)
        
        ent_time = tk.Entry(dialog, bg=self.bg_color, fg=self.text_color, insertbackground="white", bd=0, highlightthickness=1, highlightbackground=self.control_bg, width=30)
        ent_time.insert(0, default_time_str)
        ent_time.pack(pady=5)
        
        def on_confirm():
            time_str = ent_time.get().strip()
            try:
                from astropy.time import Time
                self.observation_time = Time(time_str, format='isot')
                self.fits_header['DATE-OBS'] = self.observation_time.isot
                dialog.destroy()
                self.execute_tns_query()
            except Exception as ex:
                messagebox.showerror("Error", f"Invalid observation date format:\n{ex}", parent=dialog)
                
        btn_confirm = tk.Button(dialog, text="Confirm & Search", command=on_confirm, bg="#10b981", fg="white", font=("Segoe UI", 9, "bold"), bd=0, padx=15, pady=5)
        btn_confirm.pack(pady=10)

    def execute_tns_query(self):
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        from astropy.time import Time
        import datetime
        import urllib.request
        import urllib.parse
        import ssl
        import json
        
        # Open a nice log/debug window
        log_win = tk.Toplevel(self.root)
        log_win.title("TNS Query Debug Log")
        log_win.geometry("600x400")
        log_win.configure(bg=self.panel_color)
        log_win.transient(self.root)
        
        lbl = tk.Label(log_win, text="TNS Public Server Query Log", bg=self.panel_color, fg=self.text_color, font=("Segoe UI", 10, "bold"))
        lbl.pack(pady=5)
        
        txt_log = tk.Text(log_win, bg=self.bg_color, fg=self.text_color, wrap="word", font=("Consolas", 9), bd=0, padx=10, pady=10)
        txt_log.pack(fill="both", expand=True, padx=10, pady=5)
        
        def append_log(msg):
            txt_log.insert(tk.END, msg + "\n")
            txt_log.see(tk.END)
            self.root.update()
            
        self.root.config(cursor="watch")
        self.root.update()
        
        def run_query():
            try:
                h_orig, w_orig = self.debayered_cache.shape[:2]
                center_coord = self.wcs.pixel_to_world(w_orig / 2.0, h_orig / 2.0)
                
                corner_coord = self.wcs.pixel_to_world(0, 0)
                radius_deg = center_coord.separation(corner_coord).deg
                radius_arcmin = radius_deg * 60.0
                # Fink / TNS coordinates query uses arcminutes
                radius_arcmin = np.clip(radius_arcmin, 3.0, 300.0)
                
                append_log(f"[INFO] Image Center: RA={center_coord.ra.deg:.6f}, DEC={center_coord.dec.deg:.6f}")
                append_log(f"[INFO] Image Radius: {radius_arcmin:.2f} arcminutes")
                
                coord_sky = SkyCoord(ra=center_coord.ra.deg*u.deg, dec=center_coord.dec.deg*u.deg, frame='icrs')
                ra_hms = coord_sky.ra.to_string(unit=u.hour, sep=":", precision=2)
                dec_dms = coord_sky.dec.to_string(unit=u.degree, sep=":", precision=2, alwayssign=True)
                
                append_log(f"[INFO] Converted to sexagesimal: RA={ra_hms}, DEC={dec_dms}")
                
                obs_time = self.observation_time
                start_date = obs_time - datetime.timedelta(days=365)
                append_log(f"[INFO] Observation Time (Shot): {obs_time.iso}")
                append_log(f"[INFO] Filtering transients discovered after: {start_date.iso.split()[0]}")
                
                params = {
                    'ra': ra_hms,
                    'decl': dec_dms,
                    'radius': f"{radius_arcmin:.2f}",
                    'coords_unit': 'arcmin',
                    'op': 'Submit',
                    'form_id': 'SearchObject'
                }
                
                query_str = urllib.parse.urlencode(params)
                url = f"https://www.wis-tns.org/search?{query_str}"
                
                append_log(f"[HTTP GET] Querying public URL:\n{url}\n")
                
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                ssl_context = ssl._create_unverified_context()
                
                with urllib.request.urlopen(req, timeout=20, context=ssl_context) as response:
                    html_content = response.read().decode('utf-8')
                    
                append_log(f"[HTTP SUCCESS] Received response of length: {len(html_content)} bytes")
                
                parser = TNSHTMLParser()
                parser.feed(html_content)
                
                append_log(f"[PARSER] Raw objects parsed: {len(parser.results)}")
                
                self.transient_objects = []
                before_count = 0
                after_count = 0
                
                for obj in parser.results:
                    name = obj.get('name', '')
                    ra_str = obj.get('ra', '')
                    dec_str = obj.get('dec', '')
                    disc_date_str = obj.get('discovery_date', '')
                    mag_str = obj.get('mag', '')
                    obj_type = obj.get('type', '')
                    
                    if not name.startswith(('SN ', 'AT ')) or not ra_str or not dec_str or not disc_date_str:
                        continue
                        
                    append_log(f"[FOUND] {name} ({obj_type}) Mag: {mag_str} Disc: {disc_date_str}")
                    
                    try:
                        obj_coord = SkyCoord(ra=ra_str, dec=dec_str, unit=(u.hourangle, u.deg), frame='icrs')
                        obj_ra = obj_coord.ra.deg
                        obj_dec = obj_coord.dec.deg
                        
                        disc_time = Time(disc_date_str.replace(' ', 'T'), format='isot')
                        
                        if disc_time < start_date:
                            append_log(f"   -> Skipped: discovered before start window (older than 12 months)")
                            continue
                            
                        if disc_time > obs_time:
                            color = "#ef4444"  # Red
                            after_count += 1
                            append_log(f"   -> Match: Discovered AFTER shot (Red)")
                        else:
                            color = "#10b981"  # Green
                            before_count += 1
                            append_log(f"   -> Match: Discovered BEFORE/ON shot (Green)")
                            
                        self.transient_objects.append({
                            'name': name,
                            'ra': obj_ra,
                            'dec': obj_dec,
                            'discovery_date': disc_date_str,
                            'mag': mag_str,
                            'type': obj_type,
                            'color': color
                        })
                    except Exception as parse_ex:
                        append_log(f"   -> Error parsing coordinates/time: {parse_ex}")
                        continue
                        
                append_log(f"\n[SUMMARY] Stored {len(self.transient_objects)} transients inside image field.")
                self.render_canvas(is_dragging=False)
                
                messagebox.showinfo(
                    "TNS Search Success",
                    f"Query completed successfully using TNS public database!\n\n"
                    f"Transients found: {len(self.transient_objects)}\n"
                    f"- Discovered before/on shot (Green): {before_count}\n"
                    f"- Discovered after shot (Red): {after_count}",
                    parent=log_win
                )
            except Exception as e:
                append_log(f"\n[ERROR] Query failed: {e}")
                messagebox.showerror("TNS Search Error", f"Failed to retrieve transients from TNS:\n{e}", parent=log_win)
            finally:
                self.root.config(cursor="")
                
        log_win.after(200, run_query)

    def reset_sliders(self):
        self.slider_red_offset.set(0)
        self.slider_green_offset.set(0)
        self.slider_blue_offset.set(0)
        self.slider_brightness.set(0)
        self.slider_contrast.set(0)
        self.slider_smooth.set(0)
        self.process_and_update(is_dragging=False)
        
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
