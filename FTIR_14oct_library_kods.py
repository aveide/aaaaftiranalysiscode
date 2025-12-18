#!/usr/bin/env python3
# FTIR Ultimate Studio: Import + Crop + Simulate + Identify
# FIXED: Table Header Crash resolved.

import os, sys, math, re, glob
import numpy as np

# ---- Matplotlib for background tasks ----
import matplotlib
matplotlib.use("Agg") 

# ---- GUI Framework (PySide6 + PyQtGraph) ----
try:
    import pyqtgraph as pg
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QAction, QPalette, QColor
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QLabel, QComboBox, QPushButton, QTableWidget, QTableWidgetItem,
        QSplitter, QFileDialog, QSpinBox, QDoubleSpinBox, QCheckBox, 
        QScrollArea, QMessageBox, QTabWidget, QGroupBox, QHeaderView
    )
    pg.setConfigOption('useOpenGL', False)
    pg.setConfigOption('antialias', True)
    pg.setConfigOption('background', '#1e1e1e')
    pg.setConfigOption('foreground', '#d0d0d0')
except ImportError:
    print("CRITICAL ERROR: Missing libraries.")
    print("Please run: pip install PySide6 pyqtgraph numpy scipy")
    sys.exit(1)

# ---- Optional Science ----
try:
    from scipy.signal import savgol_filter
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ==============================================================================
#                               CORE MATHEMATICS
# ==============================================================================

def apodize(x: np.ndarray, kind="Happ-Genzel") -> np.ndarray:
    n = x.size
    if n < 2: return x
    if kind == "Happ-Genzel":
        k = np.arange(n)
        t = (k - (n - 1) / 2.0) / ((n - 1) / 2.0)
        w = 0.54 + 0.46 * np.cos(np.pi * t)
    elif kind == "Blackman-Harris":
        k = np.arange(n)
        t = 2.0 * np.pi * k / (n - 1)
        w = 0.35875 - 0.48829*np.cos(t) + 0.14128*np.cos(2*t) - 0.01168*np.cos(3*t)
    elif kind == "Kaiser":
        w = np.kaiser(n, 6.0)
    else: # Boxcar
        w = np.ones(n)
    return x * w

def baseline_als(y, lam=10000, p=0.001, niter=10):
    if not HAS_SCIPY: return np.zeros_like(y)
    L = len(y)
    D = sparse.diags([1,-2,1],[0,1,2], shape=(L-2,L))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.diags(w, 0)
        Z = W + lam * (D.T @ D)
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

def fft_pipeline(x_data, y_data, zf_factor=2, apod="Happ-Genzel"):
    """ Converts Interferogram -> Spectrum """
    if len(y_data) < 2: return np.array([]), np.array([])

    # 1. Remove DC
    y_proc = y_data - np.mean(y_data)
    
    # 2. Apodize
    y_proc = apodize(y_proc, kind=apod)
    
    # 3. Zero Fill
    n = y_proc.size
    n_fft = 1 << int(math.ceil(math.log2(n * zf_factor)))
    if n_fft < n: n_fft = n
    y_proc = np.pad(y_proc, (0, n_fft - n), mode="constant")
    
    # 4. Step Size
    if len(x_data) > 1:
        dx = float(np.mean(np.diff(x_data)))
        if dx <= 1e-9: dx = 6.328e-5 # Fallback to HeNe
    else:
        dx = 6.328e-5

    # 5. FFT
    F = np.fft.rfft(y_proc)
    freqs = np.fft.rfftfreq(n_fft, d=dx)
    spectrum = np.abs(F)

    return freqs.astype(np.float32), spectrum.astype(np.float32)

def simulate_interferogram(peaks):
    dx = 6.328e-5 * 0.5 # typical step
    N = 8192
    freqs = np.fft.rfftfreq(N, d=dx)
    S = np.zeros_like(freqs)
    
    for p in peaks:
        pos = p['pos']; amp = p['amp']; width = p['width']
        gamma = width / 2.0
        # Lorentzian
        S += amp * (gamma**2) / ((freqs - pos)**2 + gamma**2)
        
    I = np.fft.irfft(S, n=N)
    I = np.fft.fftshift(I)
    x = (np.arange(N) - N//2) * dx
    return x, I, freqs, S

# ==============================================================================
#                               DATA PARSING
# ==============================================================================

def _read_jdx_internal(path):
    # Minimal JDX parser
    try:
        with open(path, 'r', errors='ignore') as f: content = f.read()
        lines = content.splitlines()
        
        xfac, yfac = 1.0, 1.0
        start_idx = 0
        
        for i, ln in enumerate(lines):
            if "##XYDATA" in ln: start_idx = i+1; break
            if "##XFACTOR=" in ln: xfac = float(ln.split("=")[1])
            if "##YFACTOR=" in ln: yfac = float(ln.split("=")[1])
            
        xs, ys = [], []
        for ln in lines[start_idx:]:
            if ln.startswith("##"): break
            # Remove squeezing characters if simple
            parts = ln.replace(',', ' ').replace('+', ' +').replace('-', ' -').split()
            try:
                vals = [float(p) for p in parts]
                if len(vals) > 0:
                    xs.append(vals[0]*xfac)
                    ys.append(vals[1]*yfac)
            except: pass
            
        if not xs: return None, None, "JDX Parse Failed"
        return np.array(xs), np.array(ys), "JDX"
    except Exception as e: return None, None, str(e)

def robust_load(path):
    try:
        if path.lower().endswith(('jdx', 'dx')):
            return _read_jdx_internal(path)

        # Helper to skip headers
        with open(path, 'r', errors='ignore') as f:
            lines = [f.readline() for _ in range(10)]
        skip = 0
        for i, line in enumerate(lines):
            if re.search(r'[a-zA-Z]', line) and not any(c in line for c in ['e','E','+','-']):
                skip = i + 1
        
        try:
            data = np.loadtxt(path, delimiter=",", skiprows=skip, comments="#")
        except:
            data = np.loadtxt(path, skiprows=skip, comments="#")
        
        if data.ndim == 1:
            # 1 Column
            y = data
            dx = 6.328e-5 
            x = (np.arange(len(y)) - len(y)//2) * dx
            return x, y, "1-Col Raw"
        elif data.ndim == 2 and data.shape[1] >= 2:
            # 2 Columns
            x = data[:,0]
            y = data[:,1]
            if x[0] > x[-1]: # Sort ascending
                idx = np.argsort(x)
                x = x[idx]
                y = y[idx]
            return x, y, "2-Col CSV"
            
    except Exception as e:
        return None, None, str(e)
    return None, None, "Unknown Error"

# ==============================================================================
#                               GUI
# ==============================================================================

class FTIRMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FTIR Ultimate Studio")
        self.resize(1400, 900)
        self.setup_style()

        self.full_x = None; self.full_y = None
        self.curr_spec_x = None; self.curr_spec_y = None
        self.library = []

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        self.init_analysis_tab()
        self.init_simulation_tab()

    def setup_style(self):
        p = self.palette()
        p.setColor(QPalette.Window, QColor(30,30,30))
        p.setColor(QPalette.WindowText, QColor(220,220,220))
        p.setColor(QPalette.Base, QColor(15,15,15))
        p.setColor(QPalette.AlternateBase, QColor(40,40,40))
        p.setColor(QPalette.Text, QColor(220,220,220))
        p.setColor(QPalette.Button, QColor(50,50,50))
        p.setColor(QPalette.ButtonText, QColor(220,220,220))
        p.setColor(QPalette.Highlight, QColor(0, 120, 215))
        self.setPalette(p)

    def init_analysis_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # Left Controls
        panel = QWidget(); panel.setMaximumWidth(350)
        vbox = QVBoxLayout(panel)
        
        # 1. Load
        gb_load = QGroupBox("1. Data Source")
        gl = QVBoxLayout(gb_load)
        btn_open = QPushButton("📂 Open File")
        btn_open.setStyleSheet("background-color: #0078d7; color: white; padding: 6px; font-weight: bold;")
        btn_open.clicked.connect(self.load_user_file)
        self.lbl_info = QLabel("No Data")
        gl.addWidget(btn_open)
        gl.addWidget(self.lbl_info)
        vbox.addWidget(gb_load)
        
        # 2. Process
        gb_proc = QGroupBox("2. FFT Settings")
        gp = QGridLayout(gb_proc)
        self.cb_apod = QComboBox(); self.cb_apod.addItems(["Happ-Genzel", "Blackman-Harris", "Boxcar"])
        self.cb_apod.currentIndexChanged.connect(self.update_fft)
        self.sb_zf = QSpinBox(); self.sb_zf.setRange(1, 8); self.sb_zf.setValue(2); self.sb_zf.setPrefix("ZF: ")
        self.sb_zf.valueChanged.connect(self.update_fft)
        gp.addWidget(QLabel("Window:"), 0, 0)
        gp.addWidget(self.cb_apod, 0, 1)
        gp.addWidget(self.sb_zf, 1, 0, 1, 2)
        vbox.addWidget(gb_proc)
        
        # 3. ID
        gb_id = QGroupBox("3. Identify")
        gi = QVBoxLayout(gb_id)
        btn_lib = QPushButton("Load Library Folder")
        btn_lib.clicked.connect(self.load_library_folder)
        
        self.chk_base = QCheckBox("Baseline Corr"); self.chk_base.setChecked(True)
        self.chk_deriv = QCheckBox("1st Deriv Match"); self.chk_deriv.setChecked(True)
        self.chk_base.toggled.connect(self.run_ident)
        self.chk_deriv.toggled.connect(self.run_ident)
        
        self.table_res = QTableWidget(0, 2)
        self.table_res.setHorizontalHeaderLabels(["Compound", "Score"])
        self.table_res.verticalHeader().setVisible(False)
        # FIX: Changed header() to horizontalHeader() and used standard enum
        self.table_res.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table_res.itemSelectionChanged.connect(self.show_library_match)
        
        gi.addWidget(btn_lib)
        gi.addWidget(self.chk_base)
        gi.addWidget(self.chk_deriv)
        gi.addWidget(self.table_res)
        vbox.addWidget(gb_id)
        
        layout.addWidget(panel)
        
        # Right Plots
        split = QSplitter(Qt.Vertical)
        
        self.plot_crop = pg.PlotWidget(title="<span style='color:#fff'><b>Interferogram</b> (Drag Blue Area to Crop)</span>")
        self.plot_crop.showGrid(True, True, 0.2)
        self.crop_region = pg.LinearRegionItem()
        self.crop_region.setBrush(pg.mkBrush(0, 100, 255, 50))
        self.crop_region.sigRegionChanged.connect(self.update_fft)
        self.plot_crop.addItem(self.crop_region)
        self.crop_region.hide()
        self.curve_raw = self.plot_crop.plot(pen='y')

        self.plot_res = pg.PlotWidget(title="<span style='color:#fff'><b>Spectrum</b></span>")
        self.plot_res.setLabel('bottom', 'Wavenumber', 'cm⁻¹')
        self.plot_res.showGrid(True, True, 0.2)
        self.curve_spec = self.plot_res.plot(pen='c')
        self.curve_match = self.plot_res.plot(pen=pg.mkPen('m', width=2, style=Qt.DashLine))
        
        split.addWidget(self.plot_crop)
        split.addWidget(self.plot_res)
        layout.addWidget(split)
        
        self.tabs.addTab(tab, "Analyzer")

    def init_simulation_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        c = QWidget(); c.setMaximumWidth(250); vl = QVBoxLayout(c)
        self.sim_tbl = QTableWidget(0, 3)
        self.sim_tbl.setHorizontalHeaderLabels(["Pos","Amp","Wid"])
        
        b_add = QPushButton("Add Peak"); b_add.clicked.connect(lambda: self.sim_add_row(1500,1,20))
        b_run = QPushButton("Simulate"); b_run.clicked.connect(self.run_simulation)
        b_run.setStyleSheet("background-color: #2da44e; color: white; font-weight: bold;")
        
        vl.addWidget(QLabel("Peaks:"))
        vl.addWidget(self.sim_tbl)
        vl.addWidget(b_add)
        vl.addWidget(b_run)
        vl.addStretch()
        
        plots = QSplitter(Qt.Vertical)
        self.sim_p1 = pg.PlotWidget(title="Simulated Interferogram")
        self.sim_p2 = pg.PlotWidget(title="Simulated Spectrum")
        plots.addWidget(self.sim_p1)
        plots.addWidget(self.sim_p2)
        
        layout.addWidget(c)
        layout.addWidget(plots)
        self.tabs.addTab(tab, "Simulator")
        self.sim_add_row(1000, 1, 10)

    # --- LOGIC ---

    def load_user_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open", "", "Data (*.txt *.csv *.dat *.jdx *.dx);;All (*)")
        if not path: return
        
        x, y, msg = robust_load(path)
        if x is None:
            QMessageBox.critical(self, "Error", msg)
            return
            
        self.full_x = x
        self.full_y = y
        self.lbl_info.setText(f"{os.path.basename(path)}\n{len(x)} pts | {msg}")
        
        # Check if already spectrum
        if np.max(np.abs(x)) > 500:
            # Already Spectrum
            self.crop_region.hide()
            self.plot_crop.setTitle("Raw Spectrum (No FFT)")
            self.curve_raw.setData(x, y)
            self.curr_spec_x, self.curr_spec_y = x, y
            self.curve_spec.setData(x, y)
            self.run_ident()
        else:
            # Interferogram
            self.plot_crop.setTitle("Interferogram (Crop Enabled)")
            self.curve_raw.setData(x, y)
            
            # Auto-crop 90% center
            span = x[-1] - x[0]
            ctr = (x[-1] + x[0]) / 2
            self.crop_region.setRegion([ctr - span*0.45, ctr + span*0.45])
            self.crop_region.show()
            self.update_fft()

    def update_fft(self):
        if self.full_x is None or not self.crop_region.isVisible(): return
        
        mn, mx = self.crop_region.getRegion()
        mask = (self.full_x >= mn) & (self.full_x <= mx)
        x_sub = self.full_x[mask]
        y_sub = self.full_y[mask]
        
        if len(x_sub) < 10: return
        
        freq, spec = fft_pipeline(x_sub, y_sub, self.sb_zf.value(), self.cb_apod.currentText())
        
        # Filter range
        mask_f = (freq > 400) & (freq < 10000)
        self.curr_spec_x = freq[mask_f]
        self.curr_spec_y = spec[mask_f]
        
        self.curve_spec.setData(self.curr_spec_x, self.curr_spec_y)
        if self.library: self.run_ident()

    def load_library_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Library")
        if not d: return
        
        self.library = []
        for fp in glob.glob(os.path.join(d, "*")):
            x, y, m = robust_load(fp)
            if x is not None and np.max(np.abs(x)) > 500: # Ensure it is a spectrum
                self.library.append({'name': os.path.basename(fp), 'x': x, 'y': y})
        
        QMessageBox.information(self, "Loaded", f"Imported {len(self.library)} spectra.")
        if self.curr_spec_y is not None: self.run_ident()

    def run_ident(self):
        if self.curr_spec_y is None or not self.library: return
        
        # Preprocess Sample
        s_y = self.curr_spec_y
        if self.chk_base.isChecked() and HAS_SCIPY:
            b = baseline_als(s_y.astype(np.float64))
            s_y = np.maximum(s_y - b, 0)
        
        # Normalize
        s_norm = s_y / (np.max(s_y) + 1e-9)
        s_proc = savgol_filter(s_norm, 11, 2, deriv=1) if (self.chk_deriv.isChecked() and HAS_SCIPY) else s_norm
        
        hits = []
        for entry in self.library:
            # Resample lib to sample
            l_y = np.interp(self.curr_spec_x, entry['x'], entry['y'], left=0, right=0)
            l_norm = l_y / (np.max(l_y) + 1e-9)
            l_proc = savgol_filter(l_norm, 11, 2, deriv=1) if (self.chk_deriv.isChecked() and HAS_SCIPY) else l_norm
            
            score = np.dot(s_proc, l_proc)
            hits.append((entry['name'], score, l_norm))
            
        hits.sort(key=lambda k: k[1], reverse=True)
        
        self.table_res.setRowCount(0)
        for name, score, raw in hits[:15]:
            r = self.table_res.rowCount()
            self.table_res.insertRow(r)
            self.table_res.setItem(r, 0, QTableWidgetItem(name))
            self.table_res.setItem(r, 1, QTableWidgetItem(f"{score:.4f}"))
            self.table_res.item(r, 0).setData(Qt.UserRole, raw)

    def show_library_match(self):
        row = self.table_res.currentRow()
        if row < 0: return
        raw = self.table_res.item(row, 0).data(Qt.UserRole)
        scale = np.max(self.curr_spec_y)
        self.curve_match.setData(self.curr_spec_x, raw * scale)

    def sim_add_row(self, p, a, w):
        r = self.sim_tbl.rowCount()
        self.sim_tbl.insertRow(r)
        self.sim_tbl.setItem(r, 0, QTableWidgetItem(str(p)))
        self.sim_tbl.setItem(r, 1, QTableWidgetItem(str(a)))
        self.sim_tbl.setItem(r, 2, QTableWidgetItem(str(w)))

    def run_simulation(self):
        peaks = []
        for r in range(self.sim_tbl.rowCount()):
            try:
                p = float(self.sim_tbl.item(r,0).text())
                a = float(self.sim_tbl.item(r,1).text())
                w = float(self.sim_tbl.item(r,2).text())
                peaks.append({'pos':p, 'amp':a, 'width':w})
            except: pass
        
        x, I, f, s = simulate_interferogram(peaks)
        self.sim_p1.plot(x, I, clear=True, pen='y')
        
        mask = (f > 400) & (f < 10000)
        self.sim_p2.plot(f[mask], s[mask], clear=True, pen='c', fillLevel=0, brush=(0,255,255,50))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = FTIRMainWindow()
    w.show()
    sys.exit(app.exec())