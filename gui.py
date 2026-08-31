"""
V-D Splitter GUI.

VOICE-DENOISE desktop shell inherited from the M-A Splitter interface work.
"""

from __future__ import annotations

import ctypes
import json
import os
import queue
import random
import subprocess
import sys
import threading
from pathlib import Path

import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk


HERE = Path(__file__).resolve().parent
PIPELINE = HERE / "pipeline.py"
SETTINGS = HERE / "settings.json"
FONT_FILE = HERE / "assets" / "fonts" / "Oxanium.ttf"

VIDEO_TYPES = [
    ("Audio/video", "*.mp4 *.mkv *.mov *.avi *.webm *.m4v *.wmv *.mpg *.mpeg *.ts *.mts *.m2ts *.flv *.3gp *.ogv *.wav *.mp3 *.flac *.m4a *.aac *.ogg *.wma *.aiff *.aif *.opus *.alac *.amr"),
    ("Video", "*.mp4 *.mkv *.mov *.avi *.webm *.m4v *.wmv *.mpg *.mpeg *.ts *.mts *.m2ts *.flv *.3gp *.ogv"),
    ("Audio", "*.wav *.mp3 *.flac *.m4a *.aac *.ogg *.wma *.aiff *.aif *.opus *.alac *.amr"),
    ("All files", "*.*"),
]

COLORS = {
    "bg": "#05070A",
    "panel": "#0D1117",
    "panel_2": "#131820",
    "metal": "#343B45",
    "metal_dark": "#222831",
    "ridge": "#303844",
    "ridge_hi": "#667385",
    "entry_edge": "#465463",
    "entry": "#090D13",
    "text": "#AFC3CF",
    "text_hi": "#C7D8E0",
    "muted": "#748899",
    "cyan": "#22E6FF",
    "magenta": "#FF2D95",
    "green": "#74FFB1",
    "log": "#060A10",
}


class Tooltip:
    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self._tip: tk.Toplevel | None = None
        self._after: str | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None) -> None:
        self._after = self.widget.after(450, self._show)

    def _show(self) -> None:
        if self._tip:
            return
        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{self.widget.winfo_rootx()+16}+{self.widget.winfo_rooty()+34}")
        tk.Label(
            self._tip,
            text=self.text,
            bg="#121820",
            fg=COLORS["text_hi"],
            relief="solid",
            borderwidth=1,
            wraplength=360,
            padx=8,
            pady=5,
            font=("Segoe UI", 9),
        ).pack()

    def _hide(self, _event=None) -> None:
        if self._after:
            self.widget.after_cancel(self._after)
            self._after = None
        if self._tip:
            self._tip.destroy()
            self._tip = None


def tip(widget: tk.Widget, text: str) -> tk.Widget:
    Tooltip(widget, text)
    return widget


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("V-D Splitter")
        root.geometry("1120x690")
        root.minsize(960, 610)
        root.overrideredirect(True)
        root.configure(bg=COLORS["bg"])

        self.proc: subprocess.Popen | None = None
        self.q: "queue.Queue[str]" = queue.Queue()
        self.last_result_dir: Path | None = None
        self._drag_xy: tuple[int, int] | None = None
        self._maximized = False
        self._texture_cache: dict[tuple[int, int, str], tk.PhotoImage] = {}
        self.display_font = self._pick_display_font()

        cfg = self._load_settings()
        self.in_path = tk.StringVar(value=cfg.get("in_path", ""))
        self.ref_path = tk.StringVar(value=cfg.get("ref_path", ""))
        self.profile_path = tk.StringVar(value=cfg.get("profile_path", ""))
        self.model_path = tk.StringVar(value=cfg.get("model_path", ""))
        self.out_path = tk.StringVar(value=cfg.get("out_path", str(HERE / "output")))
        self.device = tk.StringVar(value=cfg.get("device", "auto"))
        self.model = tk.StringVar(value=cfg.get("model", "htdemucs_ft"))
        self.segment = tk.IntVar(value=cfg.get("segment", 7))
        self.denoise_model = tk.StringVar(value=cfg.get("denoise_model", "dns64"))
        self.denoise_dry = tk.IntVar(value=cfg.get("denoise_dry", 0))
        self.polish_preset = tk.StringVar(value=cfg.get("polish_preset", "speech"))
        self.compressor = tk.BooleanVar(value=cfg.get("compressor", True))
        self.deesser = tk.BooleanVar(value=cfg.get("deesser", True))
        self.loudness = tk.BooleanVar(value=cfg.get("loudness", True))
        self.target_lufs = tk.IntVar(value=cfg.get("target_lufs", -16))
        self.peak_ceiling = tk.IntVar(value=cfg.get("peak_ceiling", 95))
        self.keep_bg = tk.BooleanVar(value=cfg.get("keep_bg", True))
        self.status = tk.StringVar(value="READY")

        self._configure_styles()
        self._build()
        root.after(100, self._drain_log)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _pick_display_font(self) -> str:
        if FONT_FILE.is_file() and os.name == "nt":
            try:
                ctypes.windll.gdi32.AddFontResourceExW(str(FONT_FILE), 0x10, 0)
            except Exception:
                pass
        fonts = {name.lower(): name for name in tkfont.families(self.root)}
        for candidate in ("Oxanium", "ST MicroSquare Ex", "Bahnschrift SemiBold", "Bahnschrift"):
            if candidate.lower() in fonts:
                return fonts[candidate.lower()]
        return "Bahnschrift SemiBold"

    def _load_settings(self) -> dict:
        try:
            return json.loads(SETTINGS.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_settings(self) -> None:
        data = {
            "in_path": self.in_path.get(),
            "ref_path": self.ref_path.get(),
            "profile_path": self.profile_path.get(),
            "model_path": self.model_path.get(),
            "out_path": self.out_path.get(),
            "device": self.device.get(),
            "model": self.model.get(),
            "segment": int(self.segment.get()),
            "denoise_model": self.denoise_model.get(),
            "denoise_dry": int(self.denoise_dry.get()),
            "polish_preset": self.polish_preset.get(),
            "compressor": bool(self.compressor.get()),
            "deesser": bool(self.deesser.get()),
            "loudness": bool(self.loudness.get()),
            "target_lufs": int(self.target_lufs.get()),
            "peak_ceiling": int(self.peak_ceiling.get()),
            "keep_bg": bool(self.keep_bg.get()),
        }
        try:
            SETTINGS.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _configure_styles(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Shell.TFrame", background=COLORS["bg"])
        style.configure("Panel.TFrame", background=COLORS["panel"])
        style.configure("Card.TFrame", background=COLORS["panel_2"])
        style.configure("TLabel", background=COLORS["bg"], foreground=COLORS["text"])
        style.configure("Muted.TLabel", background=COLORS["bg"], foreground=COLORS["muted"])
        style.configure("Panel.TLabel", background=COLORS["panel"], foreground=COLORS["text_hi"])
        style.configure(
            "Cyber.Horizontal.TProgressbar",
            troughcolor=COLORS["entry"],
            background=COLORS["magenta"],
            bordercolor=COLORS["ridge"],
        )

    def _brushed_texture(self, width: int, height: int, base: str) -> tk.PhotoImage:
        key = (min(width, 1600), min(height, 900), base)
        if key in self._texture_cache:
            return self._texture_cache[key]
        rng = random.Random(width * 193 + height * 881 + sum(ord(c) for c in base))
        br, bg, bb = tuple(int(base.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        img = tk.PhotoImage(width=width, height=height)
        rows = []
        for y in range(height):
            row_noise = rng.randint(-20, 18)
            scratch = rng.choice((0, 0, 0, -38, 32))
            row = []
            for x in range(width):
                cx = abs((x / max(1, width - 1)) - 0.5)
                glow = max(0, 1 - cx * 2.1) * 22
                d = row_noise + rng.randint(-8, 8) + scratch + glow - cx * 40
                row.append("#{0:02x}{1:02x}{2:02x}".format(
                    max(0, min(255, int(br + d))),
                    max(0, min(255, int(bg + d))),
                    max(0, min(255, int(bb + d))),
                ))
            rows.append("{" + " ".join(row) + "}")
        img.put(" ".join(rows))
        self._texture_cache[key] = img
        return img

    def _paint_metal(self, canvas: tk.Canvas, width: int, height: int, base: str) -> None:
        canvas.delete("texture")
        img = self._brushed_texture(max(1, width), max(1, height), base)
        canvas.create_image(0, 0, anchor="nw", image=img, tags="texture")
        canvas._img = img
        canvas.tag_lower("texture")

    def _button(self, parent: tk.Widget, text: str, command, accent: str | None = None) -> tk.Button:
        bg = accent or "#111720"
        fg = "#031016" if accent == COLORS["cyan"] else COLORS["text_hi"]
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground="#1F2834" if not accent else accent,
            activeforeground=COLORS["cyan"] if not accent else fg,
            relief="flat",
            bd=0,
            padx=12,
            pady=6,
            highlightthickness=1,
            highlightbackground=COLORS["entry_edge"],
            font=(self.display_font if accent else "Segoe UI Semibold", 9),
            cursor="hand2",
        )

    def _entry(self, parent: tk.Widget, variable: tk.StringVar, width: int) -> tk.Entry:
        return tk.Entry(
            parent,
            textvariable=variable,
            width=width,
            bg=COLORS["entry"],
            fg=COLORS["text"],
            insertbackground=COLORS["cyan"],
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=COLORS["entry_edge"],
            highlightcolor=COLORS["cyan"],
            font=("Segoe UI Semibold", 9),
        )

    def _combo(self, parent: tk.Widget, variable: tk.StringVar, values: list[str], width: int) -> tk.OptionMenu:
        menu = tk.OptionMenu(parent, variable, *values)
        menu.configure(
            width=width,
            anchor="w",
            bg=COLORS["entry"],
            fg=COLORS["text"],
            activebackground="#18222B",
            activeforeground=COLORS["cyan"],
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=COLORS["entry_edge"],
            indicatoron=False,
            font=("Segoe UI Semibold", 9),
        )
        menu["menu"].configure(bg=COLORS["panel_2"], fg=COLORS["text_hi"], activebackground="#183F4A")
        return menu

    def _spin(self, parent: tk.Widget, variable: tk.IntVar, from_: int, to: int) -> tk.Spinbox:
        return tk.Spinbox(
            parent,
            from_=from_,
            to=to,
            textvariable=variable,
            width=6,
            bg=COLORS["entry"],
            fg=COLORS["text"],
            buttonbackground=COLORS["metal"],
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=COLORS["entry_edge"],
            font=("Segoe UI Semibold", 9),
        )

    def _check(self, parent: tk.Widget, text: str, variable: tk.BooleanVar) -> tk.Checkbutton:
        return tk.Checkbutton(
            parent,
            text=text,
            variable=variable,
            indicatoron=False,
            bg=COLORS["panel"],
            fg=COLORS["text_hi"],
            activebackground="#18222B",
            activeforeground=COLORS["cyan"],
            selectcolor=COLORS["entry"],
            relief="flat",
            bd=0,
            padx=8,
            pady=3,
            highlightthickness=1,
            highlightbackground=COLORS["ridge"],
            font=("Segoe UI Semibold", 8),
        )

    def _section(self, parent: tk.Widget, title: str) -> tk.Frame:
        outer = tk.Frame(parent, bg=COLORS["panel"], highlightthickness=1, highlightbackground=COLORS["ridge_hi"])
        outer.pack(fill="x", pady=8)
        titlebar = tk.Canvas(outer, height=27, bg=COLORS["metal"], bd=0, highlightthickness=0)
        titlebar.pack(fill="x")
        titlebar.bind("<Configure>", lambda e: self._paint_metal(titlebar, e.width, e.height, COLORS["metal"]))
        titlebar.create_text(9, 14, text=title, anchor="w", fill=COLORS["cyan"], font=(self.display_font, 9))
        body = tk.Frame(outer, bg=COLORS["panel"], padx=12, pady=12)
        body.pack(fill="x")
        for i in range(6):
            body.columnconfigure(i, weight=1 if i in (1, 3, 5) else 0)
        return body

    def _build(self) -> None:
        shell = tk.Frame(self.root, bg=COLORS["bg"], highlightthickness=1, highlightbackground=COLORS["cyan"])
        shell.pack(fill="both", expand=True)
        self._titlebar(shell)
        self._hero(shell)

        body = ttk.Frame(shell, style="Shell.TFrame")
        body.pack(fill="both", expand=True, padx=14, pady=(6, 14))
        body.columnconfigure(0, minsize=420)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left = ttk.Frame(body, style="Shell.TFrame")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self._source_section(left)
        self._engine_section(left)
        self._polish_section(left)
        self._run_section(left)

        right = ttk.Frame(body, style="Shell.TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=3)
        right.rowconfigure(3, weight=1)
        right.columnconfigure(0, weight=1)
        ttk.Label(right, text="PROCESS LOG", style="Muted.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 5))
        log_frame = ttk.Frame(right, style="Card.TFrame", padding=1)
        log_frame.grid(row=1, column=0, sticky="nsew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log = tk.Text(log_frame, bg=COLORS["log"], fg="#AEEBF2", insertbackground=COLORS["cyan"],
                           relief="flat", bd=0, padx=10, pady=10, font=("Cascadia Mono", 9))
        self.log.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        scroll.grid(row=0, column=1, sticky="ns")

        ttk.Label(right, text="OUTPUT FILES", style="Muted.TLabel").grid(row=2, column=0, sticky="w", pady=(10, 5))
        result_frame = ttk.Frame(right, style="Card.TFrame", padding=8)
        result_frame.grid(row=3, column=0, sticky="nsew")
        result_frame.columnconfigure(0, weight=1)
        self.results = tk.Listbox(result_frame, bg=COLORS["entry"], fg=COLORS["text"],
                                  selectbackground="#183F4A", selectforeground=COLORS["cyan"],
                                  relief="flat", highlightthickness=1, highlightbackground=COLORS["ridge"])
        self.results.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.results.bind("<Double-Button-1>", lambda _e: self._open_selected())
        self._button(result_frame, "OPEN DIR", self._open_result_dir).grid(row=0, column=1, sticky="n")

    def _titlebar(self, parent: tk.Widget) -> None:
        bar = tk.Canvas(parent, height=34, bg=COLORS["metal_dark"], bd=0, highlightthickness=0)
        bar.pack(fill="x")
        bar.bind("<Configure>", lambda e: self._paint_metal(bar, e.width, e.height, COLORS["metal_dark"]))
        bar.bind("<ButtonPress-1>", self._drag_start)
        bar.bind("<B1-Motion>", self._drag_move)
        bar.bind("<Double-Button-1>", lambda _e: self._toggle_max())
        bar.create_text(14, 17, text="V-D SPLITTER // VOICE-DENOISE", anchor="w",
                        fill=COLORS["cyan"], font=(self.display_font, 10))
        controls = tk.Frame(bar, bg=COLORS["metal_dark"])
        controls.place(relx=1.0, x=-6, y=5, anchor="ne")
        self._button(controls, "_", self._minimize).pack(side="left", padx=2)
        self._button(controls, "[]", self._toggle_max).pack(side="left", padx=2)
        close = self._button(controls, "X", self._on_close)
        close.configure(bg="#23111A", activebackground=COLORS["magenta"])
        close.pack(side="left", padx=2)

    def _hero(self, parent: tk.Widget) -> None:
        hero = tk.Canvas(parent, height=88, bg=COLORS["bg"], bd=0, highlightthickness=0)
        hero.pack(fill="x")
        hero.bind("<Configure>", lambda e: self._paint_metal(hero, e.width, e.height, COLORS["bg"]))
        hero.create_text(24, 22, text="V-D SPLITTER", anchor="nw", fill=COLORS["text_hi"],
                         font=(self.display_font, 28))
        hero.create_text(26, 58, text="VOICE-DENOISE / VIDEO AUDIO EXTRACTION / REFERENCE MATCH",
                         anchor="nw", fill=COLORS["muted"], font=("Consolas", 9))
        hero.create_line(24, 78, 270, 78, fill=COLORS["cyan"], width=2)
        hero.create_line(274, 78, 410, 78, fill=COLORS["magenta"], width=2)
        tk.Label(hero, textvariable=self.status, bg=COLORS["bg"], fg=COLORS["green"],
                 font=(self.display_font, 9)).place(relx=1.0, x=-34, y=31, anchor="ne")

    def _source_section(self, parent: tk.Widget) -> None:
        sec = self._section(parent, "SOURCE / OUTPUT")
        tk.Label(sec, text="Input", bg=COLORS["panel"], fg=COLORS["text_hi"]).grid(row=0, column=0, sticky="e", padx=(0, 8), pady=4)
        tip(self._entry(sec, self.in_path, 34), "Video or audio file.").grid(row=0, column=1, columnspan=4, sticky="ew")
        self._button(sec, "Browse", self._pick_input).grid(row=0, column=5, padx=(8, 0))
        tk.Label(sec, text="Reference", bg=COLORS["panel"], fg=COLORS["text_hi"]).grid(row=1, column=0, sticky="e", padx=(0, 8), pady=4)
        tip(self._entry(sec, self.ref_path, 34), "Optional lav/recorder sample from the same shoot for tone and dynamics matching.").grid(row=1, column=1, columnspan=4, sticky="ew")
        self._button(sec, "Sample", self._pick_reference).grid(row=1, column=5, padx=(8, 0))
        tk.Label(sec, text="Profile", bg=COLORS["panel"], fg=COLORS["text_hi"]).grid(row=2, column=0, sticky="e", padx=(0, 8), pady=4)
        tip(self._entry(sec, self.profile_path, 34), "Optional camera-to-lav profile JSON made by community_training.py.").grid(row=2, column=1, columnspan=4, sticky="ew")
        self._button(sec, "JSON", self._pick_profile).grid(row=2, column=5, padx=(8, 0))
        tk.Label(sec, text="Model", bg=COLORS["panel"], fg=COLORS["text_hi"]).grid(row=3, column=0, sticky="e", padx=(0, 8), pady=4)
        tip(self._entry(sec, self.model_path, 34), "Optional model.pt made by train_reference_model.py or downloaded from Hugging Face.").grid(row=3, column=1, columnspan=4, sticky="ew")
        self._button(sec, "PT", self._pick_model).grid(row=3, column=5, padx=(8, 0))
        tk.Label(sec, text="Output", bg=COLORS["panel"], fg=COLORS["text_hi"]).grid(row=4, column=0, sticky="e", padx=(0, 8), pady=4)
        tip(self._entry(sec, self.out_path, 34), "Folder for extracted and cleaned audio.").grid(row=4, column=1, columnspan=4, sticky="ew")
        self._button(sec, "Folder", self._pick_output).grid(row=4, column=5, padx=(8, 0))

    def _engine_section(self, parent: tk.Widget) -> None:
        sec = self._section(parent, "AI ENGINE")
        tk.Label(sec, text="Device", bg=COLORS["panel"], fg=COLORS["text_hi"]).grid(row=0, column=0, sticky="e", padx=(0, 8), pady=4)
        self._combo(sec, self.device, ["auto", "cuda", "mps", "cpu"], 8).grid(row=0, column=1, sticky="w")
        tk.Label(sec, text="Segment", bg=COLORS["panel"], fg=COLORS["text_hi"]).grid(row=0, column=2, sticky="e", padx=(10, 8), pady=4)
        self._spin(sec, self.segment, 1, 60).grid(row=0, column=3, sticky="w")
        tk.Label(sec, text="Model", bg=COLORS["panel"], fg=COLORS["text_hi"]).grid(row=1, column=0, sticky="e", padx=(0, 8), pady=4)
        self._combo(sec, self.model, ["htdemucs_ft", "htdemucs", "mdx_extra", "mdx_extra_q"], 14).grid(row=1, column=1, columnspan=3, sticky="w")
        tk.Label(sec, text="Denoise", bg=COLORS["panel"], fg=COLORS["text_hi"]).grid(row=2, column=0, sticky="e", padx=(0, 8), pady=4)
        self._combo(sec, self.denoise_model, ["dns64", "dns48", "master64", "valentini_nc"], 14).grid(row=2, column=1, sticky="w")
        tk.Label(sec, text="Dry %", bg=COLORS["panel"], fg=COLORS["text_hi"]).grid(row=2, column=2, sticky="e", padx=(10, 8), pady=4)
        self._spin(sec, self.denoise_dry, 0, 60).grid(row=2, column=3, sticky="w")
        self._check(sec, "Keep background_no_voice.wav", self.keep_bg).grid(row=3, column=0, columnspan=5, sticky="w", pady=(8, 0))

    def _polish_section(self, parent: tk.Widget) -> None:
        sec = self._section(parent, "VOICE POLISH")
        tk.Label(sec, text="Preset", bg=COLORS["panel"], fg=COLORS["text_hi"]).grid(row=0, column=0, sticky="e", padx=(0, 8), pady=4)
        self._combo(sec, self.polish_preset, ["speech", "web", "broadcast", "camera-hiss", "raw"], 12).grid(row=0, column=1, sticky="w")
        tk.Label(sec, text="LUFS", bg=COLORS["panel"], fg=COLORS["text_hi"]).grid(row=0, column=2, sticky="e", padx=(10, 8), pady=4)
        self._spin(sec, self.target_lufs, -30, -8).grid(row=0, column=3, sticky="w")
        tk.Label(sec, text="Peak %", bg=COLORS["panel"], fg=COLORS["text_hi"]).grid(row=1, column=0, sticky="e", padx=(0, 8), pady=4)
        self._spin(sec, self.peak_ceiling, 50, 99).grid(row=1, column=1, sticky="w")
        self._check(sec, "Compressor", self.compressor).grid(row=1, column=2, sticky="w", padx=(10, 4))
        self._check(sec, "De-esser", self.deesser).grid(row=1, column=3, sticky="w", padx=(4, 0))
        self._check(sec, "Loudness", self.loudness).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

    def _run_section(self, parent: tk.Widget) -> None:
        sec = self._section(parent, "TRANSPORT")
        self.run_btn = self._button(sec, "RUN CLEAN", self._run, accent=COLORS["cyan"])
        self.run_btn.grid(row=0, column=0, sticky="w")
        self.stop_btn = self._button(sec, "STOP", self._stop, accent=COLORS["magenta"])
        self.stop_btn.config(state="disabled")
        self.stop_btn.grid(row=0, column=1, sticky="w", padx=8)
        self.progress = ttk.Progressbar(sec, mode="indeterminate", style="Cyber.Horizontal.TProgressbar")
        self.progress.grid(row=1, column=0, columnspan=6, sticky="ew", pady=(12, 0))

    def _pick_input(self) -> None:
        p = filedialog.askopenfilename(title="Select video/audio", filetypes=VIDEO_TYPES)
        if p:
            self.in_path.set(p)

    def _pick_reference(self) -> None:
        p = filedialog.askopenfilename(title="Select reference audio/video", filetypes=VIDEO_TYPES)
        if p:
            self.ref_path.set(p)

    def _pick_profile(self) -> None:
        p = filedialog.askopenfilename(title="Select reference profile", filetypes=[("JSON profile", "*.json"), ("All files", "*.*")])
        if p:
            self.profile_path.set(p)

    def _pick_model(self) -> None:
        p = filedialog.askopenfilename(title="Select reference model", filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")])
        if p:
            self.model_path.set(p)

    def _pick_output(self) -> None:
        p = filedialog.askdirectory(title="Select output folder", initialdir=self.out_path.get() or str(HERE))
        if p:
            self.out_path.set(p)

    def _run(self) -> None:
        src = self.in_path.get().strip()
        if not src or not Path(src).is_file():
            messagebox.showerror("No input", "Select an existing video/audio file first.")
            return
        out = self.out_path.get().strip() or str(HERE / "output")
        Path(out).mkdir(parents=True, exist_ok=True)
        self._save_settings()
        cmd = [
            sys.executable, "-u", str(PIPELINE), src,
            "--out", out,
            "--device", self.device.get(),
            "--model", self.model.get(),
            "--segment", str(self.segment.get()),
            "--denoise-model", self.denoise_model.get(),
            "--denoise-dry", str(max(0, min(60, int(self.denoise_dry.get()))) / 100),
            "--polish-preset", self.polish_preset.get(),
            "--target-lufs", str(int(self.target_lufs.get())),
            "--peak-ceiling", str(max(50, min(99, int(self.peak_ceiling.get()))) / 100),
        ]
        if not self.compressor.get():
            cmd.append("--no-compressor")
        if not self.deesser.get():
            cmd.append("--no-deesser")
        if not self.loudness.get():
            cmd.append("--no-loudness")
        ref = self.ref_path.get().strip()
        if ref:
            cmd.extend(["--reference-audio", ref])
        profile = self.profile_path.get().strip()
        if profile and not ref:
            cmd.extend(["--reference-profile", profile])
        model = self.model_path.get().strip()
        if model and not ref and not profile:
            cmd.extend(["--reference-model", model])
        if not self.keep_bg.get():
            cmd.append("--no-instrumental")
        self.last_result_dir = Path(out) / Path(src).stem
        self._launch(cmd)

    def _launch(self, cmd: list[str]) -> None:
        if self.proc and self.proc.poll() is None:
            return
        self.log.delete("1.0", "end")
        self.results.delete(0, "end")
        self._log("$ " + " ".join(f'"{c}"' if " " in c else c for c in cmd) + "\n\n")
        self.status.set("RUNNING")
        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.progress.start(12)
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        def worker() -> None:
            try:
                self.proc = subprocess.Popen(
                    cmd,
                    cwd=str(HERE),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
                )
                assert self.proc.stdout is not None
                for line in self.proc.stdout:
                    self.q.put(line)
                code = self.proc.wait()
                self.q.put(f"__DONE__{code}")
            except Exception as exc:
                self.q.put(f"\n[gui error] {exc!r}\n")
                self.q.put("__DONE__1")

        threading.Thread(target=worker, daemon=True).start()

    def _stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False)
            else:
                self.proc.terminate()
            self._log("\n[stopped]\n")
            self.status.set("STOPPED")

    def _drain_log(self) -> None:
        try:
            while True:
                item = self.q.get_nowait()
                if item.startswith("__DONE__"):
                    self._finish(item.replace("__DONE__", ""))
                else:
                    self._log(item)
        except queue.Empty:
            pass
        self.root.after(100, self._drain_log)

    def _finish(self, code: str) -> None:
        self.progress.stop()
        self.run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.proc = None
        self.status.set("DONE" if code == "0" else f"ERROR {code}")
        if code == "0":
            self._populate_results()

    def _populate_results(self) -> None:
        self.results.delete(0, "end")
        d = self.last_result_dir
        if not d:
            return
        for f in sorted((d / "audio").glob("*.wav")) if (d / "audio").is_dir() else []:
            self.results.insert("end", f.name)

    def _open_selected(self) -> None:
        sel = self.results.curselection()
        if not sel or not self.last_result_dir:
            return
        p = self.last_result_dir / "audio" / self.results.get(sel[0])
        if p.is_file():
            self._open_path(p)

    def _open_result_dir(self) -> None:
        d = self.last_result_dir or Path(self.out_path.get())
        d.mkdir(parents=True, exist_ok=True)
        self._open_path(d)

    def _open_path(self, path: Path) -> None:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        elif os.name == "nt":
            os.startfile(str(path))
        else:
            subprocess.run(["xdg-open", str(path)], check=False)

    def _log(self, text: str) -> None:
        self.log.insert("end", text)
        self.log.see("end")

    def _drag_start(self, event: tk.Event) -> None:
        if not self._maximized:
            self._drag_xy = (event.x_root - self.root.winfo_x(), event.y_root - self.root.winfo_y())

    def _drag_move(self, event: tk.Event) -> None:
        if self._drag_xy and not self._maximized:
            dx, dy = self._drag_xy
            self.root.geometry(f"+{event.x_root - dx}+{event.y_root - dy}")

    def _minimize(self) -> None:
        self.root.overrideredirect(False)
        self.root.iconify()
        self.root.after(200, lambda: self.root.overrideredirect(True))

    def _toggle_max(self) -> None:
        if self._maximized:
            self.root.state("normal")
            self.root.geometry("1120x690")
            self._maximized = False
        else:
            self.root.state("zoomed")
            self._maximized = True

    def _on_close(self) -> None:
        if self.proc and self.proc.poll() is None:
            if not messagebox.askyesno("Running", "Stop processing and exit?"):
                return
            self._stop()
        self._save_settings()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
