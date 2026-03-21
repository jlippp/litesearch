"""
Litesearch GUI — CustomTkinter dashboard for autonomous research on consumer GPUs.
Usage: python gui.py
"""

import os
import sys
import threading
import queue
import time
import subprocess

import customtkinter as ctk

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

VRAM_MIN = 2.0
VRAM_MAX = 32.0
LR_MIN = 0.001
LR_MAX = 0.5
POLL_INTERVAL_MS = 100
MAX_LOG_LINES = 500
RESULTS_FILE = "results.tsv"


def detect_gpu():
    import torch
    if not torch.cuda.is_available():
        return None, 0, 0, (0, 0), False
    name = torch.cuda.get_device_name(0)
    vram_mb = torch.cuda.get_device_properties(0).total_mem / 1024 / 1024
    vram_gb = vram_mb / 1024
    cap = torch.cuda.get_device_capability()
    use_bf16 = cap >= (7, 5)
    return name, vram_gb, vram_mb, cap, use_bf16


class LitesearchApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Litesearch — Autonomous Research for Consumer GPUs")
        self.geometry("900x700")
        self.minsize(800, 600)

        self.training_thread = None
        self.stop_event = None
        self.log_queue = None
        self.is_training = False
        self.current_config = None
        self.result = None

        self.gpu_name, self.gpu_vram_gb, self.gpu_vram_mb, self.gpu_cap, self.use_bf16 = detect_gpu()
        if self.gpu_name is None:
            self._show_no_gpu_error()
            return

        self.vram_var = ctk.DoubleVar(value=min(self.gpu_vram_gb, VRAM_MAX))
        self._build_ui()
        self._update_config_preview()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _show_no_gpu_error(self):
        self.geometry("500x200")
        label = ctk.CTkLabel(
            self,
            text="No CUDA GPU detected.\n\nLitesearch requires an NVIDIA GPU.\nPlease check your drivers and CUDA installation.",
            font=("JetBrains Mono", 14),
            text_color="#ff6b6b",
        )
        label.pack(expand=True, padx=20, pady=20)

    def _build_ui(self):
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=16, pady=16)
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(4, weight=1)

        title = ctk.CTkLabel(main, text="Litesearch", font=("JetBrains Mono", 24, "bold"), text_color="#4fc3f7")
        title.grid(row=0, column=0, sticky="w", pady=(0, 2))

        subtitle = ctk.CTkLabel(main, text="Autonomous Research for Consumer GPUs", font=("JetBrains Mono", 12), text_color="#888888")
        subtitle.grid(row=0, column=0, sticky="w", pady=(32, 0))

        dtype_str = "bfloat16" if self.use_bf16 else "float32"
        gpu_info = ctk.CTkLabel(main, text=f"GPU: {self.gpu_name} ({self.gpu_vram_gb:.1f} GB) | {dtype_str}", font=("JetBrains Mono", 11), text_color="#aaaaaa")
        gpu_info.grid(row=0, column=0, sticky="e", pady=(0, 0))

        # Sliders
        slider_frame = ctk.CTkFrame(main)
        slider_frame.grid(row=1, column=0, sticky="ew", pady=(8, 8))
        slider_frame.grid_columnconfigure(1, weight=1)

        vram_label = ctk.CTkLabel(slider_frame, text="VRAM Budget:", font=("JetBrains Mono", 12), width=120)
        vram_label.grid(row=0, column=0, padx=(12, 8), pady=8, sticky="w")

        self.vram_slider = ctk.CTkSlider(slider_frame, from_=VRAM_MIN, to=VRAM_MAX, variable=self.vram_var, number_of_steps=60, command=self._on_vram_slider_change)
        self.vram_slider.grid(row=0, column=1, padx=8, pady=8, sticky="ew")

        self.vram_value_label = ctk.CTkLabel(slider_frame, text=f"{self.vram_var.get():.1f} GB", font=("JetBrains Mono", 12, "bold"), text_color="#4fc3f7", width=70)
        self.vram_value_label.grid(row=0, column=2, padx=(8, 12), pady=8)

        self.lr_var = ctk.DoubleVar(value=0.04)

        lr_label = ctk.CTkLabel(slider_frame, text="Matrix LR:", font=("JetBrains Mono", 12), width=120)
        lr_label.grid(row=1, column=0, padx=(12, 8), pady=8, sticky="w")

        self.lr_slider = ctk.CTkSlider(slider_frame, from_=LR_MIN, to=LR_MAX, variable=self.lr_var, number_of_steps=100, command=self._on_lr_slider_change)
        self.lr_slider.grid(row=1, column=1, padx=8, pady=8, sticky="ew")

        self.lr_value_label = ctk.CTkLabel(slider_frame, text=f"{self.lr_var.get():.3f}", font=("JetBrains Mono", 12, "bold"), text_color="#4fc3f7", width=70)
        self.lr_value_label.grid(row=1, column=2, padx=(8, 12), pady=8)

        # Config + buttons
        control_frame = ctk.CTkFrame(main)
        control_frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        control_frame.grid_columnconfigure(0, weight=1)

        self.config_label = ctk.CTkLabel(control_frame, text="", font=("JetBrains Mono", 11), text_color="#cccccc", anchor="w")
        self.config_label.grid(row=0, column=0, sticky="w", padx=12, pady=(8, 4))

        btn_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
        btn_frame.grid(row=1, column=0, sticky="ew", padx=12, pady=(4, 8))

        self.start_btn = ctk.CTkButton(btn_frame, text="Start Research", font=("JetBrains Mono", 13, "bold"), fg_color="#2e7d32", hover_color="#1b5e20", height=36, width=160, command=self._on_start)
        self.start_btn.pack(side="left", padx=(0, 8))

        self.stop_btn = ctk.CTkButton(btn_frame, text="Stop", font=("JetBrains Mono", 13, "bold"), fg_color="#c62828", hover_color="#b71c1c", height=36, width=100, command=self._on_stop, state="disabled")
        self.stop_btn.pack(side="left")

        self.status_label = ctk.CTkLabel(btn_frame, text="Ready", font=("JetBrains Mono", 11), text_color="#888888")
        self.status_label.pack(side="right", padx=12)

        # VRAM bar
        vram_bar_frame = ctk.CTkFrame(main)
        vram_bar_frame.grid(row=3, column=0, sticky="ew", pady=(0, 4))
        vram_bar_frame.grid_columnconfigure(1, weight=1)

        vram_bar_label = ctk.CTkLabel(vram_bar_frame, text="VRAM:", font=("JetBrains Mono", 11), width=55)
        vram_bar_label.grid(row=0, column=0, padx=(12, 4), pady=6)

        self.vram_bar = ctk.CTkProgressBar(vram_bar_frame, height=16)
        self.vram_bar.grid(row=0, column=1, padx=4, pady=6, sticky="ew")
        self.vram_bar.set(0)

        self.vram_bar_text = ctk.CTkLabel(vram_bar_frame, text="0 / 0 MB", font=("JetBrains Mono", 10), text_color="#888888", width=120)
        self.vram_bar_text.grid(row=0, column=2, padx=(4, 12), pady=6)

        # Terminal log
        terminal_frame = ctk.CTkFrame(main)
        terminal_frame.grid(row=4, column=0, sticky="nsew", pady=(4, 0))
        terminal_frame.grid_columnconfigure(0, weight=1)
        terminal_frame.grid_rowconfigure(0, weight=1)

        self.terminal = ctk.CTkTextbox(terminal_frame, font=("JetBrains Mono", 11), text_color="#e0e0e0", fg_color="#1a1a2e", scrollbar_button_color="#333355", wrap="word")
        self.terminal.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        self._append_log("Litesearch ready. Adjust VRAM budget and click 'Start Research'.\n")
        self._append_log(f"GPU: {self.gpu_name} ({self.gpu_vram_gb:.1f} GB)\n")
        dtype_str = "bfloat16" if self.use_bf16 else "float32 (Pascal fallback)"
        self._append_log(f"Dtype: {dtype_str}\n\n")

    def _update_config_preview(self):
        vram_mb = self.vram_var.get() * 1024
        from train import compute_optimal_config
        from prepare import VOCAB_SIZE
        try:
            config = compute_optimal_config(vram_mb, self.use_bf16, VOCAB_SIZE)
            self.current_config = config
            nparams = (config['depth'] * (4 * config['n_embd']**2 + 2 * config['n_embd'] * 4 * config['n_embd'])
                       + 2 * VOCAB_SIZE * config['n_embd']
                       + (config['depth'] // 2) * VOCAB_SIZE * config['n_embd'])
            nparams_M = nparams / 1e6
            config_str = (
                f"depth={config['depth']}  "
                f"d={config['n_embd']}  "
                f"heads={config['n_head']}  "
                f"B={config['device_batch_size']}  "
                f"T={config['max_seq_len']}  "
                f"~{nparams_M:.0f}M params  "
                f"~{config['estimated_vram_mb']:.0f}MB est VRAM"
            )
            self.config_label.configure(text=config_str)
        except Exception as e:
            self.config_label.configure(text=f"Config error: {e}")
            self.current_config = None

    def _on_vram_slider_change(self, value):
        self.vram_value_label.configure(text=f"{value:.1f} GB")
        if not self.is_training:
            self._update_config_preview()

    def _on_lr_slider_change(self, value):
        self.lr_value_label.configure(text=f"{value:.3f}")

    def _on_start(self):
        if self.is_training:
            return
        if self.current_config is None:
            self._append_log("ERROR: No valid config. Adjust VRAM slider.\n")
            return

        self._append_log("Checking data...\n")
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch", "data")
        tokenizer_dir = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch", "tokenizer")
        if not os.path.exists(tokenizer_dir) or not os.listdir(tokenizer_dir):
            self._append_log("Tokenizer not found. Run: python prepare.py\n")
            self.status_label.configure(text="Run prepare.py first!", text_color="#ff6b6b")
            return
        if not os.path.exists(cache_dir) or not any(f.endswith('.parquet') for f in os.listdir(cache_dir)):
            self._append_log("Data shards not found. Run: python prepare.py\n")
            self.status_label.configure(text="Run prepare.py first!", text_color="#ff6b6b")
            return

        self.is_training = True
        self.stop_event = threading.Event()
        self.log_queue = queue.Queue()
        self.result = None

        import torch
        torch.cuda.reset_peak_memory_stats()

        config = dict(self.current_config)
        lr_override = self.lr_var.get()

        self.terminal.delete("1.0", "end")

        self._append_log("=" * 60 + "\n")
        self._append_log("Starting research experiment...\n")
        self._append_log(f"VRAM budget: {self.vram_var.get():.1f} GB\n")
        self._append_log(f"Matrix LR: {lr_override:.3f}\n")
        self._append_log(f"Config: depth={config['depth']}, d={config['n_embd']}, "
                        f"B={config['device_batch_size']}, T={config['max_seq_len']}\n")
        self._append_log("=" * 60 + "\n\n")

        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.vram_slider.configure(state="disabled")
        self.status_label.configure(text="Training...", text_color="#4fc3f7")

        self.training_thread = threading.Thread(target=self._training_worker, args=(config, lr_override), daemon=True)
        self.training_thread.start()

        self.after(POLL_INTERVAL_MS, self._poll_training)

    def _on_stop(self):
        if self.stop_event is not None:
            self.stop_event.set()
        self.status_label.configure(text="Stopping...", text_color="#ffab40")

    def _training_worker(self, config, lr_override):
        try:
            from train import run_training
            result = run_training(config=config, lr_override=lr_override, log_queue=self.log_queue, stop_event=self.stop_event)
            self.result = result
        except Exception as e:
            self.log_queue.put(f"\nFATAL ERROR: {e}\n")
            self.result = {'crashed': True, 'val_bpb': 0.0, 'peak_vram_mb': 0.0}
        self.log_queue.put("__TRAINING_DONE__")

    def _poll_training(self):
        if not self.is_training:
            return
        messages = []
        try:
            while True:
                messages.append(self.log_queue.get_nowait())
        except queue.Empty:
            pass
        for msg in messages:
            if msg == "__TRAINING_DONE__":
                self._on_training_done()
                return
            self._append_log(msg)
        self._update_vram_bar()
        self.after(POLL_INTERVAL_MS, self._poll_training)

    def _on_training_done(self):
        self.is_training = False
        self._update_vram_bar()
        if self.result and not self.result.get('crashed', False):
            self._log_results_tsv(self.result)
            val_bpb = self.result.get('val_bpb', 0)
            self.status_label.configure(text=f"Done — val_bpb: {val_bpb:.6f}", text_color="#69f0ae")
        else:
            self.status_label.configure(text="Crashed / Stopped", text_color="#ff6b6b")
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.vram_slider.configure(state="normal")

    def _update_vram_bar(self):
        import torch
        if not torch.cuda.is_available():
            return
        try:
            allocated = torch.cuda.max_memory_allocated() / 1024 / 1024
            total = self.gpu_vram_mb
            frac = min(allocated / total, 1.0) if total > 0 else 0
            self.vram_bar.set(frac)
            if frac < 0.5:
                color = "#69f0ae"
            elif frac < 0.8:
                color = "#ffd54f"
            else:
                color = "#ff6b6b"
            self.vram_bar.configure(progress_color=color)
            self.vram_bar_text.configure(text=f"{allocated:.0f} / {total:.0f} MB ({frac*100:.0f}%)", text_color=color)
        except Exception:
            pass

    def _append_log(self, text):
        self.terminal.insert("end", text)
        self.terminal.see("end")
        line_count = int(self.terminal.index("end-1c").split(".")[0])
        if line_count > MAX_LOG_LINES + 100:
            self.terminal.delete("1.0", f"{line_count - MAX_LOG_LINES + 1}.0")

    def _log_results_tsv(self, result):
        try:
            try:
                commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
            except Exception:
                commit = "unknown"
            val_bpb = result.get('val_bpb', 0.0)
            memory_gb = result.get('peak_vram_mb', 0.0) / 1024
            depth = result.get('depth', '?')
            n_embd = result.get('n_embd', '?')
            status = "crash" if result.get('crashed', False) else "keep"
            description = f"depth={depth} d={n_embd} lr={self.lr_var.get():.3f} vram={self.vram_var.get():.1f}GB"
            if not os.path.exists(RESULTS_FILE):
                with open(RESULTS_FILE, "w") as f:
                    f.write("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")
            with open(RESULTS_FILE, "a") as f:
                f.write(f"{commit}\t{val_bpb:.6f}\t{memory_gb:.1f}\t{status}\t{description}\n")
            self._append_log(f"Results logged to {RESULTS_FILE}\n")
        except Exception as e:
            self._append_log(f"Warning: Could not log results: {e}\n")

    def _on_close(self):
        if self.is_training and self.stop_event is not None:
            self.stop_event.set()
            if self.training_thread is not None:
                self.training_thread.join(timeout=3)
        self.destroy()


if __name__ == "__main__":
    app = LitesearchApp()
    app.mainloop()
