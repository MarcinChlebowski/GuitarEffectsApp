import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from ttkthemes import ThemedTk
import os
from pedalboard import Pedalboard, Chorus, Reverb, Distortion, Phaser
from pedalboard.io import AudioFile
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sounddevice as sd
import soundfile as sf
import time

def delay_effect(chunk, fs, delay_time, decay):

    delay_samples = int(delay_time * fs)

    delayed_chunk = np.zeros_like(chunk)

    for i in range(len(chunk)):
        if i - delay_samples < 0:
            delayed_chunk[i] = chunk[i]
        else:
            delayed_chunk[i] = chunk[i] + decay * delayed_chunk[i - delay_samples]
    
    return delayed_chunk

class GuitarEffectsApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Guitar Effects App")

        self.effect_var = tk.StringVar()
        self.file_path_var = tk.StringVar()
        self.recording_index = 1
        self.is_recording = False

        self.effect_var.trace_add('write', self.on_effect_change)
        self.param_labels = []
        self.param_entries = []
        self.scale_value_labels = []

        self.create_widgets()


    def create_widgets(self):
        file_label = ttk.Label(self.master, text="Wybierz plik dźwiękowy:")
        self.file_list = ttk.Combobox(self.master, textvariable=self.file_path_var, state='readonly')
        self.populate_file_list()

        file_label.place(x=25,y=40)
        self.file_list.place(x=180,y=35)

        effect_label = ttk.Label(self.master, text="Wybierz efekt:")
        effect_combo = ttk.Combobox(self.master, textvariable=self.effect_var, values=["Chorus", "Reverb", "Distortion", "Phaser", "Delay"])
        apply_button = ttk.Button(self.master, text="Zastosuj efekt", command=self.apply_effect)

        effect_label.place(x=25,y=100)
        effect_combo.place(x=115,y=95)
        apply_button.place(x=285,y=95)

        self.param_creators = {
            "Chorus": self.create_chorus_params,
            "Reverb": self.create_reverb_params,
            "Distortion": self.create_distortion_params,
            "Phaser": self.create_phaser_params,
            "Delay": self.create_delay_params
        }

        self.create_params_for_effect()

        self.record_button = ttk.Button(self.master, text="Rozpocznij nagrywanie", command=self.start_recording)
        self.stop_button = ttk.Button(self.master, text="Zakończ nagrywanie", command=self.stop_recording, state=tk.DISABLED)

        self.record_button.place(x=25, y=160)
        self.stop_button.place(x=220, y=160)

        play_original_button = ttk.Button(self.master, text="Odtwórz oryginał", command=self.play_original)
        play_processed_button = ttk.Button(self.master, text="Odtwórz przetworzone", command=self.play_processed)

        play_original_button.place(x=540, y=320)
        play_processed_button.place(x=1030, y=320)

        self.stop_original_button = ttk.Button(self.master, text="Stop oryginał", command=self.stop_original)
        self.stop_processed_button = ttk.Button(self.master, text="Stop przetworzony", command=self.stop_processed)

        self.stop_original_button.place(x=690, y=320)
        self.stop_processed_button.place(x=1220, y=320)


        self.fig_original, self.ax_original = plt.subplots(figsize=(5, 3), tight_layout=True)
        self.canvas_original = FigureCanvasTkAgg(self.fig_original, master=self.master)
        self.canvas_widget_original = self.canvas_original.get_tk_widget()
        self.canvas_widget_original.place(x=420, y=10)

        self.fig, self.ax = plt.subplots(figsize=(5, 3), tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.place(x=940, y=10)

        self.fig_combined, self.ax_combined = plt.subplots(figsize=(5, 3), tight_layout=True)
        self.canvas_combined = FigureCanvasTkAgg(self.fig_combined, master=self.master)
        self.canvas_widget_combined = self.canvas_combined.get_tk_widget()
        self.canvas_widget_combined.place(x=670, y=370)
        self.processed_handle = plt.Line2D([], [], color='blue', label='Przetworzony')
        self.original_handle = plt.Line2D([], [], color='orange', label='Oryginalny')
        

    def populate_file_list(self):
        files = os.listdir()
        audio_files = [file for file in files if file.endswith((".wav", ".mp3"))]
        self.file_list['values'] = audio_files

    def on_browse(self):
        selected_file = self.file_list.get()
        self.file_path_var.set(selected_file)

    def create_params_for_effect(self):
        selected_effect = self.effect_var.get()

        self.clear_param_widgets()

        if selected_effect in self.param_creators:
            self.param_creators[selected_effect]()
        else:
            self.clear_param_widgets()

    def clear_param_widgets(self):
        if hasattr(self, 'param_labels') and hasattr(self, 'param_entries'):
            for label in self.param_labels:
                label.destroy()
            for entry in self.param_entries:
                entry.destroy()
            self.param_labels.clear()
            self.param_entries.clear()

            if hasattr(self, 'scale_value_labels'):
                for label in self.scale_value_labels:
                    label.destroy()
                self.scale_value_labels.clear()

    def on_effect_change(self, *args):
        self.clear_param_widgets()
        self.create_params_for_effect()

    def create_chorus_params(self):
        if hasattr(self, 'scale_value_labels'):
            for label in self.scale_value_labels:
                label.destroy()
            self.scale_value_labels.clear()

        self.clear_param_widgets()
        param_labels = ["rate_hz", "depth", "centre_delay_ms", "feedback", "mix"]
        y_offset = 265
        scale_ranges = [(0, 100), (0, 1), (0, 100), (-1, 1), (0, 1)] 

        for i, (label, scale_range) in enumerate(zip(param_labels, scale_ranges)):
            param_label = ttk.Label(self.master, text=label)
            param_label.place(x=15, y=y_offset + 40 * i) 

            param_scale = ttk.Scale(self.master, from_=scale_range[0], to=scale_range[1], length=200,
                                    orient=tk.HORIZONTAL, style="Horizontal.TScale")
            param_scale.place(x=190, y=y_offset + 40 * i)

            param_scale.set(scale_range[0])

            self.param_labels.append(param_label)
            self.param_entries.append(param_scale)

            scale_value_label = ttk.Label(self.master, text=scale_range[0])
            scale_value_label.place(x=390, y=y_offset + 40 * i)
            self.scale_value_labels.append(scale_value_label)

            def update_scale_value(value, label):
                label.config(text=value)

            param_scale.config(command=lambda value, label=scale_value_label: update_scale_value(round(float(value), 2), label))
            update_scale_value(scale_range[0], scale_value_label)

        return self.scale_value_labels

    def create_reverb_params(self):
        if hasattr(self, 'scale_value_labels'):
            for label in self.scale_value_labels:
                label.destroy()
            self.scale_value_labels.clear()

        self.clear_param_widgets()
        param_labels = ["room_size", "damping", "wet_level", "dry_level", "width",
                        "freeze_mode"]
        y_offset = 265
        scale_ranges = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)] 

        for i, (label, scale_range) in enumerate(zip(param_labels, scale_ranges)):
            param_label = ttk.Label(self.master, text=label)
            param_label.place(x=15, y=y_offset + 40 * i)

            param_scale = ttk.Scale(self.master, from_=scale_range[0], to=scale_range[1], length=200,
                                    orient=tk.HORIZONTAL, style="Horizontal.TScale")
            param_scale.place(x=190, y=y_offset + 40 * i)

            param_scale.set(scale_range[0])

            self.param_labels.append(param_label)
            self.param_entries.append(param_scale)

            scale_value_label = ttk.Label(self.master, text=scale_range[0])
            scale_value_label.place(x=390, y=y_offset + 40 * i)
            self.scale_value_labels.append(scale_value_label)

            def update_scale_value(value, label):
                label.config(text=value)

            param_scale.config(command=lambda value, label=scale_value_label: update_scale_value(round(float(value), 2), label))
            update_scale_value(scale_range[0], scale_value_label)

        return self.scale_value_labels

    def create_distortion_params(self):
        if hasattr(self, 'scale_value_labels'):
            for label in self.scale_value_labels:
                label.destroy()
            self.scale_value_labels.clear()

        self.clear_param_widgets()
        param_labels = ["drive_db"]
        y_offset = 265
        scale_ranges = [(0, 50)]

        for i, (label, scale_range) in enumerate(zip(param_labels, scale_ranges)):
            param_label = ttk.Label(self.master, text=label)
            param_label.place(x=15, y=y_offset + 40 * i)

            param_scale = ttk.Scale(self.master, from_=scale_range[0], to=scale_range[1], length=200,
                                    orient=tk.HORIZONTAL, style="Horizontal.TScale")
            param_scale.place(x=190, y=y_offset + 40 * i)

            param_scale.set(scale_range[0])

            self.param_labels.append(param_label)
            self.param_entries.append(param_scale)

            scale_value_label = ttk.Label(self.master, text=scale_range[0])
            scale_value_label.place(x=390, y=y_offset + 40 * i)
            self.scale_value_labels.append(scale_value_label)

            def update_scale_value(value, label):
                label.config(text=value)

            param_scale.config(command=lambda value, label=scale_value_label: update_scale_value(round(float(value), 2), label))
            update_scale_value(scale_range[0], scale_value_label)

        return self.scale_value_labels
    
    def create_phaser_params(self):
        if hasattr(self, 'scale_value_labels'):
            for label in self.scale_value_labels:
                label.destroy()
            self.scale_value_labels.clear()

        self.clear_param_widgets()
        param_labels = ["rate_hz", "depth", "centre_frequency_hz", "feedback", "mix"]
        y_offset = 265
        scale_ranges = [(0, 100), (0, 1), (0, 1300), (-1, 1), (0, 1)]

        for i, (label, scale_range) in enumerate(zip(param_labels, scale_ranges)):
            param_label = ttk.Label(self.master, text=label)
            param_label.place(x=15, y=y_offset + 40 * i)  

            param_scale = ttk.Scale(self.master, from_=scale_range[0], to=scale_range[1], length=200,
                                    orient=tk.HORIZONTAL, style="Horizontal.TScale")
            param_scale.place(x=190, y=y_offset + 40 * i)

            param_scale.set(scale_range[0])

            self.param_labels.append(param_label)
            self.param_entries.append(param_scale)

            scale_value_label = ttk.Label(self.master, text=scale_range[0])
            scale_value_label.place(x=390, y=y_offset + 40 * i)
            self.scale_value_labels.append(scale_value_label)

            def update_scale_value(value, label):
                label.config(text=value)

            param_scale.config(command=lambda value, label=scale_value_label: update_scale_value(round(float(value), 2), label))
            update_scale_value(scale_range[0], scale_value_label)

        return self.scale_value_labels
    
    def create_delay_params(self):
        if hasattr(self, 'scale_value_labels'):
            for label in self.scale_value_labels:
                label.destroy()
            self.scale_value_labels.clear()

        self.clear_param_widgets()
        param_labels = ["Delay (s)", "Decay"]
        y_offset = 265
        scale_ranges = [(0, 2), (0, 1)]

        for i, (label, scale_range) in enumerate(zip(param_labels, scale_ranges)):
            param_label = ttk.Label(self.master, text=label)
            param_label.place(x=15, y=y_offset + 40 * i)  

            param_scale = ttk.Scale(self.master, from_=scale_range[0], to=scale_range[1], length=200,
                                    orient=tk.HORIZONTAL, style="Horizontal.TScale")
            param_scale.place(x=190, y=y_offset + 40 * i)

            param_scale.set(scale_range[0])

            self.param_labels.append(param_label)
            self.param_entries.append(param_scale)

            scale_value_label = ttk.Label(self.master, text=scale_range[0])
            scale_value_label.place(x=390, y=y_offset + 40 * i)
            self.scale_value_labels.append(scale_value_label)

            def update_scale_value(value, label):
                label.config(text=value)

            param_scale.config(command=lambda value, label=scale_value_label: update_scale_value(round(float(value), 2), label))
            update_scale_value(scale_range[0], scale_value_label)

        return self.scale_value_labels

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.stop_button.config(state=tk.NORMAL)
            self.record_button.config(state=tk.DISABLED)

            self.recording_data = []
            self.samplerate = sd.query_devices(None, 'input')['default_samplerate']
            self.recording_stream = sd.InputStream(callback=self.record_callback, samplerate=self.samplerate)
            self.recording_stream.start()


    def record_callback(self, indata, frames, time, status):
        if status:
            print(status)
        if self.is_recording:
            self.recording_data.append(indata.copy())

    def stop_recording(self):
        self.is_recording = False
        self.recording_stream.stop()
        self.stop_button.config(state=tk.DISABLED)
        self.record_button.config(state=tk.NORMAL)

        recording_array = np.vstack(self.recording_data)
        recording_path = f"Recording{self.recording_index}.wav"
        wavfile.write(recording_path, int(self.samplerate), recording_array)
        
        recording_info_path = f"Recording{self.recording_index}_info.txt"
        with open(recording_info_path, "w") as info_file:
            info_file.write(f"Length: {len(self.recording_data) / self.samplerate} seconds")

        self.recording_index += 1
        messagebox.showinfo("Nagranie zakończone", f"Nagranie zostało zapisane jako {recording_path}")

        self.recording_data.clear()

        self.populate_file_list()

    def apply_effect(self):
        file_path = self.file_path_var.get()
        if not file_path:
            tk.messagebox.showwarning("Brak pliku", "Proszę wybrać plik dźwiękowy.")
            return

        try:
            self.ax.clear()
            self.canvas.draw()

            self.progress_label = ttk.Label(self.master, text="Postęp:")
            self.progress_var = tk.DoubleVar()
            self.progress_bar = ttk.Progressbar(self.master, orient="horizontal", mode="determinate", variable=self.progress_var)
            
            self.progress_label.place(x=50,y=655)
            self.progress_bar.place(x=30,y=675)

            fs_original, signal_original = wavfile.read(file_path)
            max_amplitude_original = np.max(np.abs(signal_original))

            self.ax_original.clear()
            self.ax_original.plot(np.arange(len(signal_original)) / fs_original, signal_original, color='orange', linewidth=0.5)
            self.ax_original.set_title('Oryginalny')
            self.ax_original.set_xlabel('Czas (s)')
            self.ax_original.set_ylabel('Amplituda')
            self.ax_original.grid(True)
            self.canvas_original.draw()

            selected_effect = self.effect_var.get()
            if selected_effect == "Chorus":
                chorus_param = {
                    'rate_hz': float(self.param_entries[0].get()),
                    'depth': float(self.param_entries[1].get()),
                    'centre_delay_ms': float(self.param_entries[2].get()),
                    'feedback': float(self.param_entries[3].get()),
                    'mix': float(self.param_entries[4].get())
                }
                chorus = Chorus(**chorus_param)
                board = Pedalboard([chorus])

            elif selected_effect == "Reverb":
                reverb_param = {
                    'room_size': float(self.param_entries[0].get()),
                    'damping': float(self.param_entries[1].get()),
                    'wet_level': float(self.param_entries[2].get()),
                    'dry_level': float(self.param_entries[3].get()),
                    'width': float(self.param_entries[4].get()),
                    'freeze_mode': float(self.param_entries[5].get())
                }
                reverb = Reverb(**reverb_param)
                board = Pedalboard([reverb])

            elif selected_effect == "Distortion":
                distortion_param = {
                    'drive_db': float(self.param_entries[0].get()),
                }
                distortion = Distortion(**distortion_param)
                board = Pedalboard([distortion])


            elif selected_effect == "Phaser":
                phaser_param = {
                    'rate_hz': float(self.param_entries[0].get()),
                    'depth': float(self.param_entries[1].get()),
                    'centre_frequency_hz': float(self.param_entries[2].get()),
                    'feedback': float(self.param_entries[3].get()),
                    'mix': float(self.param_entries[4].get())
                }
                phaser = Phaser(**phaser_param)
                board = Pedalboard([phaser])

            elif selected_effect == "Delay":
                delay_time_s = float(self.param_entries[0].get())
                decay = float(self.param_entries[1].get())
                mono_signal = signal_original[:, 0]

                processed_signal = delay_effect(mono_signal, fs_original, delay_time_s, decay)

    
            else:
                board = Pedalboard([])

            output_folder = os.path.join(os.path.dirname(__file__), 'processed output')
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, 'output.wav')



            start_time = time.time()

            if selected_effect == "Delay":
                with AudioFile(output_path, 'w', fs_original, 1) as o:
                    for i in range(0, len(processed_signal), fs_original):
                        chunk = processed_signal[i:i+fs_original]
                        o.write(chunk)

                        self.progress_var.set((i + fs_original) / len(processed_signal) * 100)
                        self.master.update_idletasks()

            else:

                with AudioFile(output_path, 'w', fs_original, signal_original.shape[1]) as o:
                    for i in range(0, len(signal_original), fs_original):
                        chunk = signal_original[i:i+fs_original]

                        effected_chunk = np.array([board(chunk[:, j], fs_original) for j in range(chunk.shape[1])])
                        o.write(effected_chunk.T)

                        self.progress_var.set((i + fs_original) / len(signal_original) * 100)
                        self.master.update_idletasks()

            end_time = time.time()
            elapsed_time = end_time - start_time

            fs, signal = wavfile.read(output_path)

            max_amplitude_processed = np.max(np.abs(signal))

            scaling_factor = max_amplitude_original / max_amplitude_processed

            normalized_signal = scaling_factor * signal

            self.ax.clear()
            self.ax.plot(np.arange(len(normalized_signal)) / fs, normalized_signal, color='blue', linewidth=0.5)
            self.ax.set_title('Przetworzony')
            self.ax.set_xlabel('Czas (s)')
            self.ax.set_ylabel('Amplituda')
            self.ax.grid(True)
            self.canvas.draw()

            self.ax_combined.clear()
            self.ax_combined.plot(np.arange(len(signal)) / fs, normalized_signal, color='blue', linewidth=0.5)
            self.ax_combined.plot(np.arange(len(signal_original)) / fs_original, signal_original, color='orange', linewidth=0.5)
            self.ax_combined.set_title('Połączone wykresy')
            self.ax_combined.set_xlabel('Czas (s)')
            self.ax_combined.set_ylabel('Amplituda')
            self.ax_combined.grid(True)
            
            self.ax_combined.legend(handles=[self.original_handle, self.processed_handle], loc='lower left')
            
            self.canvas_combined.draw()

            messagebox.showinfo("Zastosowano efekt", f"Efekt został zastosowany i zapisany do pliku.\nCzas aplikowania efektu: {elapsed_time:.2f} sekundy")

            self.progress_bar.place_forget()
            self.progress_label.place_forget()


        except Exception as e:
            tk.messagebox.showerror("Błąd", f"Wystąpił błąd podczas przetwarzania pliku: {e}")


    def play_original(self):
        file_path = self.file_path_var.get()
        if not file_path:
            tk.messagebox.showwarning("Brak pliku", "Proszę wybrać plik dźwiękowy.")
            return
        try:
            self.original_data, fs = sf.read(file_path)
            sd.play(self.original_data, fs)
        except Exception as e:
            tk.messagebox.showerror("Błąd", f"Wystąpił błąd podczas odtwarzania pliku: {e}")

    def play_processed(self):
        output_path = os.path.join(os.path.dirname(__file__), 'processed output', 'output.wav')
        if not os.path.exists(output_path):
            tk.messagebox.showwarning("Brak przetworzonego pliku", "Nie znaleziono przetworzonego pliku dźwiękowego.")
            return
        try:
            self.processed_data, fs = sf.read(output_path)
            sd.play(self.processed_data, fs)
        except Exception as e:
            tk.messagebox.showerror("Błąd", f"Wystąpił błąd podczas odtwarzania pliku: {e}")

    def stop_original(self):
        sd.stop()

    def stop_processed(self):
        sd.stop()


if __name__ == "__main__":
    root = ThemedTk(theme="breeze")
    selected_theme = 'seaborn-v0_8'
    plt.style.use(selected_theme)
    app = GuitarEffectsApp(root)
    root.geometry("1450x700")  

    root.mainloop()