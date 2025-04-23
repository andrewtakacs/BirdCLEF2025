import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import subprocess
import platform

class MelSpectrogramViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Bird Mel Spectrogram Viewer")
        self.spectrograms = {}  # Cache for spectrograms
        self.current_species = None
        self.current_recordings = None
        
        # Read the training data
        try:
            self.df = pd.read_csv('rawdata/train.csv')
            self.scientific_names = sorted(self.df['scientific_name'].unique())
        except Exception as e:
            print(f"Error loading training data: {str(e)}")
            self.df = pd.DataFrame()
            self.scientific_names = []
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Create main container with scrollbar
        self.main_container = tk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas and scrollbar
        self.canvas = tk.Canvas(self.main_container)
        self.scrollbar = ttk.Scrollbar(self.main_container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas)
        
        # Configure canvas
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack scrollbar and canvas
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        # Create control frame
        control_frame = tk.Frame(self.scrollable_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        # Create dropdown for scientific names
        self.name_var = tk.StringVar()
        self.name_dropdown = ttk.Combobox(control_frame, textvariable=self.name_var, values=self.scientific_names)
        self.name_dropdown.pack(side=tk.LEFT, padx=5)
        self.name_dropdown.bind('<<ComboboxSelected>>', self.on_species_select)
        
        # Create process button
        self.process_button = tk.Button(control_frame, text="Process", command=self.process_spectrograms, state=tk.DISABLED)
        self.process_button.pack(side=tk.LEFT, padx=5)
        
        # Create frame for spectrograms
        self.spectrogram_frame = tk.Frame(self.scrollable_frame)
        self.spectrogram_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add mousewheel scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
    def open_audio_file(self, audio_path):
        try:
            if platform.system() == 'Darwin':  # macOS
                subprocess.call(('open', audio_path))
            elif platform.system() == 'Windows':
                os.startfile(audio_path)
            else:  # linux variants
                subprocess.call(('xdg-open', audio_path))
        except Exception as e:
            print(f"Error opening audio file: {str(e)}")
            
    def on_species_select(self, event):
        # Clear previous spectrograms
        for widget in self.spectrogram_frame.winfo_children():
            widget.destroy()
            
        selected_species = self.name_var.get()
        if not selected_species:
            return
            
        self.current_species = selected_species
        try:
            self.current_recordings = self.df[self.df['scientific_name'] == selected_species].head(3)
            if not self.current_recordings.empty:
                self.process_button.config(state=tk.NORMAL)
            else:
                self.process_button.config(state=tk.DISABLED)
        except Exception as e:
            print(f"Error selecting species: {str(e)}")
            self.process_button.config(state=tk.DISABLED)
        
    def process_spectrograms(self):
        if not self.current_species or self.current_recordings is None or self.current_recordings.empty:
            return
            
        # Clear previous spectrograms
        for widget in self.spectrogram_frame.winfo_children():
            widget.destroy()
            
        for idx, recording in self.current_recordings.iterrows():
            # Create a container frame for each spectrogram and its controls
            container = tk.Frame(self.spectrogram_frame)
            container.pack(fill=tk.X, pady=15)
            
            # Load and process the audio file
            audio_path = os.path.join('rawdata', 'train_audio', recording['filename'])
            try:
                # Check if we have cached spectrogram data
                if audio_path not in self.spectrograms:
                    y, sr = librosa.load(audio_path)
                    S = librosa.feature.melspectrogram(y=y, sr=sr)
                    S_dB = librosa.power_to_db(S, ref=np.max)
                    self.spectrograms[audio_path] = (y, sr, S_dB)
                else:
                    y, sr, S_dB = self.spectrograms[audio_path]
                
                # Create a new figure for each spectrogram
                fig, ax = plt.subplots(figsize=(9, 5))
                
                # Display the spectrogram
                img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax)
                ax.set_title(f'{recording["scientific_name"]} - {os.path.basename(recording["filename"])}', pad=20)
                ax.set_xlabel('Time (seconds)', fontsize=10, labelpad=10)
                ax.set_ylabel('Mel Frequency', fontsize=10, labelpad=10)
                ax.tick_params(axis='both', which='major', labelsize=8)
                fig.colorbar(img, ax=ax, format='%+2.0f dB')
                
                # Adjust layout to prevent label cutoff
                plt.tight_layout()
                
                # Create canvas and add to frame
                canvas = FigureCanvasTkAgg(fig, master=container)
                canvas.draw()
                canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                
                # Add play button that opens the audio file
                play_button = tk.Button(container, text="Play Audio", 
                                      command=lambda path=audio_path: self.open_audio_file(path))
                play_button.pack(side=tk.RIGHT, padx=10)
                
            except Exception as e:
                print(f"Error processing audio file {audio_path}: {str(e)}")

def main():
    root = tk.Tk()
    app = MelSpectrogramViewer(root)
    root.mainloop()

if __name__ == "__main__":
    main() 