# BirdCLEF2025 - Bird Audio Visualization Tool

This project provides a visualization tool for analyzing bird audio recordings as part of the BirdCLEF 2025 challenge. The tool allows users to visualize and analyze bird calls using various audio processing techniques.

## Features

- Interactive GUI for selecting and analyzing bird audio recordings
- Multiple visualization types:
  - Amplitude vs Time plot
  - Magnitude vs Frequency plot
  - Mel Spectrogram visualization
- Audio playback functionality
- Species selection by scientific name
- Scrollable interface for multiple visualizations

## Requirements

- Python 3.x
- Required packages:
  - pandas
  - numpy
  - librosa
  - matplotlib
  - tkinter

## Installation

1. Clone the repository:
```bash
git clone https://github.com/andrewtakacs/BirdCLEF2025.git
cd BirdCLEF2025
```

2. Install required packages:
```bash
pip install pandas numpy librosa matplotlib
```

## Usage

1. Place your audio files in the `rawdata/train_audio` directory
2. Place your training metadata in `rawdata/train.csv`
3. Run the visualization tool:
```bash
python Explore/audiovisual.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 