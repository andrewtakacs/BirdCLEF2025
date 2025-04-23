import os
import librosa
import numpy as np
import pandas as pd

#Set seed to reproduce "random" results for debugging
np.random.seed(42)

#Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

#Get the project root directory (two levels up from notebooks)
project_root = os.path.dirname(os.path.dirname(script_dir))

#Class labels (file names) from train audio
train_audio_path = os.path.join(project_root, 'rawdata/train_audio/')
class_labels = sorted(os.listdir(train_audio_path))

#List of test soundscapes
test_soundscape_path = os.path.join(project_root, 'rawdata/test_soundscapes/')
test_soundscapes = [os.path.join(test_soundscape_path, afile) for afile in sorted(os.listdir(test_soundscape_path))]

# Open each soundscape and make predictions for 5-second segments
# Use pandas df with 'row_id' plus class labels as columns
predictions = pd.DataFrame(columns=['row_id'] + class_labels)

for soundscape in test_soundscapes:
    # Load audio
    sig, rate = librosa.load(path=soundscape, sr=None)

    # Split into 5-second chunks
    chunks = []
    for i in range(0, len(sig), rate*5):
        chunk = sig[i:i+rate*5]
        chunks.append(chunk)
        
    # Make predictions for each chunk
    for i, chunk in enumerate(chunks):
        # Get row id  (soundscape id + end time of 5s chunk)      
        row_id = os.path.basename(soundscape).split('.')[0] + f'_{i * 5 + 5}'
        
        # Make prediction (let's use random scores for now)
        # scores = model.predict(chunk)...
        scores = np.random.rand(len(class_labels))
        
        # Append to predictions as new row
        new_row = pd.DataFrame([[row_id] + list(scores)], columns=['row_id'] + class_labels)
        predictions = pd.concat([predictions, new_row], axis=0, ignore_index=True)
        
# Save prediction as csv
predictions.to_csv('submission.csv', index=False)
predictions.head() 