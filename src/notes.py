"""
MIMIC-IV Sepsis Cohort Extraction.
This script processes discharge notes from CSV files.
"""

import os
import pandas as pd
import json
# Path for processed data storage
exportdir = os.path.join(os.getcwd(),'processed_files')

if not os.path.exists(exportdir):
    os.makedirs(exportdir)

# Read discharge notes CSV and patient timeseries CSV
discharge_notes = pd.read_csv(os.path.join(exportdir, 'discharge.csv'))
patient_ts = pd.read_csv(os.path.join(exportdir, 'patient_timeseries_v2.csv'))
icustays = pd.read_csv(os.path.join(exportdir, 'icustays.csv'), sep='|')

# Get the hadm_ids and stay_ids from patient timeseries
ts_stays = patient_ts[['stay_id']].drop_duplicates()

# Get corresponding hadm_ids from icustays
stay_hadm_mapping = icustays[['stay_id', 'hadm_id', 'subject_id']]

# Merge to get the hadm_ids we want to keep
stays_to_keep = pd.merge(ts_stays, stay_hadm_mapping, on='stay_id', how='left')

# Filter discharge notes to only include relevant admissions
filtered_notes = pd.merge(
    discharge_notes,
    stays_to_keep,
    on=['subject_id', 'hadm_id'],
    how='inner'
)
# Create notes directory
notes_dir = os.path.join(exportdir, 'notes')
if not os.path.exists(notes_dir):
    os.makedirs(notes_dir)

# Add note_id column to patient timeseries
patient_ts = pd.merge(
    patient_ts,
    filtered_notes[['stay_id', 'note_id']],
    on='stay_id',
    how='left'
)

# Save updated patient timeseries
patient_ts.to_csv(os.path.join(exportdir, 'patient_timeseries_v3.csv'), index=False)

# Save individual note JSON files
import pyprind
bar = pyprind.ProgBar(len(filtered_notes), title="Saving notes")
for _, note in filtered_notes.iterrows():
    # Clean up excessive newlines in the text
    cleaned_text = ' '.join(note['text'].split())
    
    note_data = {
        'note_id': note['note_id'],
        'subject_id': note['subject_id'], 
        'hadm_id': note['hadm_id'],
        'charttime': note['charttime'],
        'text': cleaned_text
    }
    
    filename = f"{note['note_id']}.json"
    with open(os.path.join(notes_dir, filename), 'w') as f:
        json.dump(note_data, f)
    bar.update()