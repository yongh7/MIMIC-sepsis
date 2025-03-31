import argparse
import pyprind
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
from scipy import stats
from fancyimpute import KNN
from utils import deloutabove, deloutbelow, SAH, fixgaps

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--process_raw", action='store_true', help="If specified, additionally save trajectories without normalized features")
    parser.add_argument("--save_intermediate", action='store_true', default=True, help="If specified, save off intermediate tables used to construct final patient table")
    return parser.parse_args()

def load_processed_files():
    print('Loading processed files created from database using "preprocess.py"')
    files = {
        'stay': 'icustays.csv',
        'abx': 'abx.csv', 
        'culture': 'culture.csv',
        'microbio': 'microbio.csv',
        'demog': 'demog.csv',
        'ce': 'chartevents.csv',
        'MV': 'mechvent.csv',
        'inputpreadm': 'preadm_fluid.csv',
        'fluid': 'fluid.csv',
        'vaso': 'vaso.csv',
        'UO': 'uo.csv'
    }
    
    data = {}
    for key, filename in files.items():
        data[key] = pd.read_csv(f'processed_files/{filename}', sep='|')
        
    # Load and combine lab data
    labs_ce = pd.read_csv('processed_files/labs_ce.csv', sep='|')
    labs_le = pd.read_csv('processed_files/labs_le.csv', sep='|')
    labs_le.rename(columns={'timestp': 'charttime'}, inplace=True)
    data['labU'] = pd.concat([labs_ce, labs_le], sort=False, ignore_index=True)
    
    return data

def process_microbio_data(microbio, culture):
    microbio['charttime'] = microbio['charttime'].fillna(microbio['chartdate'])
    del microbio['chartdate']
    return pd.concat([microbio, culture], sort=False, ignore_index=True)

def process_demog_data(demog):
    demog['morta_90'].fillna(0, inplace=True)
    demog['morta_hosp'].fillna(0, inplace=True) 
    demog['charlson_comorbidity_index'].fillna(0, inplace=True)
    return demog.drop_duplicates(subset=['admittime','dischtime'], keep='first')

def determine_readmission(s, dischtimes, cutoff=3600*24*30):
    subject, admission, discharge = s[['subject_id','admittime','dischtime']]
    subj_stay_idx = np.where(dischtimes[subject]==discharge)[0][0]
    s['re_admission'] = 0
    if subj_stay_idx > 0:
        if (admission - dischtimes[subject][subj_stay_idx-1]) <= cutoff:
            s['re_admission'] = 1
    return s

def fill_missing_icustay_ids(bacterio, demog, abx):
    print('Filling-in missing ICUSTAY IDs in bacterio')
    bar = pyprind.ProgBar(len(bacterio.index))
    for i in bacterio.index:
        if np.isnan(bacterio.loc[i, 'stay_id']):
            o = bacterio.loc[i, 'charttime']
            subjectid = bacterio.loc[i, 'subject_id']
            hadmid = bacterio.loc[i, 'hadm_id']
            
            ii = demog.index[demog['subject_id'] == subjectid].tolist()
            jj = demog.index[(demog['subject_id'] == subjectid) & (demog['hadm_id'] == hadmid)].tolist()
            
            for j in range(len(ii)):
                if (o >= demog.loc[ii[j], 'intime'] - 48*3600) and (o <= demog.loc[ii[j], 'outtime'] + 48*3600):
                    bacterio.loc[i,'stay_id'] = demog.loc[ii[j], 'stay_id']
                elif len(ii)==1:
                    bacterio.loc[i,'stay_id'] = demog.loc[ii[j], 'stay_id']
        bar.update()
                    
    print('Filling-in missing ICUSTAY IDs in ABx')
    bar = pyprind.ProgBar(len(abx.index))
    for i in abx.index:
        o = abx.loc[i,'starttime']
        hadmid = abx.loc[i,'hadm_id']
        ii = demog.index[demog['hadm_id'] == hadmid].tolist()
        for j in range(len(ii)):
            if o >= demog.loc[ii[j],'intime'] - 48*3600 and o <= demog.loc[ii[j], 'outtime'] + 48*3600:
                abx.loc[i, 'stay_id'] = demog.loc[ii[j], 'stay_id']
            elif len(ii) == 1:
                abx.loc[i, 'stay_id'] = demog.loc[ii[j], 'stay_id']
        bar.update()
                
    return bacterio, abx

def find_infection_onset(icustayidlist, abx, bacterio):
    print('Finding presumed onset of infection according to sepsis3 guidelines')
    bar = pyprind.ProgBar(len(icustayidlist))
    onset_rows = []
    num_onset = 0
    
    for icustayid in icustayidlist:
        ab = abx.loc[abx['stay_id'] == icustayid, 'starttime']
        bact = bacterio.loc[bacterio['stay_id'] == icustayid, 'charttime']
        subj_bact = bacterio.loc[bacterio['stay_id'] == icustayid,'subject_id']
        
        if len(ab) > 0 and len(bact) > 0:
            D = cdist(ab.values.reshape(-1,1), bact.values.reshape(-1,1))/3600
            for i in range(D.shape[0]):
                M, I = np.min(D[i,:]), np.argmin(D[i,:])
                ab1 = ab.iloc[i]
                bact1 = bact.iloc[I]
                
                if M <= 24 and ab1 <= bact1:
                    onset_rows.append({
                        'subject_id': subj_bact.iloc[0],
                        'stay_id': icustayid,
                        'onset_time': ab1
                    })
                    num_onset += 1
                    break
                elif M <= 72 and ab1 >= bact1:
                    onset_rows.append({
                        'subject_id': subj_bact.iloc[0],
                        'stay_id': icustayid,
                        'onset_time': bact1
                    })
                    num_onset += 1
                    break
        bar.update()
    print(f'Number of preliminary, presumed septic trajectories: {num_onset}')
    onset_df = pd.DataFrame(onset_rows)
    return onset_df

def main():
    args = parse_args()
    data = load_processed_files()
    
    # Process microbio data
    bacterio = process_microbio_data(data['microbio'], data['culture'])
    
    # Process demographics
    demog = process_demog_data(data['demog'])
    
    # Get list of ICU stay IDs
    icustayidlist = list(demog.stay_id.values)
    
    # Calculate readmissions
    subj_dischtime_list = demog.sort_values(by='admittime').groupby('subject_id').apply(lambda df: np.unique(df.dischtime.values))
    demog = demog.apply(lambda x: determine_readmission(x, subj_dischtime_list), axis=1)
    
    # TODO this part is not needed right now, should be moved to another script
    # Process fluid data
    data['fluid']['norm_rate_of_infusion'] = data['fluid']['tev'] * data['fluid']['rate'] / data['fluid']['amount']
    
    # Fill missing ICU stay IDs
    bacterio, data['abx'] = fill_missing_icustay_ids(bacterio, demog, data['abx'])
    
    # Find infection onset
    onset = find_infection_onset(icustayidlist, data['abx'], bacterio)
    
    # Save processed data if requested
    if args.save_intermediate:
        onset.to_csv('processed_files/onset.csv', sep='|', index=False)
        bacterio.to_csv('processed_files/bacterio_processed.csv', sep='|', index=False)
        demog.to_csv('processed_files/demog_processed.csv', sep='|', index=False)
        # Save lab data
        data['labU'].to_csv('processed_files/labu.csv', sep='|', index=False)
        data['abx'].to_csv('processed_files/abx_processed.csv', sep='|', index=False)
    
    
    return onset, bacterio, demog, data


if __name__ == "__main__":
    main()