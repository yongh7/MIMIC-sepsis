"""
MIMIC-IV Sepsis Cohort Extraction.

This file is sourced and modified from: https://github.com/matthieukomorowski/AI_Clinician
"""

import argparse
import os

import pandas as pd
import psycopg2 as pg


parser = argparse.ArgumentParser()
parser.add_argument("-u", "--username", help="Username used to access the MIMIC Database", type=str)
parser.add_argument("-p", "--password", help="User's password for MIMIC Database", type=str)
pargs = parser.parse_args()

# Initializing database connection
conn = pg.connect("dbname='mimiciv' user={0} host='localhost' options='--search_path=mimimciv' password={1}".format(pargs.username,pargs.password))

# Path for processed data storage
exportdir = os.path.join(os.getcwd(),'processed_files')

if not os.path.exists(exportdir):
    os.makedirs(exportdir)

# Load all CE data in one query
print("Loading chartevents data...")


query = """
select distinct stay_id, 
    extract(epoch from charttime) as charttime, 
    itemid,
    case
        -- Oxygen Flow Device mapping
        when itemid = 223834 and value = 'None' then '0'
        when itemid = 223834 and value = 'Nasal cannula' then '2'
        when itemid = 223834 and value = 'Face tent' then '3'
        when itemid = 223834 and value = 'Aerosol-cool' then '4'
        when itemid = 223834 and value = 'Trach mask' then '5'
        when itemid = 223834 and value = 'High flow nasal cannula' then '6'
        when itemid = 223834 and value = 'High flow neb' then '6'
        when itemid = 223834 and value = 'Non-rebreather' then '7'
        when itemid = 223834 and value = 'Venti mask' then '8'
        when itemid = 223834 and value = 'Medium conc mask' then '9'
        when itemid = 223834 and value = 'Endotracheal tube' then '10'
        when itemid = 223834 and value = 'Tracheostomy tube' then '11'
        when itemid = 223834 and value = 'T-piece' then '12'
        when itemid = 223834 and value = 'CPAP mask' then '13'
        when itemid = 223834 and value = 'Bipap mask' then '13'
        when itemid = 223834 and value = 'Oxymizer' then '14'
        when itemid = 223834 and value = 'Other' then '15'

        -- Oxygen Flow Device mapping for item 226732
        when itemid = 226732 and value = 'None' then '0'
        when itemid = 226732 and value = 'Nasal cannula' then '2'
        when itemid = 226732 and value = 'Face tent' then '3'
        when itemid = 226732 and value = 'Aerosol-cool' then '4'
        when itemid = 226732 and value = 'Trach mask' then '5'
        when itemid = 226732 and value = 'High flow nasal cannula' then '6'
        when itemid = 226732 and value = 'High flow neb' then '6'
        when itemid = 226732 and value = 'Non-rebreather' then '7'
        when itemid = 226732 and value = 'Venti mask' then '8'
        when itemid = 226732 and value = 'Medium conc mask' then '9'
        when itemid = 226732 and value = 'Endotracheal tube' then '10'
        when itemid = 226732 and value = 'Tracheostomy tube' then '11'
        when itemid = 226732 and value = 'T-piece' then '12'
        when itemid = 226732 and value = 'CPAP mask' then '13'
        when itemid = 226732 and value = 'Bipap mask' then '13'
        when itemid = 226732 and value = 'Oxymizer' then '14'
        when itemid = 226732 and value = 'Ultrasonic neb' then '15'
        when itemid = 226732 and value = 'Vapomist' then '16'
        when itemid = 226732 and value = 'Other' then '17'

        -- RASS score mapping (assuming 228096 is the RASS itemid)
        when itemid = 228096 and value like '0%Alert and calm%' then '0'
        when itemid = 228096 and value like '-1%Awakens to voice%' then '-1'
        when itemid = 228096 and value like '-2%Light sedation%' then '-2'
        when itemid = 228096 and value like '-3%Moderate sedation%' then '-3'
        when itemid = 228096 and value like '-4%Deep sedation%' then '-4'
        when itemid = 228096 and value like '-5%Unarousable%' then '-5'
        when itemid = 228096 and value like '+1%Anxious%' then '1'
        when itemid = 228096 and value like '+2%Frequent nonpurposeful%' then '2'
        when itemid = 228096 and value like '+3%Pulls or removes tube%' then '3'
        when itemid = 228096 and value like '+4%Combative%' then '4'
        
        -- For all other numeric values
        else valuenum
    end as valuenum
from mimiciv_icu.chartevents
where value is not null
and itemid in (226732, 223834, 227287, 224691, 226707, 226730, 224639, 226512, 226531, 228096,
               220045, 220179, 225309, 220050, 227243, 224167, 220181, 220052, 225312, 224322,
               225310, 224643, 227242, 220051, 220180, 220210, 224422, 224690, 220277, 220227,
               223762, 223761, 224027, 220074, 228368, 228177, 223835, 220339, 
               224700, 224686, 224684, 224421, 224687, 224697, 224695, 224696)
order by stay_id, charttime
"""
"""
 Itemid | Label
-----------------------------------------------------
 226732 | Oxygen Flow 
 223834 | Oxygen Flow
 227287 | Oxygen Flow (additional cannula)
 224691 | Oxygen Flow
 226707 | Height
 226730 | Height
 224639 | Weight
 226512 | Weight
 226531 | Weight
 228096 | Richmond-RAS Scale
 220045 | Heart Rate 
 220179 | Systolic Blood Pressure (Arterial)
 225309 | Systolic Blood Pressure (Arterial)
 220050 | Systolic Blood Pressure (Arterial)
 227243 | Systolic Blood Pressure (Arterial)
 224167 | Systolic Blood Pressure (Arterial)
 220181 | Blood Pressure mean
 220052 | Blood Pressure mean
 225312 | Blood Pressure mean
 224322 | Blood Pressure mean
 225310 | Diastolic Blood Pressure (Arterial)
 224643 | Diastolic Blood Pressure (Arterial)
 227242 | Diastolic Blood Pressure (Arterial)
 220051 | Diastolic Blood Pressure (Arterial)
 220180 | Diastolic Blood Pressure (Arterial)
 220210 | Respiratory Rate
 224422 | Respiratory Rate
 224690 | Respiratory Rate
 220277 | O2 saturation pulseoxymetry/SpO2
 220227 | SaO2
 223762 | Body Temperature
 223761 | Body Temperature
 224027 | Body Temperature
 220074 | Central Venous Pressure (CVP)
 220059 | PAP systolic
 220060 | PAP diastolic
 220061 | PAP mean
 228368 | Cardiac Index
 228177 | Cardiac Index
 223835 | Inspired O2 Fraction (FiO2)
 220339 | PEEP
 224700 | PEEP
 224686 | Tidal Volume
 224684 | Tidal Volume
 224421 | Tidal Volume
 224687 | Minute Volume
 224697 | MAP (mean airway pressure)
 224695 | Peak Insp. Pressure
 224696 | Plateau Pressure
 226755 | GCS
 227013 | GCS
"""

# Note: The original code likely used chunks to handle memory constraints or timeout issues
# If you encounter memory issues with this single query approach, we may need to revert to chunking
try:
    ce_data = pd.read_sql_query(query, conn)
    print("Saving combined chartevents data...")
    ce_data.to_csv(os.path.join(exportdir, 'chartevents.csv'), index=False, sep='|')
    print("Completed processing chartevents data")
except Exception as e:
    print("Error: The single query approach failed. You may need to use chunking if:")
    print("1. The database connection times out")
    print("2. The query result exceeds available memory")
    print("3. The database has query size limitations")
    print(f"Error details: {str(e)}")



