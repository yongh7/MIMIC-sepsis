"""
MIMIC-IV Sepsis Cohort Extraction.

This file is sourced and modified from: https://github.com/matthieukomorowski/AI_Clinician
"""

import argparse
import os

import pandas as pd
import psycopg2 as pg


parser = argparse.ArgumentParser()
parser.add_argument("-u", "--username",  help="Username used to access the MIMIC Database", type=str)
parser.add_argument("-p", "--password",  help="User's password for MIMIC Database", type=str)
pargs = parser.parse_args()

# Initializing database connection
conn = pg.connect("dbname='mimiciv' user={0} host='localhost' options='--search_path=mimimciv' password={1}".format(pargs.username,pargs.password))

# Path for processed data storage
exportdir = os.path.join(os.getcwd(),'processed_files')

if not os.path.exists(exportdir):
    os.makedirs(exportdir)


# 7. labs_le (Labs from lab events)
query = """
select xx.stay_id, extract(epoch from f.charttime) as timestp, f.itemid, f.valuenum
from(
select subject_id, hadm_id, stay_id, intime, outtime
from mimiciv_icu.icustays
group by subject_id, hadm_id, stay_id, intime, outtime
) as xx inner join  mimiciv_hosp.labevents as f on f.hadm_id=xx.hadm_id and f.charttime>=xx.intime-interval '1 day' 
and f.charttime<=xx.outtime+interval '1 day'  and f.itemid in  (50971, 50822, 50824, 50806, 50931, 51081, 50885, 51003, 51222,
50810, 51301, 50983, 50902, 50809, 51006, 50912, 50960, 50893, 50808, 50804, 50878, 50861, 51464, 50883, 50976, 50862, 51002, 50889,
50811, 51221, 51279, 51300, 51265, 51275, 51274, 51237, 50820, 50821, 50818, 50802, 50813, 50882, 50803) and valuenum is not null
order by f.hadm_id, timestp, f.itemid
"""
"""
 Itemid | Label
-----------------------------------------------------
 50971 | Potassium
 50822 | Potassium, Whole Blood
 50824 | Sodium, Whole Blood
 50806 | Chloride, Whole Blood
 50931 | Glucose
 51081 | Creatinine, Serum
 50885 | Bilirubin, Total
 51003 | Troponin T
 51222 | Hemoglobin
 50810 | Hematocrit, Calculated
 51301 | White Blood Cells
 50983 | Sodium
 50902 | Chloride
 50809 | Glucose
 51006 | Urea Nitrogen
 50912 | Creatinine
 50960 | Magnesium
 50893 | Calcium, Total
 50808 | Free Calcium
 50804 | Calculated Total CO2
 50878 | Asparate Aminotransferase (AST)
 50861 | Alanine Aminotransferase (ALT)
 51464 | Bilirubin
 50883 | Bilirubin, Direct
 50976 | Protein, Total
 50862 | Albumin
 51002 | Troponin I
 50889 | C-Reactive Protein
 50811 | Hemoglobin
 51221 | Hematocrit
 51279 | Red Blood Cells
 51300 | WBC Count
 51265 | Platelet Count
 51275 | PTT
 51274 | PT
 51237 | INR(PT)
 50820 | pH
 50821 | pO2
 50818 | pCO2
 50802 | Base Excess
 50813 | Lactate
 50882 | Bicarbonate
 50803 | Calculated Bicarbonate, Whole Blood
 
"""


d = pd.read_sql_query(query,conn)
d.to_csv(os.path.join(exportdir,'labs_le.csv'),index=False,sep='|')