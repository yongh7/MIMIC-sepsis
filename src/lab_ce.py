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


# 6. labs_ce (Labs from chartevents)
# Each itemid here corresponds to single measurement type
query = """
select stay_id, extract(epoch from charttime) as charttime, itemid, valuenum
from mimiciv_icu.chartevents
where valuenum is not null and stay_id is not null and 
itemid in  (221828, 227442, 227464, 220645, 226534, 220602, 
226536, 225664, 220621, 226537, 225624,	220615, 229761,
220635, 225625, 225667,	220587, 220644, 225690, 
225651, 220560, 227456, 227429, 227444, 220228,	
220545, 226540, 220546, 227457,	227465, 227466, 
220507, 227467, 223830, 220224, 220235, 224828, 
225668, 227443, 228640, 227686)
order by stay_id, charttime, itemid
"""

"""
 Itemid | Label
-----------------------------------------------------
 221828 | Hydralazine
 227442 | Potassium (serum)
 227464 | Potassium (whole blood)
 220645 | Sodium (serum)
 226534 | Sodium (whole blood)
 220602 | Chloride (serum)
 226536 | Chloride (whole blood)
 225664 | Glucose finger stick (range 70-100)
 220621 | Glucose (serum)
 226537 | Glucose (whole blood)
 225624 | BUN
 220615 | Creatinine (serum)
 229761 | Creatinine (whole blood)
 220635 | Magnesium
 225625 | Calcium non-ionized
 225667 | Ionized Calcium
 220587 | AST
 220644 | ALT
 225690 | Total Bilirubin
 225651 | Direct Bilirubin
 220650 | Total Protein
 227456 | Albumin
 227429 | Troponin-T
 227444 | C-Reactive Protein
 220228 | Hemoglobin
 220545 | Hematocrit (serum)
 226540 | Hematocrit (whole blood - calc)
 220546 | WBC
 227457 | Platelet Count
 227465 | PT
 227466 | PTT
 220507 | Activated Clotting Time
 227467 | INR
 223830 | PH (Arterial)
 220224 | Arterial O2 pressure
 220235 | Arterial CO2 pressure
 224828 | Arterial Base Excess
 225668 | Lactic Acid
 227443 | HCO3 (serum)
 228640 | EtCO2
 227686 | Central Venous O2% Sat

"""

d = pd.read_sql_query(query,conn)
d.to_csv(os.path.join(exportdir,'labs_ce.csv'),index=False,sep='|')