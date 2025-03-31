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

# 8. uo (Real-time Urine Output)
query = """
select stay_id, extract(epoch from charttime) as charttime, itemid, value
from mimiciv_icu.outputevents
where stay_id is not null and value is not null and itemid in 
(226559, 226560, 227510, 226561, 227489,
226584, 226563, 226564, 226565, 226557, 226558, 226713, 226567)
order by stay_id, charttime, itemid
"""

"""
 Itemid | Label
-----------------------------------------------------
 226559 | Foley
 226560 | Void
 227510 | TF Residual
 226561 | Condom Cath
 227489 | GU Irrigant/Urine Volume Out
 226584 | Ileoconduit
 226563 | Suprapubic
 226564 | R Nephrostomy
 226565 | L Nephrostomy
 226557 | R Ureteral Stent
 226558 | L Ureteral Stent
 226713 | Incontinent/voids (estimate)
 226567 | Straight Cath
"""

d = pd.read_sql_query(query,conn)
d.to_csv(os.path.join(exportdir,'uo.csv'),index=False,sep='|')