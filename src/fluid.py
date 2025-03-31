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

# 10. fluid_mv (Real-time input from metavision)
# This extraction converts the different rates and dimensions to a common unit
"""
Records with no rate = STAT: Records where the rate is not specified.
Records with rate = INFUSION: Records where the rate is specified.
fluids corrected for tonicity
"""


query = """
with t1 as
(
select stay_id, extract(epoch from starttime) as starttime, extract(epoch from endtime) as endtime, itemid, amount, rate,
case 
when itemid in (225823, 225159) then amount *0.5 
when itemid in (227531) then amount *2.75 
when itemid in (225161) then amount *3
when itemid in (220862) then amount *5
when itemid in (220995, 227533) then amount *6.66
when itemid in (228341) then amount *8
else amount end as tev -- total equivalent volume
from mimiciv_icu.inputevents
-- only real time items !!
where stay_id is not null and amount is not null and itemid in (225158, 225943, 226089, 225168,
225828, 220862, 220970, 220864, 225159, 220995, 225170, 225825, 227533, 225161, 227531, 225171, 225827,
225941, 225823, 228341)
)
select stay_id, starttime, endtime, itemid, round(cast(amount as numeric),3) as amount,
round(cast(rate as numeric),3) as rate,round(cast(tev as numeric),3) as tev -- total equiv volume
from t1
order by stay_id, starttime, itemid
"""


"""
 Itemid | Label
-----------------------------------------------------
 225158 | NaCl 0.9%
 225943 | Solution
 226089 | Piggyback
 225168 | Packed Red Blood Cells
 225828 | LR
 220862 | Albumin 25%
 220970 | Fresh Frozen Plasma
 220864 | Albumin 5%
 225170 | Platelets
 225825 | D5NS
 225171 | Cryoprecipitate
 225827 | D5LR
 225941 | D5 1/4NS
 225823 | Dextrose 5%, 1/2 NS
 225159 | NaCl 0.45%
 227531 | Mannitol
 225161 | NaCl 3% (Hypertonic Saline)
 220995 | Sodium Bicarbonate 8.4%
 227533 | Sodium Bicarbonate 8.4% (Amp)
 228341 | NaCl 23.4%
 # I am not very sure if the equivalence conversion is correct. should confirm with domain expert.
"""



d = pd.read_sql_query(query,conn)
d.to_csv(os.path.join(exportdir,'fluid.csv'),index=False,sep='|')