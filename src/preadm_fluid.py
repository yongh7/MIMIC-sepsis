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


# 12. preadm_fluid (Pre-admission fluid intake)
query = """
with mv as
(
select ie.stay_id, sum(ie.amount) as sum
from mimiciv_icu.inputevents ie, mimiciv_icu.d_items ci
where ie.itemid=ci.itemid and ie.itemid in (
226361,226363, 226364, 226365, 226367, 226368, 
226369, 226370, 226371, 226372, 226375, 226376, 
227070, 227071, 227072)
group by stay_id
)

select pt.stay_id,
case when mv.sum is not null then mv.sum
else null end as inputpreadm
from mimiciv_icu.icustays pt
left outer join mv
on mv.stay_id=pt.stay_id
order by stay_id
"""

"""
 Itemid | Label
-----------------------------------------------------
 226361 | Pre-Admission/Non-ICU Intake
 226363 | Cath Lab Intake
 226364 | OR Crystalloid Intake
 226365 | OR Colloid Intake
 226367 | OR FFP Intake
 226368 | OR Packed RBC Intake
 226369 | OR Platelet Intake
 226370 | OR Autologous Blood Intake
 226371 | OR Cryoprecipitate Intake
 226372 | OR Cell Saver Intake
 226375 | PACU Crystalloid Intake
 226376 | PACU Colloid Intake
 227070 | PACU Packed RBC Intake
 227071 | PACU Platelet Intake
 227072 | PACU FFP Intake
"""


d = pd.read_sql_query(query,conn)
d.to_csv(os.path.join(exportdir,'preadm_fluid.csv'),index=False,sep='|')