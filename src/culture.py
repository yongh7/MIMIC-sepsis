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

# Extraction of sub-tables
# There are 43 tables in the Mimic III database. 
# 26 unique tables; the other 17 are partitions of chartevents that are not to be queried directly 
# See: https://mit-lcp.github.io/mimic-schema-spy/
# We create 15 sub-tables when extracting from the database

# From each table we extract subject ID, admission ID, ICU stay ID 
# and relevant times to assist in joining these tables
# All other specific information extracted will be documented before each section of the following code.


# NOTE: The next three tables are built to help identify when a patient may be 
# considered to be septic, using the Sepsis 3 criteria


# 1. culture
# These correspond to blood/urine/CSF/sputum cultures etc
# There are 18 chartevent tables in the Mimic III database, one unsubscripted and 
# the others subscripted from 1 to 17. We use the unsubscripted one to create the 
# culture subtable. The remaining 17 are just partitions and should not be directly queried.
# The labels corresponding to the 51 itemids in the query below are:
"""
 Itemid | Label
-----------------------------------------------------
 225401 | Blood Cultured
 225437 | CSF Culture
 225444 | Pan Culture
 225451 | Sputum Culture
 225454 | Urine Culture
 225722 | Arterial Line Tip Cultured
 225723 | CCO PAC Line Tip Cultured
 225724 | Cordis/Introducer Line Tip Cultured
 225725 | Dialysis Catheter Tip Cultured
 225726 | Tunneled (Hickman) Line Tip Cultured
 225727 | IABP Line Tip Cultured
 225728 | Midline Tip Cultured
 225729 | Multi Lumen Line Tip Cultured
 225730 | PA Catheter Line Tip Cultured
 225731 | Pheresis Catheter Line Tip Cultured
 225732 | PICC Line Tip Cultured
 225733 | Indwelling Port (PortaCath) Line Tip Cultured
 225734 | Presep Catheter Line Tip Cultured
 225735 | Trauma Line Tip Cultured
 225736 | Triple Introducer Line Tip Cultured
 225768 | Sheath Line Tip Cultured
 225814 | Stool Culture
 225816 | Wound Culture
 225817 | BAL Fluid Culture
 225818 | Pleural Fluid Culture
 226131 | ICP Line Tip Cultured
 227726 | AVA Line Tip Cultured
"""
query = """
select subject_id, hadm_id, stay_id, extract(epoch from charttime) as charttime, itemid
from mimiciv_icu.chartevents
where itemid in (225401, 225437, 225444, 225451, 225454, 225814,
  225816, 225817, 225818, 225722, 225723, 225724, 225725, 225726, 225727, 225728, 225729, 225730, 225731,
  225732, 225733, 227726,  225734, 225735,
  225736, 225768, 226131)
order by subject_id, hadm_id, charttime
"""

d = pd.read_sql_query(query,conn)
d.to_csv(os.path.join(exportdir, 'culture.csv'),index=False,sep='|')