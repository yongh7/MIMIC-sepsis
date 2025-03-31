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

# 2. microbio (Microbiologyevents)
# extract(epoch from charttime) : The number of seconds since 1970-01-01 00:00:00 UTC
query = """
select subject_id, hadm_id, extract(epoch from charttime) as charttime, extract(epoch from chartdate) as chartdate 
from mimiciv_hosp.microbiologyevents
"""

d = pd.read_sql_query(query,conn)
d.to_csv(os.path.join(exportdir, 'microbio.csv'),index=False,sep='|')
