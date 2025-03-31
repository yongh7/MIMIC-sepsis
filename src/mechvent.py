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

# mechvent (Mechanical ventilation)
query = """
select
    stay_id, extract(epoch from charttime) as charttime    -- case statement determining whether it is an instance of mech vent
    , max(
      case
        when itemid is null or value is null then 0 -- can't have null values
        when itemid = 223894 and value != 'Other/Remarks' THEN 1  -- VentTypeRecorded
        when itemid = 226732 and value = 'Ventilator' THEN 1 -- O2 delivery device == ventilator
        when itemid in
          (
          224687 -- minute volume
          , 224685, 224684, 224686 -- tidal volume
          , 224697, 224695, 224696, 224746, 224747 
          , 226873, 224738, 224419, 224750, 227187 -- Insp pressure
          , 224707, 224709, 224705, 224706 -- APRV pressure
          , 220339, 224700 -- PEEP
          , 224702 -- PCV
          , 227809, 227810 -- ETT
          , 224701 -- PSVlevel
          )
          THEN 1
        else 0
      end
      ) as mechvent

  from mimiciv_icu.chartevents ce
  where value is not null
  and itemid in
  (
      223894 -- vent type
      , 226732 -- O2 delivery device
      , 224687 -- minute volume
      , 224685, 224684, 224686 -- tidal volume
      , 224697, 224695, 224696, 224746, 224747 -- High/Low/Peak/Mean/Neg insp force ("RespPressure")
      , 226873, 224738, 224419, 224750, 227187 -- Insp pressure
      , 224707, 224709, 224705, 224706 -- APRV pressure
      , 220339, 224700 -- PEEP
      , 224702 -- PCV
      , 227809, 227810 -- ETT
      , 224701 -- PSVlevel
  )
  group by stay_id, charttime
"""

d = pd.read_sql_query(query,conn)
d.to_csv(os.path.join(exportdir,'mechvent.csv'),index=False,sep='|')