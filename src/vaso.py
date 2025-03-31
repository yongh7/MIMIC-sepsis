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



# 13. vaso_mv (Vasopressors from metavision)
# This extraction converts the different rates and dimensions to a common units
"""
Drugs converted in noradrenaline-equivalent
Body weight assumed 80 kg when missing
"""
query = """
select stay_id, itemid, extract(epoch from starttime) as starttime, extract(epoch from endtime) as endtime, -- rate, -- ,rateuom,
case when itemid in (221906) and rateuom='mcg/kg/min' then round(cast(rate as numeric),3)  -- norad
when itemid in (221906) and rateuom='mcg/min' then round(cast(rate/80 as numeric),3)  -- norad
when itemid in (221289) and rateuom='mcg/kg/min' then round(cast(rate as numeric),3) -- epi
when itemid in (221289) and rateuom='mcg/min' then round(cast(rate/80 as numeric),3) -- epi
when itemid in (222315) and rate > 0.2 then round(cast(rate*5/60  as numeric),3) -- vasopressin, in U/h
when itemid in (222315) and rateuom='units/min' then round(cast(rate*5 as numeric),3) -- vasopressin
when itemid in (222315) and rateuom='units/hour' then round(cast(rate*5/60 as numeric),3) -- vasopressin
when itemid in (221749) and rateuom='mcg/kg/min' then round(cast(rate*0.45 as numeric),3) -- phenyl
when itemid in (221749) and rateuom='mcg/min' then round(cast(rate*0.45 / 80 as numeric),3) -- phenyl
when itemid in (221662) and rateuom='mcg/kg/min' then round(cast(rate*0.01 as numeric),3)  -- dopa
when itemid in (221662) and rateuom='mcg/min' then round(cast(rate*0.01/80 as numeric),3) else null end as rate_std-- dopa
from mimiciv_icu.inputevents
where itemid in (221749, 221906, 221289, 222315, 221662) and rate is not null and statusdescription <> 'Rewritten'
order by stay_id, itemid, starttime
"""

"""
Itemid | Label
-----------------------------------------------------
221906 | Norepinephrine
221289 | Epinephrine
222315 | Vasopressin
221749 | Phenylephrine
221662 | Dopamine
"""

d = pd.read_sql_query(query,conn)
d.to_csv(os.path.join(exportdir,'vaso.csv'),index=False,sep='|')