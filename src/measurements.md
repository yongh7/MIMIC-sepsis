## Measurements

Most of the measurements in the patient timeseries data are self-explanatory. The following are some of the more complex measurements that require some explanation.


### Mechanical Ventilation

A binary flag indicating whether the patient is on mechanical ventilation.

### Tidal Volume

The tidal volume is the volume of air inhaled or exhaled during a single breath. It is measured in milliliters (mL). A normal tidal volume is between 500 and 1000 mL.

### Minute Volume

The minute volume is the volume of air inhaled or exhaled during a single minute. It is measured in liters per minute (L/min). The minute volume is calculated by multiplying the tidal volume by the respiratory rate. The normal range is between 5 and 10 L/min.

### Oxygen Flow Rate

The oxygen flow rate is the rate of oxygen flow through the patient's respiratory system. It is measured in liters per minute (L/min). The oxygen flow rate is calculated by dividing the oxygen flow device by the oxygen flow.

### PEEP

PEEP is the positive end-expiratory pressure, which is the pressure at the end of the expiration. It is measured in centimeters of water (cmH2O). The normal range is between 5 and 15 cmH2O.

### SOFA Score

The SOFA score is a measure of the severity of sepsis. It is calculated using the following criteria:

- Respiratory (PaO2/FiO2)
- Cardiovascular (MAP)
- Liver (Bilirubin)
- Neurological (GCS)
- Renal (Creatinine)
- Coagulation (INR)


### Richmond-RAS Scale

The Richmond-RAS Scale is a scale used to assess the severity of sepsis. It is measured in points. when GCS data is missing, we imputed it using the Richmond-RAS Scale.

### FiO2
FiO2 is an important measurement for assessing oxygenation and respiratory function. When FiO2 values are missing, we estimate them based on oxygen flow rates and delivery devices using the following rules:

1. For nasal cannula or no device:
- Flow ≤ 1 L/min: FiO2 = 24%
- Flow ≤ 2 L/min: FiO2 = 28% 
- Flow ≤ 3 L/min: FiO2 = 32%
- Flow ≤ 4 L/min: FiO2 = 36%
- Flow ≤ 5 L/min: FiO2 = 40%
- Flow ≤ 6 L/min: FiO2 = 44%
- Flow ≤ 8 L/min: FiO2 = 50%
- Flow ≤ 10 L/min: FiO2 = 55%
- Flow ≤ 12 L/min: FiO2 = 62%
- Flow ≤ 15 L/min: FiO2 = 70%

2. For face masks and similar devices:
- Flow ≤ 4 L/min: FiO2 = 36%
- Flow ≤ 6 L/min: FiO2 = 40%
- Flow ≤ 8 L/min: FiO2 = 58%
- Flow ≤ 10 L/min: FiO2 = 66%
- Flow ≤ 12 L/min: FiO2 = 69%
- Flow ≤ 15 L/min: FiO2 = 75%

3. For non-rebreather masks:
- Flow ≤ 6 L/min: FiO2 = 60%
- Flow ≤ 8 L/min: FiO2 = 70%
- Flow 8-10 L/min: FiO2 = 80%
- Flow 10-15 L/min: FiO2 = 90%
- Flow ≥ 15 L/min: FiO2 = 100%

4. For CPAP/BiPAP masks:
- Flow < 10 L/min: FiO2 = 60%
- Flow 10-15 L/min: FiO2 = 80%
- Flow ≥ 15 L/min: FiO2 = 100%

5. For Oxymizer devices:
- Flow < 5 L/min: FiO2 = 40%
- Flow 5-10 L/min: FiO2 = 60%
- Flow ≥ 10 L/min: FiO2 = 80%

When no oxygen is being delivered (room air), FiO2 is set to 21%.

FiO2 is a critical measurement needed to:
1. Calculate the P/F ratio (PaO2/FiO2) which is used in SOFA scoring
2. Assess severity of respiratory dysfunction
3. Guide oxygen therapy and ventilator management
4. Monitor response to respiratory interventions


### Charlson Comorbidity Index

The Charlson Comorbidity Index (CCI) is a measure of the severity of comorbidities. It is measured in points.
link: https://www.mdcalc.com/calc/3917/charlson-comorbidity-index-cci

### Antibiotics

The antibiotics data includes:
- abx_given: Binary flag (0/1) indicating if antibiotics are active in the current time window
- hours_since_first_abx: Time in hours since the first antibiotic was administered (null if no antibiotics given)
- num_abx: Number of unique antibiotics active in the current time window

### Vasopressors

Vasopressor data includes:
- vaso_median: Median standardized vasopressor rate during the current time window
- vaso_max: Maximum standardized vasopressor rate during the current time window

The rates are standardized to norepinephrine-equivalent doses. A rate of 0 indicates no vasopressors were administered during that time window.

Vasopressors are used to:
1. Maintain adequate blood pressure and tissue perfusion
2. Treat shock states (especially septic shock)
3. Support cardiovascular function
4. Calculate SOFA cardiovascular subscore

### Gender 
A binary flag indicating the patient's gender. 0 for male, 1 for female.


### Fluid
The fluid data includes:
- fluid_total: Cumulative fluid intake up to the current time window
- fluid_step: Fluid intake during the current time window only
- uo_total: Cumulative urine output up to the current time window 
- uo_step: Urine output during the current time window only
- balance: Net fluid balance (fluid_total - uo_total)

One caveat is that preadmission fluid data is not available (will try to fix this in the future), so the balance may not be 100% accurate.

#### Standardization of fluid data
The fluid data is standardized using a total equivalent volume (TEV) approach to account for different fluid concentrations:

1. Isotonic solutions (1x concentration):
- NaCl 0.9% (Normal Saline)
- Lactated Ringers (LR)
- Solutions/Piggybacks
- Blood Products (Packed RBCs, FFP, Platelets, Cryoprecipitate)
- D5NS (5% Dextrose in Normal Saline)
- D5LR (5% Dextrose in Lactated Ringers)

2. Hypotonic solutions (0.5x concentration):
- NaCl 0.45% (Half Normal Saline) 
- Dextrose 5% in 1/2 Normal Saline

3. Hypertonic solutions (multiplier based on concentration):
- Mannitol (2.75x)
- NaCl 3% (3x)
- Albumin 25% (5x)
- Sodium Bicarbonate 8.4% (6.66x)
- NaCl 23.4% (8x)

The TEV is calculated by multiplying the actual volume by the concentration factor. For example:
- 100ml of NaCl 0.9% = 100ml TEV
- 100ml of NaCl 0.45% = 50ml TEV  
- 100ml of NaCl 3% = 300ml TEV

This standardization allows for more accurate comparison of fluid volumes across different types and concentrations of solutions.


### Septic Shock

The septic shock data includes:
- septic_shock: Binary flag (0/1) indicating if the patient is in septic shock
- septic_shock_censored: Binary flag (0/1) indicating if the patient is censored (post first septic shock)

The septic shock criteria are:
1. Adequate fluid resuscitation (defined as minimum fluid intake (2000mL) over previous 12 hours)
2. After adequate fluids:
    - MAP < 65 mmHg AND
    - Requires vasopressors (indicated by hypotension despite fluids) AND
    - Lactate > 2 mmol/L

The septic shock flag values:
- 0 = No septic shock
- 1 = Septic shock identified
- 2 = Censored (post first shock)
