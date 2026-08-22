# Processed Data

This folder contains the **processed outputs** used by the Power BI
analysis.

Raw MAST experimental files are not redistributed here.

## Raw Data Source

Download the required source data from the UKAEA Open Data MAST archive:

https://opendata.ukaea.uk/mast-data/

The analysed cohort is associated with the experiment:

**Determine power scaling of NBI fast ion population**

**Shot range: 27929--27938**

## Suggested Contents

Place the final Power BI input datasets in this folder, for example:

``` text
data/
├── nbi_state_response_v7.csv
└── nbi_state_shot_summary_v7.csv
```

Only include outputs that are actually used by the final analysis.

## Reproduction

To reproduce the processed datasets:

1.  Download the required public shot data from UKAEA Open Data.
2.  Run the analysis scripts in `../src/`.
3.  Save the final processed outputs here.
4.  Load these outputs into Power BI.

## Important Variables

The processed datasets may contain fields describing:

-   shot;
-   segment number and ID;
-   NBI configuration;
-   segment timing and duration;
-   mean NBI power;
-   mean neutron rate;
-   mean plasma current;
-   signal standard deviations;
-   integrated neutron output;
-   segment-to-segment change metrics.

See the Python processing scripts for the exact derivation of each
output.
