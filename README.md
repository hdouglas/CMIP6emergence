# CMIP6emergence
Code for producing analysis in Douglas et al. (2022)

Steps for running the analysis

SETUP
This code assumes that CMIP5/6 netcdf files are accessible in a directory with the following folder heirarchy:
Variable>Table ID>Source ID>Experiment ID>Variant Label>Grid Label
e.g. SourceDir/tas/Amon/FGOALS-g3/ssp245/r1i1p1f1/gn/tas_Amon_FGOALS-g3_ssp245_r1i1p1f1_gn_201501-201912.nc
CMIP5 files have no grid label, so this folder is omitted for them.

RUNNING
- Scripts are ordered by name. 00 files need not be run for producing main figures in the paper.
- User-changeable script parameters are demarcated.
- Note that function scripts also have user-changeable parameters that need to be checked and aligned with the scripts.
- There's the mainline script series (denoted dea2022_xx_...), but also a parallel series for the CMIP5/RCP data (denoted dea2022_cmip5_xx_...), and a separate series for processing population data (denoted dea2022_pops_xx_...). Later scripts in the mainline series require completion of the other series. 
- In the folder where the scripts are saved, also create a subfolder called 'Plots'.
- If you wish to do the analysis on only those models with ECS in the CMIP5 range, this can be toggled on/off in Script 2. However, you'll have to run the b versions of the subsequent scripts, which call files with different names (see the end of Script 2). 
- The R script is for generating the population-scaled cartograms.
