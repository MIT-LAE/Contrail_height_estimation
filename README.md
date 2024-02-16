The code in this repository can be used to collocate contrails detected on GOES-16 imagery in CALIOP LIDAR data. The same can be done for cirrus clouds. The resulting data can be used to, amongst others, develop a contrail altitude estimation algorithm. For creating such models, the code in the `mcast-models` repository can be used.

# Setup
The `environment.yml` file can be used to create a `conda` environment with the
required packages for using the code within this repository:
```shell
conda env create --file environment.yml
```
After installation of the required packages, this environment can be activated using the following command
```shell
conda activate contrail-altitude-estimation
```

# Instructions for running the code

## Input data
There are several steps involved in the collocation of GOES-16 and CALIOP LIDAR data. Firstly, input files from different sources are required to perform the collocation. These are (AWS = Amazon Web Services):
| Data | Remote location | Location on `hex.mit.edu` | Required for |
| ---- | -------- | -------- | ------------ |
| GOES-16 ABI-L2 MCMIPC/F data | [AWS](https://registry.opendata.aws/noaa-goes/) | `/net/d13/data/vmeijer/data/noaa-goes16/` and `/net/d13/data/lkulik/data/noaa-goes16/` | Adding GOES-16 radiances to collocated pixel data |
| GOES-16 ABI-L2 MCMIPC/F orthographic projections | N/A | `/net/d13/data/vmeijer/data/` and `/net/d13/data/lkulik/data/` | Contrail detection, collocation, visualization |
| Contrail detections | N/A | `/net/d13/data/vmeijer/data/` and `/home/vmeijer/covid19/data/predictions_wo_sf/` | Collocation of contrails |
| CALIOP L1b data  | https://www-calipso.larc.nasa.gov | `/net/d15/data/vmeijer/CALIOP_L1/` | Collocation of contrails |
| CALIOP L2 data  | https://www-calipso.larc.nasa.gov | `/net/d15/data/vmeijer/CALIOP_L2/` | Collocation of cirrus |
| IIR L1 data  | https://www-calipso.larc.nasa.gov | `/net/d15/data/vmeijer/IIR_L1/` | Visualization |
| ERA5 data | [Copernicus CDS](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=overview)| `/net/d15/data/vmeijer/ERA5/` | For advection during the collocation|

## Script execution order
The scripts in the `scripts/` folder make use of the code within the `CAP` folder to perform the collocation. The different scripts should be run in a particular order. Ensure that the `contrail-altitude-estimation` `conda` environment is activated.

NOTE TO SELF: Input formats for the scripts below should be specified still.

### Contrail collocation
1. Run the `coarse' collocation step, which checks whether contrails are detected in the vicinity of the CALIPSO (satellite equipped with CALIOP) ground track:
```shell
python coarse_L1_collocation.py
```
2. Run the `fine' collocation step, which uses the results from the `coarse' collocation step:
```shell
python fine_L1_collocation.py
```
3. For manual inspection of the collocation results, figures can be generated using:
```shell
python generate_L1_figures.py
```
4. GOES-16 radiance and auxiliary data can be added to the collocation results using the scripts:
```shell
python append_goes_data.py
python append_auxiliary_data.py
```
### Cirrus collocation
There is only a single collocation step for the cirrus data:
```shell
python L2_collocation.py
```
GOES-16 radiance and auxiliary data can be added to the collocation results using the scripts:
```shell
python append_goes_data.py
python append_auxiliary_data.py
```




