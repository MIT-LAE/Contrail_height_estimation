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
There are several steps involved in the collocation of GOES-16 and CALIOP LIDAR data. Firstly, input files from different sources are required to perform the collocation. These are (AWS = Amazon Web Services):
| Data | Remote location | Location on `hex.mit.edu` | Required for |
| ---- | -------- | -------- | ------------ |
| GOES-16 ABI-L2 MCMIPC/F data | [AWS](https://registry.opendata.aws/noaa-goes/) | `/net/d13/data/vmeijer/data/noaa-goes16/` and `/net/d13/data/lkulik/data/noaa-goes16/` | Adding GOES-16 radiances to collocated pixel data |
| GOES-16 ABI-L2 MCMIPC/F orthographic projections | N/A | `/net/d13/data/vmeijer/data/` and `/net/d13/data/lkulik/data/` | Contrail detection, collocation, visualization |
| Contrail detections | N/A | `/net/d13/data/vmeijer/data/` and `/home/vmeijer/covid19/data/predictions_wo_sf/` | Collocation of contrails |
| CALIOP L1b data  | https://www-calipso.larc.nasa.gov | `/net/d15/data/vmeijer/CALIOP_L1/` | Collocation of contrails |
| CALIOP L2 data  | https://www-calipso.larc.nasa.gov | `/net/d15/data/vmeijer/CALIOP_L2/` | Collocation of cirrus |
| IIR L1 data  | https://www-calipso.larc.nasa.gov | `/net/d15/data/vmeijer/IIR_L1/` | Visualization |

