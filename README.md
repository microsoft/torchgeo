![TorchGeo](logo/logo-color.svg)

Datasets, transforms, and models for geospatial data.

[![docs](https://github.com/microsoft/torchgeo/actions/workflows/docs.yaml/badge.svg)](https://github.com/microsoft/torchgeo/actions/workflows/docs.yaml)
[![style](https://github.com/microsoft/torchgeo/actions/workflows/style.yaml/badge.svg)](https://github.com/microsoft/torchgeo/actions/workflows/style.yaml)
[![tests](https://github.com/microsoft/torchgeo/actions/workflows/tests.yaml/badge.svg)](https://github.com/microsoft/torchgeo/actions/workflows/tests.yaml)

## Project setup

### Conda

```bash
conda config --set channel_priority strict
conda env create --file environment.yml
conda activate torchgeo

# verify that the PyTorch can use the GPU
python -c "import torch; print(torch.cuda.is_available())"
```

## Example training run

```bash
# run the training script with a config file
python train.py config_file=conf/landcoverai.yaml
```

## Developing

```
make tests
```

## Datasets

|                 Dataset                 	|                Imagery Type                	|       Label Type      	|  Dataset Type 	|                     External Link                    	|
|:---------------------------------------:	|:------------------------------------------:	|:---------------------:	|:-------------:	|:----------------------------------------------------:	|
| Smallholder Cashew Plantations in Benin 	| Sentinel-2 (71 scene time series)          	| Semantic segmentation 	| GeoDataset    	| https://registry.mlhub.earth/10.34911/rdnt.hfv20i/   	|
| Cars Overhead With Context (COWC)       	| 0.15m/px overhead imagery                  	| Object detection      	| VisionDataset 	| https://gdo152.llnl.gov/cowc/                        	|
| CV4A Kenya Crop Type                    	| Sentinel-2 (13 scene time series)          	| Semantic segmentation 	| GeoDataset    	| https://registry.mlhub.earth/10.34911/rdnt.dw605x/   	|
| Tropical Cyclone Wind Estimation        	| GOES single band imagery                   	| Regression            	| VisionDataset 	| http://registry.mlhub.earth/10.34911/rdnt.xs53up/    	|
| Landcover.ai                            	| RGB aerial imagery at 0.5m/px and 0.25m/px 	| Semantic segmentation 	| GeoDataset    	| https://landcover.ai/                                	|
| NWPU VHR-10                             	| Google Earth RGB and Vaihingen CIR         	| Object detection      	| VisionDataset 	| https://github.com/chaozhong2010/VHR-10_dataset_coco 	|
| SEN12MS                                 	| Sentinel-1 and Sentinel-2                  	| Semantic segmentation 	| GeoDataset    	| https://github.com/schmitt-muc/SEN12MS               	|
| So2Sat                                  	| Sentinel-1 and Sentinel-2                  	| Classification        	| VisionDataset 	| https://github.com/zhu-xlab/So2Sat-LCZ42             	|


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
