 ## Deep Feature Gaussian Processes for Single-Scene Aerosol Optical Depth Reconstruction
[Shengjie Liu](https://skrisliu.com), [Lu Zhang](https://luzhangstat.github.io/)

This is a GitHub repo at [https://github.com/skrisliu/dfgp](https://github.com/skrisliu/dfgp)


### Key Packages
	torch==2.0.1
	gpytorch==1.11

Tested on Python 3.9, Ubuntu 18.04.6 LTS, with 1080 Ti 11GB GPU. Running on CPU is possible but significantly slow. 


### Demo on MODIS-LA
	data/modis/im.npy         # features, 240*300*(11+xy)
	data/modis/aod.npy        # label, 240*300
	data/modis/trainmask.npy  # train mask, 240*300
	data/modis/testmask.npy   # test mask, 240*300
	data/modis/fea64.npy      # deep features, 240*300*64
	data/modis/cnn.pt         # trained CNN model

This repo includes the MODIS-LA data. A pre-trained CNN network is stored at "data/modis/cnn.pt", and pre-trained features at "data/modis/fea64.py".
 

For the EMIT-BJ data, download at ...


##### DFGP
	Python demo12_modis_dfgp.py


##### DFGPs
	Python demo12_modis_dfgps.py

##### Competitors
###### CNN 
	Python demo11_modis_cnn.py

Require networks.py and rscls.py to clip the image and load network



