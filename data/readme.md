# MODIS AOD, LA
modis/im.npy: 240*300*(11+2). 8 spectral bands, EVI, Impervious, Road Network, +xy coordinates
modis/aod.npy: 240*300. raw data, need to normalize to 0-1
modis/trainmask.npy: 240*300. Train mask, True=train data
modis/testmask.npy: 240*300. Test mask, True=test data
modis/fea64.npy: 240*300*64. Deep features extracted from a CNN
modis/cnn.pt: The CNN that generated the deep features


# EMIT AOD, BJ
emit/im.npy: 570*1040*(285+2). 285 spectral bands, +xy coordinates
emit/aod.npy: 570*1040. raw data, need to normalize to 0-1
emit/trainmask.npy: 570*1040. Train mask, True=train data
emit/testmask.npy: 570*1040. Test mask, True=test data
emit/fea64.npy: 240*300*64. Deep features extracted from a CNN
emit/cnn.pt: The CNN that generated the deep features