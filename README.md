# iWildCam_2019
Camera Traps (or Wild Cams) enable the automatic collection of large quantities of image data. Biologists all over the world use camera traps to monitor biodiversity and population density of animal species. We have recently been making strides towards automating the species classification challenge in camera traps, but as we try to expand the scope of these models from specific regions where we have collected training data to nearby areas we are faced with an interesting probem: how do you classify a species in a new region that you may not have seen in previous training data?

In order to tackle this problem, we have prepared a challenge where the training data and test data are from different regions, namely The American Southwest and the American Northwest. The species seen in each region overlap, but are not identical, and the challenge is to classify the test species correctly. To this end, we will allow training on our American Southwest data (from CaltechCameraTraps), on iNaturalist 2017/2018 data, and on simulated data generated from Microsoft AirSim. We have provided a taxonomy file mapping our classes into the iNat taxonomy.

This is an FGVCx competition as part of the FGVC6 workshop at CVPR 2019, and is sponsored by Microsoft AI for Earth. There is a github page for the competition [here](https://github.com/visipedia/iwildcam_comp). Please open an issue if you have questions or problems with the dataset.


## Data Overview
The training set contains 196,157 images from 138 different locations in Southern California. You may also choose to use supplemental training data from iNaturalist 2017, iNaturalist 2018, iNaturalist 2019, and images simulated with Microsoft AirSim. As a courtesy, we have curated all the images from iNaturalist 2017/2018 containing classes that might be in the test set and mapped them into the iWildCam categories. This data (which we call "iNat Idaho") can be downloaded from our git page [here](https://github.com/visipedia/iwildcam_comp).

The test set contains 153,730 images from 100 locations in Idaho.
