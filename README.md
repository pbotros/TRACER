# TRACER - 3D
TRACER-3D is a useful toolbox for visualizing the trajectories of recording electrodes(e.g Neuropixels) through different sub-regions of the rat brain. Anatomical delineations are referenced from the Waxholm Space atlas of the adult Sprague Dawley rat brain (https://www.nitrc.org/projects/whs-sd-atlas). There are three packages in the TRACKER toolbox. One is for locating electrode tracks in the brain post-hoc using histological images, one is for generating coordinates prior to surgery in order to target specific brain regions, and one for locating and visulize virus. So happy tracking! 

Please refer to the User Manual that is available in the repository for further instructions. 
Run pip install requirements.txt to install the neccessary packages as used during the creation of this toolbox.

```
conda create -n tracer python=3.8
conda activate tracer
```

TODO:
- You need lots of RAM: at least ~25GB
- pip3 install opencv-python-headless instead of pip3 install opencv-python; see https://forum.qt.io/topic/119109/using-pyqt5-with-opencv-python-cv2-causes-error-could-not-load-qt-platform-plugin-xcb-even-though-it-was-found/26


References:

Papp, Eszter A., et al. "Waxholm Space atlas of the Sprague Dawley rat brain." Neuroimage 97 (2014): 374-386.

Shamash, Philip, et al. "A tool for analyzing electrode tracks from slice histology." BioRxiv (2018): 447995.
