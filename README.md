# paper-rsf-vesselseg
This repository contains the source code for the paper "Generating Gigavoxel-Scale Microvascular Geometry Using GPU-Accelerated RSF Level Sets" (currently submitted).

This code is designed to binarize gigavoxel-scale 3D microscopy images of microvasculature collected using micro-CT, light sheet fluorescence microscopy (LSFM), and knife-edge scanning microscopy (KESM). The input is assumed to be an 8-bit stack of raw image files. The process requires four main steps with different software components:

1) The data set is divided into overlapping volumes (saved as NumPy files) that can be independently processed. This is done using the Python script "disassemble.py".
2) An initial contour for the network is created using a pre-trained U-Net convolutional neural network. We provide both the architecture as a Python script (unetseg.py). A link to the training data is provided as part of the entire data repository for the paper.
3) The contour is refined by evolving level set contour using a GPU-enabled RSF method. This is provided as a C++/CUDA executable that can be built using CMake.
4) The resulting level set contours are then re-assembled into a large NumPy file representing the level set and embedded isocontour for the vascular network. This is done using the Python script "reassemble.py".

## Disassemble Data

## Train U-Net and Segment Initial Contour
All of the raw data for this paper is available using Resilio Sync with the following key:
XXXXXXXXX

This repository has the following directory structure:

```
raw_data                                  // main directory storing complete data sets used in the paper
    ├── kesm_brain                        // mouse brain imaged using knife-edge scanning microscopy
    ├── lsfm_ovary_mouse_wythe            // mouse ovary imaged using light sheet microscopy
    ├── lsfm_brain_mouse_wythe            // mouse brain imaged using light sheet microscopy
    ├── microct_brain258_mouse_wythe      // brain images collected using micro-CT
    └── microct_brain420_mouse_wythe
training_data                 // main directory containing training/target pairs for U-Net
    ├── kesm_brain            // kesm data
        ├── ---.zip           // entire data set as a zipped image stack
        ├── input             // input training data for U-Net training (data that the network takes as input)
        └── target            // target data for U-Net training (data that the network tries to produce)
    ├── lsfm_ovary_mouse_wythe            // (all other directories follow this format for training/target pairs)
        ├── ---.zip           // entire data set as a zipped image stack
        ├── input
        └── target
    ├── lsfm_brain_mouse_wythe
        ├── ---.zip           // entire data set as a zipped image stack
        ├── input
        └── target
    ├── microct_brain258_mouse_wythe
        ├── ---.zip           // entire data set as a zipped image stack
        ├── input
        └── target
    └── microct_brain420_mouse_wythe
        ├── ---.zip           // entire data set as a zipped image stack
        ├── input
        └── target
```

## Parallel RSF Model
The code for running a parallel CUDA RSF level set model is provided as C++/CUDA source code that can be built using CMake. The following libraries are required (we recommend using vcpkg as a package manager for any operating system):

* library 1
* library 2
* TIRA Toolkit - this repository is maintained by our lab and available here: https://github.com/STIM-Lab/tiralib

After the rsf executable is compiled, the Python script "rsfbatch.py" will calculate the isocontours given an input directory (consisting of U-Net outputs) and a target directory (to store the isocontours).

## Re-Assemble Isocontours
The final step is to reassemble all of the contours into a single implicit isosurface. This is done using the "reassemble.py" script, which produces a large NumPy file.
