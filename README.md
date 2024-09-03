# USV processing 

This programme processes ultrasound speckles (`Speckle_XXX.dat`) and config files (`config.mat`) based on Thomas Gallot's / Vincent Grenard's / Sébastien Manneville's scripts. 

There are four main programmes here : 

* `Process.ipynb` processes the data, including :
  * Making a Reference (using Numpy)
  * Beamforming (using Numba)
  * Computing the Hilbert transform to get the signal intensity (using Pytorch)
  * Computing the Displacements using cross-correlations (using Torchaudio)
  * Computes the velocity (using Numpy) if a calibration is provided

* `Benchmark.ipynb` provides side-by side comparisons between old and new data processing
* `Plot.ipynb` plots the data [for now ]
* `Calib.ipynb` creates a calibration file that will be used to compute the velocity

The code will benefit from CUDA capabilities for beamforming and [TODO] for computing the displacements. 