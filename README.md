# Wavefield Simulations for Ice Penetrating Radar on Mars
This repository contains acoustic wavefield simulations designed to model radar sounding efforts performed by the MARSIS and SHARAD instruments over Mars' polar ice caps. This repository does not contain any true data or examples using real topography. Just the actual modeling scripts themselves.

## Scope
The following codes exist here written both for CPU and GPU:
- Acoustic cartesian 2D 
- Acoustic cartesian 3D
- Acoustic spherical 2D (polar)
- Acoustic spherical 3D
The following codes only exist for CPU:
- Elastic cartesian 3D
- Elastic cartesian 2D

## System Requirements
1. Installation of NVCC for compiling CUDA code.
2. NVIDIA toolkit and updated NVIDIA drivers
3. CUDA enabled GPU. (All tests were done with a GPU with 80GB of memory. It is unlikely that much is required, but higher cell count models require significantly more memory.)
4. Madagascar installation for compatablility with SConstructs and input/output .rsf and .vpl files.

## Directory Structure
* **CODE:** Contains all the c and CUDA scripts which generate the sf* program files using the SConstruct file. Additionally contains necessary, non-standard utilities written in c or CUDA.
* **acoustic_po:** Example for acoustic model in polar coordinates run on the CPU or GPU depending on system configuration.
* **acoustic_sp:** Example for acoustic model in spherical coordinates run on the CPU or GPU depending on system configuration.
* **awefd:**
  * **2d:** Example for acoustic model in 2D cartesian run on the CPU only.
  * **3d:** Example for acoustic model in 3D cartesian run on the CPU only.
* **cartesian:**
  * **2d:** Example for acoustic model in 2D cartesian run on the CPU or GPU depending on system configuration.
  * **3d:** Example for acoustic model in 3D cartesian run on the CPU or GPU depending on system configuration.
* **elastic:** Everything pertaining to the elastic model including c code and an example.
* **sponge:** Contains a matlab script to visualize different sponges for damping reflections.
