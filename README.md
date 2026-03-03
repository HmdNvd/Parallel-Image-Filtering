# High-Resolution Real-Time Image Filtering and Convolution using Parallel and Distributed Computing Techniques

**Course:** Parallel and Distributed Computing (CS 4172) 
**Semester:** Fall 2025 

## 📌 Project Overview
Modern applications in medical imaging, satellite monitoring, and scientific visualization demand real-time image enhancement through computationally intensive filtering and convolution operations. Processing these ultra-high-resolution images requires billions of operations per second, making it a task beyond the capability of conventional serial computing. 

This project implements a scalable, parallelized image filtering and convolution pipeline capable of performing real-time edge detection and noise reduction. The solution leverages both shared-memory parallelism and distributed-memory processing to achieve substantial speedup over a serial baseline implementation.

## ⚙️ Features & Implementations
* **Serial Implementation:** Baseline implementation for performance comparison.
* **Shared-Memory Parallelism (OpenMP):** Utilizes multithreading to parallelize the convolution loops across CPU cores.
* **Distributed-Memory Processing (MPI):** Divides the image into chunks (rows) with overlapping halo regions, distributing the workload across multiple nodes/processes.
* **Filters Applied:** * Grayscale Conversion
  * Laplacian Filter (Edge Detection)
  * 3x3 Median Filter (Noise Reduction)

## 🚀 How to Compile and Run

### Prerequisites
* A C++ compiler (e.g., GCC)
* OpenMP installed
* MPI installed (e.g., MPICH or OpenMPI)
* An input image (e.g., `.png` or `.jpg`) in the same directory.

### 1. OpenMP Version
**Compile:**
`g++ -fopenmp openMP.cpp -o openmp_filter`

**Run:**
`./openmp_filter`
*(The program will prompt you to enter the image filename).*

### 2. MPI Version
**Compile:**
`mpic++ main.cpp -o mpi_filter`

**Run:**
`mpiexec -n 4 ./mpi_filter`
*(Replace `4` with the number of processes you wish to use).*




## 🛠️ Built With
* C++
* OpenMP
* MPI
* `stb_image` library for image loading/saving.
