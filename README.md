# M1 GPUs for (scientific) computations in C++

In this repo, I explore the capabilities of the Metal Shading Language 
combined with C++, specifically for use of accelerating scientific codes.

This repo accompaines "Seamless GPU acceleration for C++ based physics using the M1's
unified processing units, a case study for elastic wave propagation and full-waveform
inversion". If you use VSCode, all instructions to compile can be found in `.vscode`.

I also wrote a few blog posts about the work in this repo:

- [Getting started](https://larsgeb.github.io/2022/04/20/m1-gpu.html)
- [SAXPY and FD](https://larsgeb.github.io/2022/04/22/m1-gpu.html)

However, if you just want to try out the code, make sure you install `llvm` 
and `libomp` using brew and Xcode and its command line tools. By cloning the 
repo and opening it in VSCode you should have all build configurations.

Additionally, I wrote a preprint (with the actual paper currently in peer-review) on how
to use MSL for modelling of PDEs [available on arXiv](https://arxiv.org/abs/2206.01791).
