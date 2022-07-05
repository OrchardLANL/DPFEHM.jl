---
title: 'DPFEHM: a differentiable subsurface physics simulator'
tags:
  - Julia
  - hydrology
  - multiphase flow
  - transport
  - wave equation
authors:
  - name: Daniel O'Malley
    corresponding: true
    orcid: 0000-0003-0432-3088
    affiliation: 1
  - name: Sarah Y. Greer
    orcid: 0000-0001-6463-0296
    affiliation: "1, 2"
  - name: Aleksandra Pachalieva
    orcid: 0000-0003-1246-0410
    affiliation: 1
  - name: Wu Hao
    orcid: 0000-0002-9402-7401
    affiliation: 1
  - name: Dylan Harp
    orcid: 0000-0001-9777-8000
    affiliation: 3
  - name: Velimir V. Vesselinov
    orcid: 0000-0002-6222-0530
    affiliation: 4
affiliations:
 - name: Los Alamos National Laboratory, USA
   index: 1
 - name: Massachussets Institute of Technology, USA
   index: 2
 - name: The Freshwater Trust, USA
   index: 3
 - name: SmartTensors, LLC, USA
   index: 4
date: 30 June 2022
bibliography: paper.bib

---

# Summary

The Earth's subsurface is a key resource that provides energy via fossil fuels and geothermal sources, stores drinking water, and is used in the fight against climate change via carbon sequestration.
Simulating the physical processes that occur in the Earth's subsurface with computers enables better use of this resource.
`DPFEHM` is a Julia package that includes computer models with a focus on the Earth's subsurface, especially fluid flow, which is critical for the aforementioned applications.
`DPFEHM` is able to solve the groundwater flow equations (single phase flow), Richards equation (air/water), the advection-dispersion equation, and the 2d wave equation.
One of the key features of `DPFEHM` is that it supports automatic differentiation, so it can be integrated into machine learning workflows using frameworks such as Flux or PyTorch.


# Statement of need

Numerical models of subsurface flow and transport such as MODFLOW [@harbaugh2005modflow], FEHM [@zyvoloski1997summary], PFLOTRAN [@lichtner2015pflotran], etc. are ubiquitous, but these models cannot be efficiently integrated with machine learning frameworks.
They cannot because they do not support automatic differentation, which is needed for the gradient-based optimization methods that are ubiquitous in machine learning workflows.
An automatically-differentiable model like `DPFEHM` can be seamlessly integrated into these machine learning workflows.
This enables machine learning workflows with `DPFEHM` in the loop, e.g., to learn to manage pressure in a scenario where wastewater or carbon dioxide are being injected into the subsurface [@pachalieva2022physics].
It is additionally useful for non-machine learning workflows, because gradient calculations are also ubiqitous in more traditional workflows such as inverse analysis [@wu2022inverse] and uncertainty quantification [@betancourt2017conceptual].
Of course, it can also be used to efficiently simulate complex physics related to flow and transport in the subsurface [@greer2022comparison] without exploiting the differentiability very deeply.

An alternative to a differentiable numerical model is to use a differentiable machine learning model that is trained on data from a non-differentiable numerical model such as those listed above.
However, this approach has two major drawbacks.
First, such a machine learning model may be insufficiently trustworthy, depending on the application.
By contrast, `DPFEHM` uses a finite volume method that ensures local mass conservation, which is backed by rigorous accuracy guarantees.
Second, in scenarios with complex physics, it may be impossible to generate sufficient data to train an adequate machine learning model.
By contrast, `DPFEHM` does not require any training -- just a description of the equations, boundary conditions, etc. that define the problem.

`DPFEHM` was designed to be a research tool to explore the interface between numerical models and machine learning models.
To date, it has been used in several publications including [@greer2022comparison,@wu2022inverse,@pachalieva2022physics].
`DPFEHM` uses a two-point flux approximation finite volume scheme.
This means that an orthogonal grid is required to ensure convergence, similar to other codes such as FEHM and PFLOTRAN
Alternative codes such as Amanzi-ATS [@mercer2020amanzi], which use a more advanced mimetic finite difference discretization, do not require orthogonal meshes.
The performance advantage for `DPFEHM` over non-differentiable alternatives such as those mentioned previously comes in computing the gradient of a function that involves the solution of a subsurface physics equation.
In these settings, the cost of computing a gradient with `DPFEHM` is typically around the cost of running two physics simulations.
For non-differentiable models, the cost is equal to performing a number of simulations that is proportional to the number of parameters -- exorbitant when the number of parameters is large.
This is important for subsurface physics, because there is often one or more parameters at each node in the mesh.

# Installation

`DPFEHM` can be installed from within Julia by running `import Pkg; Pkg.add("DPFEHM")`.
The installation can subsequently be tested by running `Pkg.test("DPFEHM")`.

# Acknowledgements

This project was supported by Los Alamos National Laboratory LDRD project 20200575ECR.
S. Y. Greer acknowledges support from the United States Department of Energy through the Computational Science Graduate Fellowship (DOE CSGF) under grant number DE-SC0019323.

# References
