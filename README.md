# MADS Milestone I Final Project

**Project Overview**

This repository presents the final project for the University of Michigan's Masters of Applied Data Science course, Milestone I. The project, conducted over five weeks, delves into the interplay of health and demographic factors as surveyed by the National Health and Nutrition Examination Survey (NHANES).

## Project Details

**Authors:** Josh Fisher, Nick Bermudez, Zhou Jiang

**Date:** February 15, 2024

### Dependencies

- Python version: 3.10.13
- All additional dependencies can be found in `requirements.txt`. 

If using Anaconda, you can create a virtual environment with the following commands:

```bash
conda create --name MADS_Milestone1 python=3.10.13
conda activate MADS_Milestone1
pip install -r <path to requirements.txt>
```

## Project Structure

- **main.ipynb:**
  - The main entry into the program is through `main.ipynb`, which is the final code submission for the Milestone project.

- **nhanes.py:**
  - Contains a variety of functions for manipulating the NHANES demographic and total nutrient datasets.

- **visualization.py:**
  - Contains modularized code for producing beautiful graphs.
  - Scripts that create NHANES specific visualizations.
  - Automation scripts for EDA.

- **collab_notebook.ipynb:**
  - A scratch notebook that we used to collaborate and develop ideas.
    - Left in for transparency.

## Usage Agreement

The data and analyses provided in this project are intended for educational and informational purposes only. Any conclusions drawn from this project should be interpreted with caution and may not be suitable for making medical or policy decisions without consulting appropriate experts or professionals. We strictly prohibit promoting false or misleading information regarding health, nutrition, or demographic groups.
