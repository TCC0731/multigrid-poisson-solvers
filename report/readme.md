# Poisson Multigrid Solvers Report

This folder is the Beamer starter for the report on Poisson multigrid solvers.
It reuses the template structure from
`report/Geometric_Dynamics_of_Signal_Propagation_Predict_Trainability_of_Transformers`
and now focuses on SOR and, especially, multigrid.

## Current Files

- `slide.tex`: main presentation file and report outline
- `titlepage.tex`: title page and table of contents
- `userdefined.tex`: author, date, and helper definitions
- `style.sty`: shared theme and colors from the template
- `ref.bib`: bibliography placeholder for future citations

## Current Scope

The starter deck already includes sections for:

- problem setup
- SOR as the baseline solver
- multigrid as the main method
- result placeholders for SOR, MG, and CUDA
- next steps

## Build

Run `latexmk -xelatex slide.tex` from this directory. The `.latexmkrc` file sends all generated files to `build/`, so the report folder itself stays clean.

## Notes

- The deck currently avoids image dependencies so it can be expanded safely.
- Fill in `author`, `institute`, and any figures or references as the report grows.
