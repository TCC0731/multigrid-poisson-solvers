---
name: use-conda-mg
description: Enforces the use of the `mg` conda environment when running Python code or terminal commands within the `multigrid-poisson-solvers` repository.
---
# Use Conda MG Skill

This skill ensures that all Python-related operations and terminal commands performed within the `multigrid-poisson-solvers` repository use the `mg` conda environment.

## Instructions

1. **Python Execution**: Whenever you need to run a Python script in this repository, always use:
   ```bash
   conda run -n mg python <path_to_script>
   ```
2. **Terminal Commands**: If a command requires the environment to be active (e.g., checking installed packages, running tests), use `conda run -n mg` as a prefix or ensure the environment is activated, but `conda run` is preferred for non-interactive commands.
3. **Consistency**: Do not use `python` or `pip` directly without the conda environment prefix unless explicitly instructed otherwise by the user for a specific one-off task.
4. **Context**: This rule applies to all subdirectories within `/home/tsc0731/CompAstr/multigrid-poisson-solvers`.
