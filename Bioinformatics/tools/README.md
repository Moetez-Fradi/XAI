# External tools

Installed by `fetch_tools_linux.sh` or `fetch_tools_macos.sh` into this directory (not committed at full size).

| Path | Contents |
|------|----------|
| `bin/micromamba` | Conda-style package manager |
| `mamba/envs/bio-tools/` | mkdssp, Foldseek, TM-align, PyMOL, … |
| `blast/` | Isolated BLAST+ prefix |
| `env_linux.sh` | `source` after fetch — extends `PATH` |

Activate bio-tools:

```bash
export PATH="/path/to/Bioinformatics/tools/mamba/envs/bio-tools/bin:$PATH"
export PATH="/path/to/Bioinformatics/tools/blast/bin:$PATH"
```

PyMOL (Exp D/E figures):

```bash
export PATH="$PWD/tools/mamba/envs/bio-tools/bin:$PATH"
pymol plot_scripts/outputs/pymol/exp_e_session.pml
```
