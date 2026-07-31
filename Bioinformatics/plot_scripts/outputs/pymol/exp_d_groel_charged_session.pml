# Exp D PyMOL — charged exposed surface (red)
# Pair: 1q3s_D vs 1jon_A
reinitialize
bg_color white
load data/raw/pdb/1Q3S.cif.gz, meso
remove meso and not chain D
create meso_chain, meso
delete meso
load data/raw/pdb/1JON.cif.gz, thermo
remove thermo and not chain A
create thermo_chain, thermo
delete thermo
python
import csv
from pymol import cmd
for obj, csvpath in [('meso_chain', r'plot_scripts/outputs/pymol/exp_d_1q3s_D_charged.csv'), ('thermo_chain', r'plot_scripts/outputs/pymol/exp_d_1jon_A_charged.csv')]:
    with open(csvpath) as f:
        for row in csv.DictReader(f):
            resi = row['resi']
            val = float(row['value'])
            cmd.alter(f'{obj} and resi {resi}', f'b={val}')
    cmd.sort(obj)
python end
color gray80, meso_chain thermo_chain
spectrum b, white_red, meso_chain thermo_chain, minimum=0, maximum=1
show cartoon, meso_chain thermo_chain
align thermo_chain, meso_chain
translate [-25,0,0], meso_chain
translate [25,0,0], thermo_chain
zoom meso_chain thermo_chain, 5
# Export: png plot_scripts/outputs/pymol/exp_d_groel_charged.png, 2400, 1800
