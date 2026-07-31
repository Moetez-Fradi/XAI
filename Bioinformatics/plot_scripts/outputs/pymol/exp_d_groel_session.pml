# Exp D PyMOL — meso vs thermo GroEL (RSA coloring)
# Pair: 1q3s_D vs 1jon_A | xq_spearman=0.1587850180095731
reinitialize
bg_color white
set cartoon_fancy_helices, 1

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
for obj, csvpath in [('meso_chain', r'plot_scripts/outputs/pymol/exp_d_1q3s_D_rsa.csv'), ('thermo_chain', r'plot_scripts/outputs/pymol/exp_d_1jon_A_rsa.csv')]:
    with open(csvpath) as f:
        for row in csv.DictReader(f):
            resi = row['resi']
            val = float(row['value'])
            cmd.alter(f'{obj} and resi {resi}', f'b={val}')
    cmd.sort(obj)
python end

spectrum b, blue_white_red, meso_chain, minimum=0, maximum=1
spectrum b, blue_white_red, thermo_chain, minimum=0, maximum=1
show cartoon, meso_chain thermo_chain
align thermo_chain, meso_chain
translate [-25,0,0], meso_chain
translate [25,0,0], thermo_chain
zoom meso_chain thermo_chain, 5

# Export: png plot_scripts/outputs/pymol/exp_d_groel_rsa.png, 2400, 1800
