# Exp E PyMOL — query 1uwl_B vs neighbor 2fkn_C
# xq_enrichment=1.4936086646177695
reinitialize
bg_color white
load data/raw/pdb/1UWL.cif.gz, prot
remove prot and not chain B
create query_chain, prot
delete prot
python
import csv
from pymol import cmd
for csvpath, col in [(r'plot_scripts/outputs/pymol/exp_e_1uwl_B_attribution.csv', 'b'), (r'plot_scripts/outputs/pymol/exp_e_1uwl_B_site_mask.csv', 'q')]:
    with open(csvpath) as f:
        for row in csv.DictReader(f):
            cmd.alter(f'query_chain and resi {row["resi"]}', f'{col}={row["value"]}')
    cmd.sort('query_chain')
python end
spectrum b, yellow_red, query_chain, minimum=0, maximum=1
show cartoon, query_chain
color gray80, query_chain
show sticks, query_chain and q > 0.5
color tv_red, query_chain and q > 0.5
zoom query_chain, 8
# Export: png plot_scripts/outputs/pymol/exp_e_active_site.png, 2400, 1800
