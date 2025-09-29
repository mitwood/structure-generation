# Ni/Nb metallic glass example

## example for system with chemical and structural disorder

See supporting information in Goff,Mullen,Sta.,Yang,Wood  2024

## USAGE

To get the target distribution:

python get_target.py <target_structure>

example:

python get_target.py Ni62_Nb38_rate1e13_cnt10k_PeriodicCell.cif


And follow with GRS protocol script to find representative structures.

python GRS_protocol.py <target_structure> <num_represenative>

example:

python GRS_protocol.py Ni62_Nb38_rate1e13_cnt10k_PeriodicCell.cif 100
