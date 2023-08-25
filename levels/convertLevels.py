import levels
import csv

for lvl_idx, lvl_map in enumerate(levels.levels):
    with open(f'Level{lvl_idx + 1}.csv', 'w') as f:
        wr = csv.writer(f)
        wr.writerows(lvl_map)
