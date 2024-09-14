import levels
import csv

# Enumarate the file "levels" from this online website: https://www.mathsisfun.com/games/sokoban.html
# Convert each level to a csv file
for lvl_idx, lvl_map in enumerate(levels.levels):
    with open(f'Level{lvl_idx + 1}.csv', 'w') as f:
        wr = csv.writer(f)
        wr.writerows(lvl_map)
