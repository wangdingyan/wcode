from glob import glob
file_names = glob('/mnt/d/WDrugDataset/interim/*/*.txt')
for fil in file_names:
    with open(fil, 'r') as f:
        lines = f.readlines()
    for l in lines:
        if 'rna' in l and l.startswith(">"):
            print(fil, l.strip())

