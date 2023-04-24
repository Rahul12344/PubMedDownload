import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


def dist(c):
    if c['dataset'] == "VIP":
        df = pd.read_csv(c['filenames']['virus-annotation'], sep='\t')
        df = df[df['Virus(es)'].notna()]
        items = df['Virus(es)'].tolist()
    else: 
        df = pd.read_csv(c['filenames']['bacteria-annotation'], sep='\t')
        df = df[df['Species(s)'].notna()]
        items = df['Species(s)'].tolist()
    items = [item.lower() for item in items]
    items = [item.replace(" and ", ",") for item in items]
    items = [item.replace(" & ", ",") for item in items]
    items = [item.replace(" or ", ",") for item in items]
    items = [item.split(',') for item in items]
    items = [bacteria.strip() for item in items for bacteria in item]
    
    items = [item.replace("helicobacter pylori", "h. pylori") for item in items]
    items = [item.replace("bacillus calmette-guérin (bcg)", "bcg") for item in items]
    items = [item.replace("staphylococcal enterotoxin b (seb)", "seb") for item in items]
    items = [item.replace("mycobacterium bovis bacille calmette guérin (bcg)", "bcg") for item in items]
    items = [item.replace("group a streptococcal isolates", "group a streptococc") for item in items]
    items = [item.replace("legionella pneumophila pneumonia", "l. p. pneumonia") for item in items]
    items = [item.replace("salmonella enterica serovar typhimurium", "salmonella typhi") for item in items]
    
    counts = Counter(items)
    sorted_counts = sorted(counts.items(), key=lambda x:(x[1], x[0]), reverse=True)
    
    bacteria, num = [[i for i, _ in sorted_counts ], [j for _, j in sorted_counts]]
    bacteria = [bacterium.capitalize() for bacterium in bacteria]
    plt.bar(bacteria, num, color=[(1.0, 0.5, 0.0, 0.6)]*len(bacteria),  edgecolor='black')
    plt.xticks(rotation=90)
    plt.ylabel("Number of BIPs Containing Bacteria", rotation="vertical")
    plt.xlabel("Bacteria Name", rotation="horizontal")
    plt.show()
    
    