from collections import defaultdict

class Parser:
    def __init__(self, file, connector):
        self.file = file
        self.connector = connector
        
    def ReadFile(self):
        ensemble_genes = defaultdict(list)
        ids_ = []
        f=open(self.file, "r")
        lines = f.readlines()
    
        for line in lines:
            ensemble = line.split("\t")
            ids = ensemble[1].rstrip(",\n").split(",")
            for id_ in ids:
                curr_id = id_.split("-")[0]
                ensemble_genes[ensemble[0]].append(curr_id)
                if curr_id != "interactions information" and curr_id != 'retracted':
                    ids_.append(int(curr_id))
            
        f.close()
        
        connectors = defaultdict(str)
        f=open(self.connector, "r")
        lines = f.readlines()

        for i in range(1, len(lines)):
            symbols = lines[i].split("\t")
            if(len(symbols) > 1):
                connectors[symbols[0].rstrip("\n")] = symbols[1].rstrip("\n")
            else:
                connectors[symbols[0]] = ""
        return ensemble_genes, ids_, connectors