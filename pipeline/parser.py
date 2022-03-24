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
        return ensemble_genes, ids_, connectors.values()
        
    def ReadMalariaFile(self):
        f=open(self.file, "r")
        lines = f.readlines()
        ids = []
        for line in lines:
            info = line.split(",")
            sub_ids = info[2].split("*")
            for sub_id in sub_ids:
                ids.append(sub_id[0:8])
        f.close()
        return ids
    
    def ReadMartExport(self):
        f=open(self.file, "r")
        lines = f.readlines()
        hgncs = []
        for i in range(1,len(lines)):
            info = lines[i].split("\t")
            if(info[1] != '\n'):
                hgncs.append(info[1].strip('\n'))
        f.close()
        return hgncs
    
    def ReadBacteriaFiles(self):
        f=open(self.file, "r")
        bacteria_true_positives, bacteria_true_negatives = [], []
        lines = f.readlines()
        ids = []
        for line in lines:
            print(line)
            info = line.split(",")
            if info[1] == "\"1\"\n":
                bacteria_true_positives.append(info[0])
            elif info[1] == "\"0\"\n":
                bacteria_true_negatives.append(info[0])
        f.close()
        return bacteria_true_positives, bacteria_true_negatives
    
    def ReadUpdatedBacteriaFiles(self):
        f=open("new_pos_assigned.csv", "r")
        f2=open("negatives_bacteria.csv", "r")
        bacteria_true_positives, bacteria_true_negatives = [], []
        lines = f.readlines()[1:]
        ids = []
        for line in lines:
            info = line.split(",")
            bacteria_true_positives.append(info[0].strip(".txt"))
        
        lines = f2.readlines()[1:]
        ids = []
        for line in lines:
            info = line.split(",")
            if info[0].strip(".txt") not in bacteria_true_positives:
                bacteria_true_negatives.append(info[0].strip(".txt"))
            
        f.close()
        f2.close()
        return bacteria_true_positives, bacteria_true_negatives
    
    def ModifyCleanedFile(self, bacteria_true_positives, bacteria_true_negatives):
        f=open("final_cleaned.csv")
        lines = f.readlines()
        added = False
        for pos in bacteria_true_positives:
            for i in range(len(lines)):
                if pos in lines[i]:
                    lines[i] = "{},\"1\"\n".format(pos)
                    added = True
            if not added:
                lines.append("{},\"1\"\n".format(pos))      
            added = False
        """for neg in bacteria_true_negatives:
            for i in range(len(lines)):
                if neg in lines[i]:
                    lines[i] = "{},\"0\"".format(neg)
                    added = True
            if not added:
                lines.append("{},\"1\"".format(neg))      
            added = False"""
            
        f.close()
            
        f=open("final_cleaned.csv", "w")
        f.writelines(lines)
        f.close()

        