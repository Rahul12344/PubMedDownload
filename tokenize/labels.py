from collections import defaultdict

class Labels:
    def __init__(self, export_file_path, VIP_file_path):
        self.VIP_file = open(VIP_file_path, "r")
        self.export_file = open(export_file_path, "r")
    
    def BuildLabeler(self):
        self.labels = defaultdict(int)
        lines = self.VIP_file.readlines()
    
        for line in lines:
            ensemble = line.split("\t")
            if(ensemble[0][0:4] == "ENSG"):
                self.labels[ensemble[0]] = 1
        
        lines = self.export_file.readlines()
        for line in lines:
            ensemble = line.split("\t")
            if(ensemble[0][0:4] == "ENSG"):
                if ensemble[0] not in self.labels:
                    self.labels[ensemble[0]] = 0
        
        return self.labels
    
        
    