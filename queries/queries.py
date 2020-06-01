import requests
import xml.etree.ElementTree as ET
import logging
import csv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

class PubMedQuery:
    def __init__(self, HGNC, valid_ids):
        self.prefix = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.HGNC = HGNC
        self.valid_ids = valid_ids

    def createQuery(self, start):
        return self.prefix + "esearch.fcgi?db={0}&term={1}&retstart={2}&retmax=100000".format("pubmed", self.HGNC, start)
    
    def Query(self):
        XML_ids = []
        self.valid_ids.sort()
        smallest_id = self.valid_ids[0]
        largest_id = self.valid_ids[len(self.valid_ids)-1]
        for start in range(0, 1054817, 100000):
            r = requests.get(self.createQuery(start))
            root = ET.fromstring(r.content)
            for IdList in root.findall('IdList'):
                for ID in IdList.findall('Id'):
                    if int(ID.text) >= smallest_id and int(ID.text) <= largest_id:
                        XML_ids.append(ID.text)
        return XML_ids
     
def Query(uid, set_of_valid_ids, abstract_labels, file_labels):
    try:
        r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db={0}&id={1}&retmode=text&rettype=abstract&api_key=49c77251ac91cbaa16ec5ae4269ab17d9d09".format("pubmed", uid))
        output = " ".join(r.text.split("\n"))
        if uid in set_of_valid_ids:
            abstract_labels[1].append(output)
            file_labels[1].append(uid)
            print(abstract_labels)
        else:
            abstract_labels[0].append(output)
            file_labels[0].append(uid)
            print(abstract_labels)
        f = open("{0}.txt".format(str(uid)), "w")
        f.write(r.text)
        f.close()
    except requests.HTTPError as e:
        logger.error('Error Code: {0}'.format(e.code))
        logger.error('Error: {0}'.format(e.read()))
        raise requests.HTTPError 

