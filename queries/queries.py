import urllib2
import xml.etree.ElementTree as ET

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
            req = urllib2.Request(self.createQuery(start))
            r = urllib2.urlopen(req)
            root = ET.fromstring(r.read())
            for IdList in root.findall('IdList'):
                for ID in IdList.findall('Id'):
                    if int(ID.text) >= smallest_id and int(ID.text) <= largest_id:
                        XML_ids.append(ID.text)
        return XML_ids
    
#@backoff.on_exception(backoff.expo,
#                      (requests.exceptions.Timeout,
#                       requests.exceptions.HTTPError))   
def Query(uid):
    try:
        r = urllib2.urlopen("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db={0}&id={1}&retmode=text&rettype=abstract&api_key=49c77251ac91cbaa16ec5ae4269ab17d9d09".format("pubmed", uid))
        f = open("{0}.txt".format(str(uid)), "w")
        f.write(r.read())
        f.close()
        #print(r.read())
    except urllib2.HTTPError as e:
        print (e.code)
        print (e.read()) 
        
