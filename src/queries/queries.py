import requests
import xml.etree.ElementTree as ET
import logging
import csv
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

class PubMedQuery:
    def __init__(self, HGNC, valid_ids):
        self.prefix = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.HGNC = HGNC
        self.valid_ids = valid_ids

    def createQuery(self, hgnc, start):
        return self.prefix + "esearch.fcgi?db={0}&term={1}&retstart={2}&retmax=100000".format("pubmed", "virus", start)
    
    def Query(self):
        XML_ids = []
        self.valid_ids.sort()
        num_ones = 0
        smallest_id = self.valid_ids[0]
        largest_id = self.valid_ids[len(self.valid_ids)-1]
        """for hgnc in hgncs:
            for start in range(0, largest_id, 100000):
                r = requests.get(self.createQuery(hgnc, start))
                root = ET.fromstring(r.content)
                for IdList in root.findall('IdList'):
                    for ID in IdList.findall('Id'):
                        if int(ID.text) >= smallest_id and int(ID.text) <= largest_id:
                            XML_ids.append(ID.text)"""
        for start in range(0, largest_id, 100000):
            r = requests.get(self.createQuery(hgnc, start))
            root = ET.fromstring(r.content)
            for IdList in root.findall('IdList'):
                for ID in IdList.findall('Id'):
                    if int(ID.text) >= smallest_id and int(ID.text) <= largest_id:
                        XML_ids.append(ID.text)

        return XML_ids
    
    def QueryOutsiderange(self, hgncs):
        XML_ids = []
        self.valid_ids.sort()
        num_ones = 0
        smallest_id = self.valid_ids[0]
        largest_id = self.valid_ids[len(self.valid_ids)-1]
        limit = 10000
        num = 0
        start = 0
        r = requests.get(self.createQuery("hgnc", start))
        try:
            root = ET.fromstring(r.content)
            for IdList in root.findall('IdList'):
                for ID in IdList.findall('Id'):
                    if int(ID.text) > largest_id:
                        XML_ids.append(ID.text)
                        num += 1
                    if num >= limit:
                        print(len(XML_ids))
                        return XML_ids
        except Exception as e:
            print("err")   
        empty_response = False
        while root.find('RetMax') != None and root.find('RetMax').text != "0":
            start += 100000
            r = requests.get(self.createQuery("hgnc", start))
            try:
                root = ET.fromstring(r.content)
                for IdList in root.findall('IdList'):
                    for ID in IdList.findall('Id'):
                        if int(ID.text) > largest_id:
                            XML_ids.append(ID.text)
                            num += 1
            except Exception as e:
                print("err")
            if num >= limit:
                print(len(XML_ids))
                return XML_ids
            

        print(len(XML_ids))
        return XML_ids
     
def Query(uid, set_of_valid_ids, path, testing_threshold, validation_threshold):
    try:
        dataset = random.random()  
        r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db={0}&id={1}&retmode=text&rettype=abstract&api_key=49c77251ac91cbaa16ec5ae4269ab17d9d09".format("pubmed", uid))
        if int(uid) in set_of_valid_ids:
            if dataset <= testing_threshold:
                f = open("/Users/rahulnatarajan/Research/Garud/testing/pos/{0}.txt".format(str(uid)), "w")
                f.write(r.text)
                f.close()
            elif dataset <= testing_threshold + validation_threshold:
                f = open("/Users/rahulnatarajan/Research/Garud/validation/pos/{0}.txt".format(str(uid)), "w")
                f.write(r.text)
                f.close()
            else:
                f = open("/Users/rahulnatarajan/Research/Garud/training/pos/{0}.txt".format(str(uid)), "w")
                f.write(r.text)
                f.close()
            logger.info('Added abstracts related to {0}'.format(uid))
        else:
            if dataset <= testing_threshold:
                f = open("/Users/rahulnatarajan/Research/Garud/testing/neg/{0}.txt".format(str(uid)), "w")
                f.write(r.text)
                f.close()
            elif dataset <= testing_threshold + validation_threshold:
                f = open("/Users/rahulnatarajan/Research/Garud/validation/neg/{0}.txt".format(str(uid)), "w")
                f.write(r.text)
                f.close()
            else:
                f = open("/Users/rahulnatarajan/Research/Garud/training/neg/{0}.txt".format(str(uid)), "w")
                f.write(r.text)
                f.close()
            logger.info('Added abstracts related to {0}'.format(uid))
    except requests.HTTPError as e:
        logger.error('Error Code: {0}'.format(e.code))
        logger.error('Error: {0}'.format(e.read()))
        raise requests.HTTPError

def QueryNew(uid):
     try:
        r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db={0}&id={1}&retmode=text&rettype=abstract&api_key=49c77251ac91cbaa16ec5ae4269ab17d9d09".format("pubmed", uid))
        f = open("/Users/rahulnatarajan/Research/Garud/train/new_vips/{0}.txt".format(str(uid)), "w")
        #print(r.text)
        f.write(r.text)
        f.close()
     except requests.HTTPError as e:
        logger.error('Error Code: {0}'.format(e.code))
        logger.error('Error: {0}'.format(e.read()))
        raise requests.HTTPError
    
class MalariaQuery:
    def __init__(self, term_one, term_two):
        self.prefix = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.term_one = term_one
        self.term_two = term_two
        
    def createQuery(self):
        return self.prefix + "esearch.fcgi?db={0}&term={1}+AND+{2}&retstart=0&retmax=5000".format("pubmed", self.term_one, self.term_two)
    
    def Query(self):
        XML_ids = []
        r = requests.get(self.createQuery())
        root = ET.fromstring(r.content)
        for IdList in root.findall('IdList'):
            for ID in IdList.findall('Id'):
                XML_ids.append(ID.text)

        return XML_ids

def MalariaQueryForDataset(uid, set_of_valid_ids, path, testing_threshold, validation_threshold):
    try:
        dataset = random.random()  
        r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db={0}&id={1}&retmode=text&rettype=abstract&api_key=49c77251ac91cbaa16ec5ae4269ab17d9d09".format("pubmed", uid))
        if int(uid) in set_of_valid_ids:
            if dataset <= testing_threshold:
                f = open("/Users/rahulnatarajan/Research/Garud/malaria_testing/pos/{0}.txt".format(str(uid)), "w")
                f.write(r.text)
                f.close()
            elif dataset <= testing_threshold + validation_threshold:
                f = open("/Users/rahulnatarajan/Research/Garud/malaria_validation/pos/{0}.txt".format(str(uid)), "w")
                f.write(r.text)
                f.close()
            else:
                f = open("/Users/rahulnatarajan/Research/Garud/malaria_training/pos/{0}.txt".format(str(uid)), "w")
                f.write(r.text)
                f.close()
            logger.info('Added abstracts related to {0}'.format(uid))
        else:
            if dataset <= testing_threshold:
                f = open("/Users/rahulnatarajan/Research/Garud/malaria_testing/neg/{0}.txt".format(str(uid)), "w")
                f.write(r.text)
                f.close()
            elif dataset <= testing_threshold + validation_threshold:
                f = open("/Users/rahulnatarajan/Research/Garud/malaria_validation/neg/{0}.txt".format(str(uid)), "w")
                f.write(r.text)
                f.close()
            else:
                f = open("/Users/rahulnatarajan/Research/Garud/malaria_training/neg/{0}.txt".format(str(uid)), "w")
                f.write(r.text)
                f.close()
            logger.info('Added abstracts related to {0}'.format(uid))
    except requests.HTTPError as e:
        logger.error('Error Code: {0}'.format(e.code))
        logger.error('Error: {0}'.format(e.read()))
        raise requests.HTTPError 
    
class BacteriaQuery:
    def __init__(self, term_one, term_two):
        self.prefix = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.term_one = term_one
        self.term_two = term_two
        
    def createQuery(self): # bacteria+AND+[gene]+NOT+virus
        return self.prefix + "esearch.fcgi?db={0}&term=(bacteria[mesh]+AND+{1}+AND+human)+NOT+{2}[mesh]+NOT+mice&retstart=0&retmax=5000".format("pubmed", self.term_two, "virus")
    
    def Query(self):
        XML_ids = []
        try:
            r = requests.get(self.createQuery())
            root = ET.fromstring(r.content)
            for IdList in root.findall('IdList'):
                for ID in IdList.findall('Id'):
                    XML_ids.append(ID.text)

            return XML_ids
        except:
            return []


def BacteriaDatasetQuery(uid):
    try:
        r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db={0}&id={1}&retmode=text&rettype=abstract&api_key=49c77251ac91cbaa16ec5ae4269ab17d9d09".format("pubmed", uid))
        f = open("/Users/rahulnatarajan/Research/Garud/bacteria_new/{0}.txt".format(str(uid)), "w")
        f.write(r.text)
        f.close()
        logger.info('Added abstracts related to {0}'.format(uid))
    except requests.HTTPError as e:
        logger.error('Error Code: {0}'.format(e.code))
        logger.error('Error: {0}'.format(e.read()))
        raise requests.HTTPError 
    
def BacteriaPredictionDatasetQuery(uid, set_of_valid_ids, testing_threshold, validation_threshold):
    try:
        dataset = random.random()  
        r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db={0}&id={1}&retmode=text&rettype=abstract&api_key=49c77251ac91cbaa16ec5ae4269ab17d9d09".format("pubmed", uid))
        if uid in set_of_valid_ids:
            if dataset <= testing_threshold:
                f = open("/Users/rahulnatarajan/Research/Garud/train/dataset/updated_bacteria_test/pos/{0}.txt".format(str(uid)), "w")
                f.write(r.text)
                f.close()
            elif dataset <= testing_threshold + validation_threshold:
                f = open("/Users/rahulnatarajan/Research/Garud/train/dataset/updated_bacteria_validation/pos/{0}.txt".format(str(uid)), "w")
                f.write(r.text)
                f.close()
            else:
                f = open("/Users/rahulnatarajan/Research/Garud/train/dataset/updated_bacteria_train/pos/{0}.txt".format(str(uid)), "w")
                f.write(r.text)
                f.close()
            logger.info('Added abstracts related to {0}'.format(uid))
        else:
            if dataset <= testing_threshold:
                f = open("/Users/rahulnatarajan/Research/Garud/train/dataset/updated_bacteria_test/neg/{0}.txt".format(str(uid)), "w")
                f.write(r.text)
                f.close()
            elif dataset <= testing_threshold + validation_threshold:
                f = open("/Users/rahulnatarajan/Research/Garud/train/dataset/updated_bacteria_validation/neg/{0}.txt".format(str(uid)), "w")
                f.write(r.text)
                f.close()
            else:
                f = open("/Users/rahulnatarajan/Research/Garud/train/dataset/updated_bacteria_train/neg/{0}.txt".format(str(uid)), "w")
                f.write(r.text)
                f.close()
            logger.info('Added abstracts related to {0}'.format(uid))
    except requests.HTTPError as e:
        logger.error('Error Code: {0}'.format(e.code))
        logger.error('Error: {0}'.format(e.read()))
        raise requests.HTTPError 

