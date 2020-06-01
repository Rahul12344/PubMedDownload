from threading import Thread
import logging
import os.path
from os import path
import csv

try:
   import queue
except ImportError:
   import Queue as queue
   
import time

from queries import queries
from pipeline import pipeline
from pipeline import parser
from tokenize import labels

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

class DownloadWorker(Thread):   
    def __init__(self, q, pipe, HGNC_Parsing, f):
        Thread.__init__(self)
        self.pipe = pipe
        self.q = q
        self.valid_ids = HGNC_Parsing
        self.f = f
        
    def query(self, curr_uid):
        queries.Query(curr_uid, self.f, self.valid_ids)
        
    def run(self):
        while True:
            curr_ = self.q.get()
            logger.info('Downloading abstracts related to {0}'.format(curr_))
                
            try:
                if path.isfile("/u/scratch/r/rahul/PubMedDownload/{0}.txt".format(curr_)):
                    logger.info('Already downloaded {0}'.format(curr_))
                else:
                    self.query(curr_)
                logger.info('Downloaded abstracts related to {0}'.format(curr_))
                time.sleep(1)
            except Exception as e:
                logger.error("Failed to download:{0}".format(str(e)))
                self.q.put(curr_)
                logger.info('Re-queueing {0}'.format(curr_))
                time.sleep(1)
                self.q.task_done()

def main():
    labeler = labels.Labels(export_file_path="mart_export.txt", VIP_file_path="VIPs_PMID_for_Rahul.txt")
    gene_labels = labeler.BuildLabeler()
    f = open("labels.csv", "w")
    f.write("labels,values\n")
    
    readFile = parser.Parser("VIPs_PMID_for_Rahul.txt", "mart_export.txt")
    ensemble_genes, HGNC_Parsing, connectors = readFile.ReadFile()
    
    curr_query = queries.PubMedQuery("virus", HGNC_Parsing)
    hgncs = curr_query.Query()
    
    pipe = pipeline.Pipeline()
    
    ts = time.time()
    q = queue.Queue(maxsize=0)
    for x in range(10):
        worker = DownloadWorker(q, pipe, HGNC_Parsing, f)
        worker.daemon = True
        worker.start()
    for hgnc in hgncs:
        logger.info('Queueing {0}'.format(hgnc))
        q.put(hgnc)
        
    q.join()
    logging.info('Downloaded {0} objects'.format(len(hgncs)))
    logging.info('Took %s s', time.time() - ts)
    
    f.close()

if __name__ == '__main__':
    main()