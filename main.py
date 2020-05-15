from threading import Thread
import logging

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
    def __init__(self, q, pipe, HGNC_Parsing):
        Thread.__init__(self)
        self.pipe = pipe
        self.q = q
        self.valid_ids = HGNC_Parsing
        
    def query(self, curr_uid):
        queries.Query(curr_uid)
        
    def run(self):
        while True:
            curr_ = self.q.get()
            logger.info('Downloading abstracts related to {0}'.format(curr_))
            
            self.query(curr_)
            
            self.q.task_done()
            logger.info('Downloaded abstracts for {0}'.format(curr_))
            time.sleep(0.75)   
   
def main():
    labeler = labels.Labels("mart_export.txt", "VIPs_PMID_for_Rahul.txt")
    gene_labels = labeler.BuildLabeler()
    f = open("labels.txt", "w")
    for gene_label, value in gene_labels.items():
        f.write("{0}: {1}\n".format(gene_label, value))
    f.close()
    
    readFile = parser.Parser("VIPs_PMID_for_Rahul.txt", "mart_export.txt")
    ensemble_genes, HGNC_Parsing, connectors = readFile.ReadFile()
    
    
    curr_query = queries.PubMedQuery("virus", HGNC_Parsing)
    hgncs = curr_query.Query()
    
    pipe = pipeline.Pipeline()
    
    ts = time.time()
    q = queue.Queue(maxsize=0)
    for x in range(10):
        worker = DownloadWorker(q, pipe, HGNC_Parsing)
        worker.daemon = True
        worker.start()
    for hgnc in hgncs:
        logger.info('Queueing {0}'.format(hgnc))
        q.put(hgnc)
        
    q.join()
    logging.info('Downloaded {0} objects'.format(len(hgncs)))
    logging.info('Took %s s', time.time() - ts)

if __name__ == '__main__':
    main()