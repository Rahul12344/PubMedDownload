from threading import Thread
import logging
import os.path
from os import path
import csv
from collections import defaultdict
import argparse
import random

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
    def __init__(self, q, HGNC_Parsing, path, test_size, val_size, dataset):
        Thread.__init__(self)
        self.q = q
        self.valid_ids = HGNC_Parsing
        self.path = path
        self.test_size = test_size
        self.val_size = val_size
        self.dataset = dataset

    def query(self, curr_uid):
        if self.dataset == "VIP":
            queries.Query(uid=curr_uid, set_of_valid_ids=self.valid_ids, path=self.path, testing_threshold=self.test_size, validation_threshold=self.val_size)
        elif self.dataset == "malaria":
            queries.MalariaQueryForDataset(uid=curr_uid, set_of_valid_ids=self.valid_ids, path=self.path, testing_threshold=self.test_size, validation_threshold=self.val_size)
        elif self.dataset == "bacteria":
            queries.MalariaQueryForDataset(uid=curr_uid, set_of_valid_ids=self.valid_ids, path=self.path, testing_threshold=self.test_size, validation_threshold=self.val_size)
            
    def run(self):
        if self.dataset == "VIP":
            while True:
                curr_ = self.q.get()
                logger.info('Downloading abstracts related to {0}'.format(curr_))       
                try:
                    if path.isfile("{0}/validation/pos/{1}.txt".format(path, curr_)):
                        logger.info('Already downloaded {0}'.format(curr_))
                    if path.isfile("{0}/testing/pos/{1}.txt".format(path, curr_)):
                        logger.info('Already downloaded {0}'.format(curr_))
                    if path.isfile("{0}/training/pos/{1}.txt".format(path, curr_)):
                        logger.info('Already downloaded {0}'.format(curr_))
                    if path.isfile("{0}/validation/neg/{1}.txt".format(path, curr_)):
                        logger.info('Already downloaded {0}'.format(curr_))
                    if path.isfile("{0}/testing/neg/{1}.txt".format(path, curr_)):
                        logger.info('Already downloaded {0}'.format(curr_))
                    if path.isfile("{0}/training/neg/{1}.txt".format(path, curr_)):
                        logger.info('Already downloaded {0}'.format(curr_))
                    else:
                        self.query(str(curr_))
                    logger.info('Downloaded abstracts related to {0}'.format(curr_))
                    self.q.task_done()
                    time.sleep(1)
                except Exception as e:
                    logger.error("Failed to download:{0}".format(str(e)))
                    self.q.put(curr_)
                    logger.info('Re-queueing {0}'.format(curr_))
                    self.q.task_done()
        elif self.dataset == "malaria":
            while True:
                curr_ = self.q.get()
                logger.info('Downloading abstracts related to {0}'.format(curr_))       
                try:
                    if path.isfile("{0}/malaria_validation/{1}.txt".format(path, curr_)):
                        logger.info('Already downloaded {0}'.format(curr_))
                    if path.isfile("{0}/malaria_testing/{1}.txt".format(path, curr_)):
                        logger.info('Already downloaded {0}'.format(curr_))
                    if path.isfile("{0}/malaria_training/{1}.txt".format(path, curr_)):
                        logger.info('Already downloaded {0}'.format(curr_))
                    else:
                        self.query(str(curr_))
                    logger.info('Downloaded abstracts related to {0}'.format(curr_))
                    self.q.task_done()
                    time.sleep(1)
                except Exception as e:
                    logger.error("Failed to download:{0}".format(str(e)))
                    self.q.put(curr_)
                    logger.info('Re-queueing {0}'.format(curr_))
                    self.q.task_done()
        elif self.dataset == "bacteria":
            while True:
                curr_ = self.q.get()
                logger.info('Downloading abstracts related to {0}'.format(curr_))       
                try:
                    if path.isfile("{0}/bacteria_validation/{1}.txt".format(path, curr_)):
                        logger.info('Already downloaded {0}'.format(curr_))
                    if path.isfile("{0}/bacteria_testing/{1}.txt".format(path, curr_)):
                        logger.info('Already downloaded {0}'.format(curr_))
                    if path.isfile("{0}/bacteria_training/{1}.txt".format(path, curr_)):
                        logger.info('Already downloaded {0}'.format(curr_))
                    else:
                        self.query(str(curr_))
                    logger.info('Downloaded abstracts related to {0}'.format(curr_))
                    self.q.task_done()
                    time.sleep(1)
                except Exception as e:
                    logger.error("Failed to download:{0}".format(str(e)))
                    self.q.put(curr_)
                    logger.info('Re-queueing {0}'.format(curr_))
                    self.q.task_done()
                
def main():
    parsers = argparse.ArgumentParser(description='Download handler')
    parsers.add_argument('--val_size', type=float, default=0.2, help='Percentage of dataset to be in validation set')
    parsers.add_argument('--test_size',type=float, default=0.2, help='Percentage of dataset to be in test set')
    parsers.add_argument('--neg_size', type=int, default=3000, help='Number of non-true positives to be downloaded')
    parsers.add_argument('--pos_size', type=int, default=3000, help='Number of true positives to be downloaded')
    parsers.add_argument('--dataset', type=str, default=1.0, help='Dataset type [malaria/VIP]')
    parsers.add_argument('--path', type=str, default=".", help='File path to download to')
    args = parsers.parse_args()
    
    validation_set_percentage = float(args.val_size)
    test_set_percentage = float(args.test_size)
    pos_size = int(args.pos_size)
    neg_size = int(args.neg_size)
    file_path = args.path
    dataset = args.dataset
    
    labeler = labels.Labels(export_file_path="mart_export.txt", VIP_file_path="VIPs_PMID_for_Rahul.txt")
    gene_labels = labeler.BuildLabeler()
    
    if dataset == "VIP":
        readFile = parser.Parser("VIPs_PMID_for_Rahul.txt", "mart_export.txt")
        ensemble_genes, HGNC_Parsing, connectors = readFile.ReadFile()
    
        curr_query = queries.PubMedQuery("virus", HGNC_Parsing)
        hgncs = curr_query.Query()
            
        ts = time.time()
        q_pos = queue.Queue(maxsize=0)
        q_neg = queue.Queue(maxsize=0)
        
        num_pos = list(range(len(HGNC_Parsing)))
        random.shuffle(num_pos)
        num_total = list(range(len(hgncs)))
        
        random.shuffle(num_total)
    
        pos_counter = 0
        for x in range(10):
            worker = DownloadWorker(q_pos, HGNC_Parsing, file_path, test_set_percentage, validation_set_percentage, dataset)
            worker.daemon = True
            worker.start()
        while pos_counter < min(len(HGNC_Parsing), pos_size):
            logger.info('Queueing pos {0}'.format(HGNC_Parsing[num_pos[pos_counter]]))
            q_pos.put(str(HGNC_Parsing[num_pos[pos_counter]]))
            pos_counter += 1
            
        q_pos.join()
        
        neg_counter = 0
        num_negs = 0
        for x in range(10):
            worker = DownloadWorker(q_neg, HGNC_Parsing, file_path, test_set_percentage, validation_set_percentage, dataset)
            worker.daemon = True
            worker.start()
        while num_negs < min(neg_size, len(hgncs)):
            if int(hgncs[num_total[neg_counter]]) not in HGNC_Parsing:
                logger.info('Queueing neg {0}'.format(hgncs[num_total[neg_counter]]))
                q_neg.put(str(hgncs[num_total[neg_counter]]))
                num_negs += 1
            neg_counter += 1
        
        q_neg.join()
            
        logging.info('Downloaded {0} objects'.format(len(hgncs)))
        logging.info('Took %s s', time.time() - ts)
        
    elif dataset == "malaria":
        readFile = parser.Parser("journal.pgen.1007023.s006.csv", "")
        malaria_true_positives = readFile.ReadMalariaFile()

        curr_query = queries.MalariaQuery("malaria", "gene")
        malaria_neg_set = curr_query.Query()
        curr_query = queries.MalariaQuery("plasmodium", "gene")
        plasmodium_neg_set = curr_query.Query()
        malaria_neg_set.extend(plasmodium_neg_set)
        
        ts = time.time()
        q_pos = queue.Queue(maxsize=0)
        q_neg = queue.Queue(maxsize=0)
    
        pos_counter = 0
        for x in range(10):
            worker = DownloadWorker(q_pos, malaria_true_positives, file_path, test_set_percentage, validation_set_percentage, dataset)
            worker.daemon = True
            worker.start()
        for true_positive in malaria_true_positives:
            logger.info('Queueing pos {0}'.format(true_positive))
            q_pos.put(str(true_positive))
            
        q_pos.join()
        
        for x in range(10):
            worker = DownloadWorker(q_neg, malaria_true_positives, file_path, test_set_percentage, validation_set_percentage, dataset)
            worker.daemon = True
            worker.start()
        for true_negative in malaria_neg_set:
            logger.info('Queueing neg {0}'.format(true_negative))
            q_neg.put(str(true_negative))
        
        q_neg.join()
            
        logging.info('Downloaded {0} objects'.format(len(malaria_true_positives) + len(malaria_neg_set)))
        logging.info('Took %s s', time.time() - ts)
        
    elif dataset == "bacteria":
        readFile = parser.Parser("mart_export.txt", "")
        hgncs = readFile.ReadMartExport()

        bacteria_id_set = []
        for hgnc in hgncs:
            curr_query = queries.BacteriaQuery("bacteria", str(hgnc))
            curr_list = curr_query.Query()
            print(curr_list)
            bacteria_id_set.extend(curr_list)
        
        with open('pubmedids_for_bacteria.txt', 'w') as f:
            for item in bacteria_id_set:
                f.write("%s\n" % item)
        


if __name__ == '__main__':
    main()