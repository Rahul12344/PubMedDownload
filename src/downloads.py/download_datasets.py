from threading import Thread
import logging
import os
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
from tokenizer import labels

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

class DownloadWorker(Thread):   
    def __init__(self, q, valid_ids, path, test_size, val_size, dataset):
        Thread.__init__(self)
        self.q = q
        self.valid_ids = valid_ids
        self.path = path
        self.test_size = test_size
        self.val_size = val_size
        self.dataset = dataset

    def query(self, curr_uid):
        if self.dataset == "VIP":
            queries.Query(uid=curr_uid, set_of_valid_ids=self.valid_ids, path=self.path, testing_threshold=self.test_size, validation_threshold=self.val_size)
        elif self.dataset == "VIPextra":
            queries.QueryNew(uid=curr_uid)
        elif self.dataset == "malaria":
            queries.MalariaQueryForDataset(uid=curr_uid, set_of_valid_ids=self.valid_ids, path=self.path, testing_threshold=self.test_size, validation_threshold=self.val_size)
        elif self.dataset == "bacteria_dl":
            queries.BacteriaDatasetQuery(uid=curr_uid)
        elif self.dataset == "bacteria_prediction":
            queries.BacteriaPredictionDatasetQuery(uid=curr_uid, set_of_valid_ids=self.valid_ids, testing_threshold=self.test_size, validation_threshold=self.val_size)
        elif self.dataset == "bacteria_update":
            queries.BacteriaPredictionDatasetQuery(uid=curr_uid, set_of_valid_ids=self.valid_ids, testing_threshold=self.test_size, validation_threshold=self.val_size)
            
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
        elif self.dataset == "VIPextra":
            while True:
                curr_ = self.q.get()
                logger.info('Downloading abstracts related to {0}'.format(curr_))       
                try:
                    if path.isfile("{0}/new_vips/{1}.txt".format(path, curr_)):
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
        elif self.dataset == "bacteria_dl":
            while True:
                curr_ = self.q.get()
                logger.info('Downloading abstracts related to {0}'.format(curr_))       
                try:
                    if path.isfile("{0}/bacteria_new/{1}.txt".format(path, curr_)):
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
        elif self.dataset == "bacteria_prediction":
            while True:
                curr_ = self.q.get()
                logger.info('Downloading abstracts related to {0}'.format(curr_))       
                try:
                    if path.isfile("{0}/bacteria_validation/pos/{1}.txt".format(path, curr_)):
                        logger.info('Already downloaded {0}'.format(curr_))
                    if path.isfile("{0}/bacteria_testing/pos/{1}.txt".format(path, curr_)):
                        logger.info('Already downloaded {0}'.format(curr_))
                    if path.isfile("{0}/bacteria_training/pos/{1}.txt".format(path, curr_)):
                        logger.info('Already downloaded {0}'.format(curr_))
                    if path.isfile("{0}/bacteria_validation/neg/{1}.txt".format(path, curr_)):
                        logger.info('Already downloaded {0}'.format(curr_))
                    if path.isfile("{0}/bacteria_testing/neg/{1}.txt".format(path, curr_)):
                        logger.info('Already downloaded {0}'.format(curr_))
                    if path.isfile("{0}/bacteria_training/neg/{1}.txt".format(path, curr_)):
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
        elif self.dataset == "bacteria_update":
            while True:
                curr_ = self.q.get()
                logger.info('Downloading abstracts related to {0}'.format(curr_))       
                try:
                    if path.isfile("{0}/updated_bacteria_validation/pos/{1}.txt".format(path, curr_)):
                        logger.info('Already downloaded {0}'.format(curr_))
                    if path.isfile("{0}/updated_bacteria_test/pos/{1}.txt".format(path, curr_)):
                        logger.info('Already downloaded {0}'.format(curr_))
                    if path.isfile("{0}/updated_bacteria_train/pos/{1}.txt".format(path, curr_)):
                        logger.info('Already downloaded {0}'.format(curr_))
                    if path.isfile("{0}/updated_bacteria_validation/neg/{1}.txt".format(path, curr_)):
                        logger.info('Already downloaded {0}'.format(curr_))
                    if path.isfile("{0}/updated_bacteria_test/neg/{1}.txt".format(path, curr_)):
                        logger.info('Already downloaded {0}'.format(curr_))
                    if path.isfile("{0}/updated_bacteria_train/neg/{1}.txt".format(path, curr_)):
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
                    time.sleep(1)
                    self.q.task_done()                
def main():
    parsers = argparse.ArgumentParser(description='Download handler')
    parsers.add_argument('--val_size', type=float, default=0.2, help='Percentage of dataset to be in validation set')
    parsers.add_argument('--test_size',type=float, default=0.2, help='Percentage of dataset to be in test set')
    parsers.add_argument('--neg_size', type=int, default=3000, help='Number of non-true positives to be downloaded')
    parsers.add_argument('--pos_size', type=int, default=3000, help='Number of true positives to be downloaded')
    parsers.add_argument('--dataset', type=str, default="VIP", help='Dataset type [malaria/VIP/bacteria/bacteria_dl/bacteria_prediction/VIPextra/bacteria_update]')
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
        
        print(sorted(HGNC_Parsing))
    
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
        
    elif dataset == "VIPextra":
        readFile = parser.Parser("VIPs_PMID_for_Rahul.txt", "mart_export.txt")
        ensemble_genes, HGNC_Parsing, connectors = readFile.ReadFile()
        
        connectors = [i for i in connectors if i]
        
    
        curr_query = queries.PubMedQuery("virus", HGNC_Parsing)
        hgncs = curr_query.QueryOutsiderange(connectors)
                        
        ts = time.time()
        q = queue.Queue(maxsize=0)
    
        pos_counter = 0
        for x in range(10):
            worker = DownloadWorker(q, hgncs, file_path, test_set_percentage, validation_set_percentage, dataset)
            worker.daemon = True
            worker.start()
        for uid in hgncs:
            logger.info('Queueing pos {0}'.format(uid))
            q.put(str(uid))
            
        q.join()
                    
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
                
    elif dataset == "bacteria_dl":
        bacteria_id_set = []
        f = open("pubmedids_for_bacteria.txt")
        for line in f.readlines():
            bacteria_id_set.append(int(line))
        f.close()

        
        ts = time.time()
        q = queue.Queue(maxsize=0)
    
        for x in range(10):
            worker = DownloadWorker(q, bacteria_id_set, file_path, test_set_percentage, validation_set_percentage, "bacteria_dl")
            worker.daemon = True
            worker.start()
        for b_id in bacteria_id_set:
            logger.info('Queueing pos {0}'.format(b_id))
            q.put(str(b_id))
            
        q.join()            
        logging.info('Downloaded {0} objects'.format(len(bacteria_id_set)))
        logging.info('Took %s s', time.time() - ts)      
          
    elif dataset == "bacteria_prediction":
        readFile = parser.Parser("final_cleaned.csv", "")
        bacteria_true_positives, bacteria_true_negatives = readFile.ReadBacteriaFiles()
        print(bacteria_true_positives)
        print(bacteria_true_negatives)
        
        ts = time.time()
        q_pos = queue.Queue(maxsize=0)
        q_neg = queue.Queue(maxsize=0)
    
        pos_counter = 0
        for x in range(6):
            worker = DownloadWorker(q_pos, bacteria_true_positives, file_path, test_set_percentage, validation_set_percentage, dataset)
            worker.daemon = True
            worker.start()
        for true_positive in bacteria_true_positives:
            logger.info('Queueing pos {0}'.format(true_positive))
            q_pos.put(str(true_positive))
            
        q_pos.join()
        
        for x in range(6):
            worker = DownloadWorker(q_neg, bacteria_true_positives, file_path, test_set_percentage, validation_set_percentage, dataset)
            worker.daemon = True
            worker.start()
        for true_negative in bacteria_true_negatives:
            logger.info('Queueing neg {0}'.format(true_negative))
            q_neg.put(str(true_negative))
        
        q_neg.join()
        
        logging.info('Downloaded {0} objects'.format(len(bacteria_true_positives) + len(bacteria_true_negatives)))
        logging.info('Took %s s', time.time() - ts)
        
    elif dataset == "bacteria_update":
        readFile = parser.Parser("final_cleaned.csv", "")
        tps, _ = readFile.ReadUpdatedBacteriaFiles()
        readFile.ModifyCleanedFile(tps)
        
        bacteria_true_positives, bacteria_true_negatives = readFile.ReadBacteriaFiles()
        bacteria_true_positives = list(set(bacteria_true_positives))
        bacteria_true_negatives = list(set(bacteria_true_negatives))
        
        """ts = time.time()
        q_pos = queue.Queue(maxsize=0)
        q_neg = queue.Queue(maxsize=0)
    
        pos_counter = 0
        for _ in range(6):
            worker = DownloadWorker(q_pos, bacteria_true_positives, file_path, test_set_percentage, validation_set_percentage, dataset)
            worker.daemon = True
            worker.start()
        i = 0
        for true_positive in bacteria_true_positives:
            i += 1
            logger.info('Queueing pos {0}'.format(true_positive))
            logger.info('Queueing {}'.format(i))
            q_pos.put(str(true_positive))
            
        q_pos.join()
        
        for _ in range(6):
            worker = DownloadWorker(q_neg, bacteria_true_positives, file_path, test_set_percentage, validation_set_percentage, dataset)
            worker.daemon = True
            worker.start()
        for true_negative in bacteria_true_negatives:
            logger.info('Queueing neg {0}'.format(true_negative))
            q_neg.put(str(true_negative))
            
        q_neg.join()
        
        logging.info('Downloaded {0} objects'.format(len(bacteria_true_positives) + len(bacteria_true_negatives)))
        logging.info('Took %s s', time.time() - ts)"""
        

if __name__ == '__main__':
    main()
    
    