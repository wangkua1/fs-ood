import csv
import matplotlib.pyplot as plt
import os 
import ipdb
import itertools

class CSVLogger():
    def __init__(self, every, fieldnames, resume=True, filename='log.csv'):
        self.every  = every
        self.filename = filename
        resume = os.path.exists(self.filename) and resume# resume
        if resume:
            self.csv_file = open(filename, 'a')
        else:
            self.csv_file = open(filename, 'w')
        self.fieldnames=fieldnames
        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        if not resume:
            self.writer.writeheader()
            self.csv_file.flush()
        self.time = 0

    def is_time(self):
        return self.time%self.every==0

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()


def plot_csv(csv_path, fig_path):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        dict_of_lists = {}
        ks = None
        for i, r in enumerate(reader):
            if i == 0:
                for k in r:
                    dict_of_lists[k] = []
                ks = r
            else:
                for _i, v in enumerate(r):
                    if v == '':
                        break
                    dict_of_lists[ks[_i]].append(float(v))
    fig = plt.figure()
    for k in dict_of_lists:
        if k == 'global_iteration' or len(dict_of_lists[k])==0:
            continue
        plt.clf()
        plt.plot(dict_of_lists['global_iteration'], dict_of_lists[k])
        plt.title(k)
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(fig_path), f'_{k}.png'), bbox_inches='tight', pad_inches=0)
    plt.close(fig)



    
        

