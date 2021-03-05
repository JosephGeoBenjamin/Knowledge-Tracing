import json
import os
import csv
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class Conversions():
    def __init__(self, skill_f, student_f, question_f,
                    type_ = 'file'):
        """
        Pad Dummy Idx: 0
        New Case Idx: 1
        """

        self.skill2idx = {}
        self.stud2idx = {}
        self.ques2idx = {}

        self.sk_list = []
        self.st_list = []
        self.qs_list = []

        if type_ == 'file':
            self._update_list_from_file(skill_f, student_f, question_f)
        elif type_ == 'list':
            self._update_list(skill_f, student_f, question_f)
        else:
            raise Exception("Type unknown,  pass either list or File")

        self._create_index()

    def _create_index(self):
        self.skill2idx['msk'] = 0
        self.skill2idx['avg'] = 1
        for i, l in enumerate(self.sk_list):
            self.skill2idx[l] = i+2
        #---
        self.stud2idx['msk'] = 0
        self.stud2idx['avg'] = 1
        for i, l in enumerate(self.st_list):
            self.stud2idx[l] = i+2
        #---
        self.ques2idx['msk'] = 0
        self.ques2idx['avg'] = 1
        for i, l in enumerate(self.qs_list):
            self.ques2idx[l] = i+2

    def _update_list_from_file(self, skill_file, student_file, question_file):

        with open(question_file) as f:
            qs_list = f.readlines()
        self.qs_list = [s.rstrip() for s in qs_list]
        ##---
        with open(student_file) as f:
            st_list = f.readlines()
        self.st_list = [s.rstrip() for s in st_list]
        ##---
        with open(skill_file) as f:
            sk_list = f.readlines()
        self.sk_list = [s.rstrip() for s in sk_list]

    def _update_list(self, skill_list, student_list, question_list):
        if isinstance(skill_list,list) and isinstance(student_list,list) and isinstance(question_list,list):
            self.sk_list = skill_list
            self.st_list = student_list
            self.qs_list = question_list
        else:
            raise Exception("Wrong type data passed, expected list")


class AssistDataset(Dataset):
    """ INPUT JSON FORMAT
    [   stud_x001: {
            sessions:   [
                {   skill: skill_name,
                    interactions: [
                        {   qid: qx0001,
                            attempt_count: X,
                            first_response_ans: "1" or "0",
                            first_response_time: xxxKsec,
                            total_time: xxxKsec,
                        }, ...
                    ]
                }, ...
            ],
        }, ...
    ]
    """

    def __init__(self, json_path, skill_file, student_file, question_file):
        self.pad_size = 1000
        self.bpath = os.path.dirname(json_path)
        self.bname = os.path.basename(json_path)
        self.json_data = json.load(open(json_path))
        self.idxr = Conversions(skill_f=skill_file,
                                student_f=student_file,
                                question_f=question_file, )

        self.records = self._create_records()
        print(len(self.records)//4)
        if ( self.records.shape[0] % 4) != 0:
            print("\n !!!! Something went wrong with records, the lines are ODD  expected to be EVEN !!!!\n")

    def __getitem__(self, idx):  ## <---
        inp1 = self.records[idx*4]
        inp2 = self.records[(idx*4) +1]
        inp3 = self.records[(idx*4) +2]
        inp_sz = sum(inp1 >= 0)
        out = self.records[(idx*4) +3]

        return inp1, inp2, inp3, inp_sz, out

    def __len__(self):  ## <---
        return len(self.records) // 4

    def _expand_interaction(self, sess, stud):
        ita_list = sess['interactions']
        out_binary = []
        ques_l = []
        for it in ita_list:
            if it['first_response_ans'] == "1":
                out_binary.append(1)
                ques_l.append(self.idxr.ques2idx.get(it['qid'],1))
                continue

            att = int(it['attempt_count'])
            if att:
                att_l = [0] * att
                att_l[-1] = 1
                ques_l.extend( [self.idxr.ques2idx.get(it['qid'],1) ]*att )
                out_binary.extend(att_l)

        stud_l = [self.idxr.stud2idx.get(stud, 1)] * len(out_binary)
        skill_l = [self.idxr.skill2idx.get(sess['skill'], 1)] * len(out_binary)
        return stud_l, skill_l, ques_l, out_binary


    def _create_records(self):
        '''Return CSV reader Object
        '''
        final = []
        for stud in self.json_data.keys():
            # All sessions will be in single sequence
            a = []; b = []; c = []; d = []
            for se in self.json_data[stud]['sessions']:
                a_,b_,c_,d_ = self._expand_interaction(se, stud)
                a.extend(a_); b.extend(b_); c.extend(c_); d.extend(d_)
            if a:
                final.extend([a,b,c,d])

        arr = np.ones((len(final),self.pad_size), dtype=np.int) *-1

        for i, lst in enumerate(final):
           arr[i][:len(lst)] = lst
        path =  os.path.join(self.bpath, "temp_record_{}_npy.txt".format(self.bname))
        np.savetxt(path, arr, fmt='%i' )

        return arr


class GuviSplitDataset(Dataset):
    """
    """

    def __init__(self, csv_file):
        self.pad_size = 1000
        self.bpath = os.path.dirname(csv_file)
        self.bname = os.path.basename(csv_file)
        self.df = pd.read_csv(csv_file)
        self.idxr = Conversions(skill_f= self._get_id_list('skill_id'),
                                student_f=self._get_id_list('student_id'),
                                question_f=self._get_id_list('ques_id'),
                                type_='list')

        self.records = self._create_records()
        print(len(self.records)//4)
        if ( self.records.shape[0] % 4) != 0:
            print("\n !!!! Something went wrong with records, the lines are ODD  expected to be EVEN !!!!\n")

    def __getitem__(self, idx):  ## <---
        inp1 = self.records[idx*4]
        inp2 = self.records[(idx*4) +1]
        inp3 = self.records[(idx*4) +2]
        inp_sz = sum(inp1 >= 0)
        out = self.records[(idx*4) +3]

        return inp1, inp2, inp3, inp_sz, out

    def __len__(self):  ## <---
        return len(self.records) // 4

    def _get_id_list(self, column):
        return sorted(list(self.df[column].unique()))

    def _create_records(self):
        '''Return CSV reader Object
        '''
        final = []
        prev_stud = self.df['student_id'][0]
        a_ = []; b_ = []; c_ = []; d_ = []
        for index, row in self.df.iterrows():
            if prev_stud == row['student_id']:
                a_.append(row['student_id']); b_.append(row['skill_id'])
                c_.append(row['ques_id']); d_.append(row['correct'])
            else:
                final.append(a_); final.append(b_)
                final.append(c_); final.append(d_)
                a_ = [row['student_id']]; b_ = [row['skill_id']]
                c_ = [row['ques_id']]; d_ = [row['correct']]
            prev_stud = row['student_id']
        final.append(a_); final.append(b_); final.append(c_); final.append(d_)

        arr = np.ones((len(final), self.pad_size), dtype=np.int) *-1

        for i, lst in enumerate(final):
           arr[i][:len(lst)] = lst
        path =  os.path.join(self.bpath, "temp_record_{}_npy.txt".format(self.bname))
        np.savetxt(path, arr, fmt='%i' )

        return arr
