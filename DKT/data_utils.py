import json
import os
import csv
from torch.utils.data import Dataset
import numpy as np

class Conversions():
    def __init__(self, skill_file, student_file, question_file):

        self.skill2idx = {}
        self.stud2idx = {}
        self.ques2idx = {}

        self._create_index(skill_file, student_file, question_file)

    def _create_index(self, skill_file, student_file, question_file):

        with open(skill_file) as f:
            sk_list = f.readlines()
        self.sk_list = [s.rstrip() for s in sk_list]

        self.skill2idx['avg'] = 0
        for i, l in enumerate(self.sk_list):
            self.skill2idx[l] = i+1
        #---

        with open(student_file) as f:
            st_list = f.readlines()
        self.st_list = [s.rstrip() for s in st_list]

        self.stud2idx['avg'] = 0
        for i, l in enumerate(self.st_list):
            self.stud2idx[l] = i+1
        #---

        with open(question_file) as f:
            qs_list = f.readlines()
        self.qs_list = [s.rstrip() for s in qs_list]

        self.ques2idx['avg'] = 0
        for i, l in enumerate(self.qs_list):
            self.ques2idx[l] = i+1


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
        self.bpath = os.path.dirname(json_path)
        self.json_data = json.load(open(json_path))
        self.idxr = Conversions(skill_file=skill_file,
                                student_file=student_file,
                                question_file=question_file, )

        self.records = self._create_records()
        if ( self.records.shape[0] % 4) != 0:
            print("\n !!!! Something went wrong with records, the lines are ODD  expected to be EVEN !!!!\n")

    def __getitem__(self, idx):  ## <---
        inp1 = self.records[idx*4]
        inp2 = self.records[(idx*4) +1]
        inp3 = self.records[(idx*4) +2]
        inp_sz = sum(inp1 > 0)
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
                ques_l.append(self.idxr.ques2idx.get(it['qid'],0))
                continue

            att = int(it['attempt_count'])
            if att:
                att_l = [0] * att
                att_l[-1] = 1
                ques_l.extend( [self.idxr.ques2idx.get(it['qid'],0) ]*att )
                out_binary.extend(att_l)

        stud_l = [self.idxr.stud2idx.get(stud, 0)] * len(out_binary)
        skill_l = [self.idxr.skill2idx.get(sess['skill'], 0)] * len(out_binary)
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

        arr = np.ones((len(final),1000), dtype=np.int) *-1

        for i, lst in enumerate(final):
           arr[i][:len(lst)] = lst
        path =  os.path.join(self.bpath, "temp_record_train_npy.txt")
        np.savetxt(path, arr, fmt='%i' )

        return arr
