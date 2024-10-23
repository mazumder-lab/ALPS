from modelutils import *
from datautils import *
import itertools
import random
import collections

from models_zeroshot import *

import itertools

from modelutils import *
import collections


from zero_shot_util import *
import models_zeroshot

DEV = torch.device('cuda:0')

import torch.nn.functional as F

import os




def smooth_crossentropy(pred, gold, smoothing=0.0):
    n_class = pred.size(1)
    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)
    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)







class RequestFactory:
    def __getattr__(self, attr):
        def fn(*args):
            return Request(attr, args)

        return fn



REQUEST_RETURN_LENGTHS = {
    "loglikelihood": 2,
    "greedy_until": None,
    "loglikelihood_rolling": None,
}


class Request:
    def __init__(self, request_type, args, index=None):
        if request_type not in REQUEST_RETURN_LENGTHS.keys():
            raise NotImplementedError(
                "The request type {} is not implemented!".format(request_type)
            )

        self.request_type = request_type
        self.args = args
        self.index = index

    def __iter__(self):
        if REQUEST_RETURN_LENGTHS[self.request_type] is None:
            raise IndexError("This request type does not return multiple arguments!")
        for i in range(REQUEST_RETURN_LENGTHS[self.request_type]):
            yield Request(self.request_type, self.args, i)

    def __getitem__(self, i):
        if REQUEST_RETURN_LENGTHS[self.request_type] is None:
            raise IndexError("This request type does not return multiple arguments!")
        return Request(self.request_type, self.args, i)

    def __eq__(self, other):
        return (
            self.request_type == other.request_type
            and self.args == other.args
            and self.index == other.index
        )

    def __repr__(self):
        return f"Req_{self.request_type}{self.args}[{self.index}]\n"



rf = RequestFactory()


def doc_to_text_lambada(doc):
    return preprocess_lambada(doc["text"].strip()).rsplit(" ", 1)[0]

def doc_to_target_lambada( doc):
        return " " + doc["text"].rsplit(" ", 1)[1]


def preprocess_lambada(text):
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("''", '"')
    text = text.replace("``", '"')
    return '\n' + text.strip()


def process_results_lambada( doc, results):
    ll, is_greedy = results[0]
    
    return {"ppl": ll, "acc": int(is_greedy)}


def mean(arr):
    return sum(arr) / len(arr)


def lambada(args):
    


    lm = models_zeroshot.get_model(args.model).create_from_arg_string({"args": args})
    from datasets import load_from_disk

    lambada_dir = os.path.join(args.data_path,'lambada')
    testloader = load_from_disk(lambada_dir)
    

    task_docs = list(testloader)
    rnd = random.Random()
    rnd.seed(args.seed)
    rnd.shuffle(task_docs)

    description = ""
    requests = collections.defaultdict(list)
    requests_origin = collections.defaultdict(list)
    docs = {}
    task_name = 'lambada'

    for doc_id, doc in enumerate(itertools.islice(task_docs, 0, None)):
        docs[(task_name, doc_id)] = doc
        reqs = rf.loglikelihood(doc_to_text_lambada(doc), doc_to_target_lambada(doc))
        if not isinstance(reqs, (list, tuple)):
                reqs = [reqs]
        for i, req in enumerate(reqs):
                requests[req.request_type].append(req)
                # i: index in requests for a single task instance
                # doc_id: unique id that we can get back to a doc using `docs`
                requests_origin[req.request_type].append((i, 'lambada', doc, doc_id))
    
    process_res_queue = collections.defaultdict(list)
    
    for reqtype, reqs in requests.items():
        resps = getattr(lm, reqtype)([req.args for req in reqs])
        resps = [
            x if req.index is None else x[req.index] for x, req in zip(resps, reqs)
        ]
        for resp, (i, task_name, doc, doc_id) in zip(resps, requests_origin[reqtype]):
            process_res_queue[(task_name, doc_id)].append((i, resp))
       



     
    vals = collections.defaultdict(list)

    # unpack results and sort back in order and return control to Task
    for (task_name, doc_id), requests in process_res_queue.items():
        requests.sort(key=lambda x: x[0])
        requests = [x[1] for x in requests]

        
        doc = docs[(task_name, doc_id)]





        metrics_dict = process_results_lambada(doc, requests)
        for metric, value in metrics_dict.items():
            vals[(task_name, metric)].append(value)

    acc = mean(vals[ ('lambada', 'acc')])

    outF = open(args.results_path, "w")
    outF.write(str(acc))
    outF.write("\n")
    outF.flush()

