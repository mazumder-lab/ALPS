import time
import torch
import torch.nn as nn
from modelutils import *
from alps import *
import numpy as np
import os
import pandas as pd



def get_llama(model, path, cached = True):

    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto', cache_dir=path, local_files_only=cached)
    model.seqlen = 2048
    return model







@torch.no_grad()
def llama_sequential(model, dataloader, dev, nsamples=128):
    print('Starting ...')


    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    seqlen = model.seqlen

    print('Ready.')

    tot_params = 0
    tot_nnz = 0

    for i in range(len(layers)):


        layer = layers[i].to(dev)
        full = find_layers(layer)


        sequential = [list(full.keys())]
        scd = {}
        print('----')

        for names in sequential:
            subset = {n: full[n] for n in names}


            for name in subset:
                
                if args.method == 'ALPS':
                    scd[name] = ALPS_prune(subset[name], nsamples=nsamples, seqlen=seqlen)
                else:
                    raise Exception

            def add_batch(name):
                def tmp(_, inp, out):
                    scd[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
               
                if args.method == 'ALPS':
                    scd[name].ALPS_admm(sp=args.sp, nm_n=args.nm_n, nm_m=args.nm_m, rho=0.1)
               

                d1 = scd[name].layer.weight.data.shape[0]
                d2 = scd[name].layer.weight.data.shape[1]
                nnz = len( (scd[name].layer.weight.data.abs() > 0).nonzero(as_tuple=True)[0])
                tot_params += d1*d2
                tot_nnz += nnz



                scd[name].free()
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        
        layers[i] = layer.cpu()
        del layer
        del scd 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    


    return tot_nnz/tot_params








@torch.no_grad()
def llama_eval(model, testenc, dev):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)
        

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    

    model.config.use_cache = use_cache

    return ppl.item()





if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='Llama model to load; pass `meta-llama/Llama-2-7b-hf`.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )

    parser.add_argument(
        'method', type=str, choices=['ALPS'],
        help='Method to run.'
    )

    parser.add_argument(
        'sp', type=float, 
        help='Sparsity level'
    )

    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )

    parser.add_argument(
        '--model_path',
        type=str, default='./model', help='Path to the cached model.'
    )
    
    parser.add_argument(
        '--data_path',
        type=str, default='./data/', help='Path to the cached data.'
    )
    
    parser.add_argument(
        '--nm_n',
        type=int, default=0, help='N for N:M'
    )

    parser.add_argument(
        '--nm_m',
        type=int, default=0, help='M for N:M'
    )

    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )

    parser.add_argument(
        '--rho', type=float, default=300.0,
        help='initial rho'
    )

  

    

    args = parser.parse_args()


    model = get_llama(args.model, path = args.model_path) # Read the model
    model.eval() # Put to eval mode, no gradients are used.


    dataloader, testloader = get_loaders(
        args.dataset, data_path=args.data_path, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    ) # Calibration data

    tick = time.time()
    if args.method != 'Dense':
        sp = llama_sequential(model, dataloader, DEV, nsamples=args.nsamples)
    else:
        sp = args.sp
    runtime = time.time() - tick

    sp = 1 - sp


    datasets = ['wikitext2','ptb', 'c4'] # Test datasets
    results = {'Time': runtime, 'Sparsity': sp}
    for dataset in datasets: 
        dataloader, testloader = get_loaders(
            dataset, data_path=args.data_path, seed=args.seed, model=args.model, seqlen=model.seqlen, nsamples=args.nsamples
        ) 
        ppl = llama_eval(model, testloader, DEV) # Evaluate on test data
        results[dataset] = ppl

    df = pd.DataFrame(results, index=[0])
    if args.nm_n == 0:
        FILE = 'ppl_' + args.model.replace('/','--') + '_' + args.dataset + '_' + args.method + '_' + str(args.sp) + '_' + str(args.seed) + '_' + str(args.nsamples) + '_' + str(args.rho) 
    else:
        FILE = 'ppl_' + args.model.replace('/','--') + '_' + args.dataset + '_' + args.method + '_' + str(args.nm_n) + '_' + str(args.nm_m) + '_' + str(args.seed) + '_' + str(args.nsamples) + '_' + str(args.rho)

    filename = FILE  +'.csv'
    addr = os.path.join('./results',filename)
    df.to_csv(addr,index=False,)
    
    filename = FILE  +'.pt'
    addr = os.path.join('./pruned_models',filename)
    torch.save(model, addr)
    model = None
    

    args.chkpt = addr

    filename = 'zeroshot_' + FILE  +'.csv'
    args.results_path = os.path.join('./results',filename)
        
        



    from arc import *
    arc_challenge(args)
    file = open(args.results_path, 'r')
    rs = [line.rstrip('\n') for line in file][0]
    results['arc_challenge'] = rs
    
    
    from piqa import *
    piqa(args)
    file = open(args.results_path, 'r')
    rs = [line.rstrip('\n') for line in file][0]
    results['piqa'] = rs


    from arc_easy import *
    arc_easy(args)
    file = open(args.results_path, 'r')
    rs = [line.rstrip('\n') for line in file][0]
    results['arc_easy'] = rs


    df = pd.DataFrame(results, index=[0])
    filename = FILE  +'.csv'
    addr = os.path.join('./results',filename)
    df.to_csv(addr,index=False,)
    
    if os.path.exists(args.results_path):
        os.remove(args.results_path)



