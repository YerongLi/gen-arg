import os 
import json 
import argparse 
import re 
from copy import deepcopy
from collections import defaultdict 
from tqdm import tqdm
import spacy 


from utils import load_ontology,find_arg_span, compute_f1, get_entity_span, find_head, WhitespaceTokenizer

nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

'''
Scorer for argument extraction on ACE & KAIROS.
For the RAMS dataset, the official scorer is used. 

Outputs: 
Head F1 
Coref F1 
'''
def clean_span(ex, span):
    tokens = ex['tokens']
    if tokens[span[0]].lower() in {'the', 'an', 'a'}:
        if span[0]!=span[1]:
            return (span[0]+1, span[1])
    return span 

def extract_args_from_template(ex, template, ontology_dict):

    def computeLPSArray(pat, M, lps):
        len = 0 # length of the previous longest prefix suffix
    
        lps[0] # lps[0] is always 0
        i = 1
    
        # the loop calculates lps[i] for i = 1 to M-1
        while i < M:
            if pat[i]== pat[len]:
                len += 1
                lps[i] = len
                i += 1
            else:
                # This is tricky. Consider the example.
                # AAACAAAA and i = 7. The idea is similar 
                # to search step.
                if len != 0:
                    len = lps[len-1]
    
                    # Also, note that we do not increment i here
                else:
                    lps[i] = 0
                    i += 1
    def KMPSearch(pat, txt):
        M = len(pat)
        N = len(txt)
    
        # create lps[] that will hold the longest prefix suffix 
        # values for pattern
        lps = [0]*M
        j = 0 # index for pat[]
    
        # Preprocess the pattern (calculate lps[] array)
        computeLPSArray(pat, M, lps)
    
        i = 0 # index for txt[]
        while i < N:
            if pat[j] == txt[i]:
                i += 1
                j += 1
    
            if j == M:
                print("Found pattern at index " + str(i-j))
                j = lps[j-1]
    
            # mismatch after j matches
            elif i < N and pat[j] != txt[i]:
                # Do not match lps[0..lps[j-1]] characters,
                # they will match anyway
                if j != 0:
                    j = lps[j-1]
                else:
                    i += 1
    # extract argument text
    if 'role_description' not in ontology_dict[ex['event']['event_type']]:
        template_words = template.strip().split()
        predicted_words = ex['predicted'].strip().split()    
        predicted_args = defaultdict(list) # each argname may have multiple participants 
        t_ptr= 0
        p_ptr= 0 
        evt_type = ex['event']['event_type']
        # print('event type')
        # print(evt_type)
        while t_ptr < len(template_words) and p_ptr < len(predicted_words):
            if re.match(r'<(arg\d+)>', template_words[t_ptr]):
                m = re.match(r'<(arg\d+)>', template_words[t_ptr])
                arg_num = m.group(1)
                try:
                    arg_name = ontology_dict[evt_type][arg_num]
                except KeyError:
                    print(evt_type)
                    exit() 

                if predicted_words[p_ptr] == '<arg>':
                    # missing argument
                    p_ptr +=1 
                    t_ptr +=1  
                else:
                    arg_start = p_ptr 
                    while (p_ptr < len(predicted_words)) and ((t_ptr== len(template_words)-1) or (predicted_words[p_ptr] != template_words[t_ptr+1])):
                        p_ptr+=1 
                    arg_text = predicted_words[arg_start:p_ptr]
                    print('arg_text')
                    print(arg_text)
                    predicted_args[arg_name].append(arg_text)
                    t_ptr+=1 
                    # aligned 
            else:
                t_ptr+=1 
                p_ptr+=1 
    else:
        template_words = template.strip().split()
        predicted_words = ex['predicted'].strip().split()    
        predicted_args = defaultdict(list) # each argname may have multiple participants 
        t_ptr= 0
        p_ptr= 0 
        evt_type = ex['event']['event_type']
        print('event type')
        print(evt_type)
        print('Before while loop')
        print(predicted_words)
        for role in ontology_dict[event_type]['roles']:
            KMPSearch(predicted_words, ontology_dict[event_type]['role_description'].split())

    return predicted_args







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen-file',type=str,default='checkpoints/gen-all-ACE-freq-pred/predictions.jsonl' )
    parser.add_argument('--test-file', type=str,default='data/ace/zs-freq-10/test.oneie.json')
    parser.add_argument('--coref-file', type=str)
    parser.add_argument('--head-only', action='store_true')
    parser.add_argument('--coref', action='store_true')
    parser.add_argument('--dataset',type=str, default='ACE', choices=['ACE', 'KAIROS','AIDA'])
    parser.add_argument('--ontology_file', type=str, default=None)
    
    args = parser.parse_args() 

    ontology_dict = load_ontology(dataset=args.dataset, ontology_file=args.ontology_file)

    if args.dataset == 'KAIROS' and args.coref and not args.coref_file:
        print('coreference file needed for the KAIROS dataset.')
        raise ValueError
    if args.dataset == 'AIDA' and args.coref:
        raise NotImplementedError

    examples = {}
    doc2ex = defaultdict(list) # a document contains multiple events 
    with open(args.gen_file,'r') as f:
        for lidx, line in enumerate(f): # this solution relies on keeping the exact same order 
            pred = json.loads(line.strip()) 
            examples[lidx] = {
                'predicted': pred['predicted'],
                'gold': pred['gold'],
                'doc_id': pred['doc_key']
            }
            doc2ex[pred['doc_key']].append(lidx)
    # print('doc2ex')
    # print(doc2ex)
    with open(args.test_file, 'r') as f:
        for line in f:
            doc = json.loads(line.strip())
            if 'sent_id' in doc.keys():
                doc_id = doc['sent_id']
                # print('evaluating on sentence level')
            else:
                doc_id = doc['doc_id']
                # print('evaluating on document level')
            for idx, eid in enumerate(doc2ex[doc_id]):
                examples[eid]['tokens'] = doc['tokens']
                examples[eid]['event'] = doc['event_mentions'][idx]
                examples[eid]['entity_mentions'] = doc['entity_mentions']
    
    coref_mapping = defaultdict(dict) # span to canonical entity_id mapping for each doc 
    if args.coref:
        if args.dataset == 'KAIROS' and args.coref_file:
            with open(args.coref_file, 'r') as f, open(args.test_file, 'r') as test_reader:
                for line, test_line  in zip(f, test_reader):
                    coref_ex = json.loads(line)
                    ex = json.loads(test_line)
                    doc_id = coref_ex['doc_key']
                    
                    for cluster, name in zip(coref_ex['clusters'], coref_ex['informative_mentions']):
                        canonical = cluster[0]
                        for ent_id in cluster:
                            ent_span = get_entity_span(ex, ent_id) 
                            ent_span = (ent_span[0], ent_span[1]-1) 
                            coref_mapping[doc_id][ent_span] = canonical
                    # this does not include singleton clusters 
        else:
            # for the ACE dataset 
            with open(args.test_file) as f:
                for line in f:
                    doc=json.loads(line.strip())
                    doc_id = doc['sent_id']
                    for entity in doc['entity_mentions']:
                        mention_id = entity['id']
                        ent_id = '-'.join(mention_id.split('-')[:-1]) 
                        coref_mapping[doc_id][(entity['start'], entity['end']-1)] = ent_id # all indexes are inclusive 

        

    pred_arg_num =0 
    gold_arg_num =0
    arg_idn_num =0 
    arg_class_num =0 

    arg_idn_coref_num =0
    arg_class_coref_num =0

    for ex in tqdm(list(examples.values())[:1]):
        context_words = ex['tokens']
        doc_id = ex['doc_id']
        doc = None 
        if args.head_only:
            doc = nlp(' '.join(context_words))
        
        # get template 
        evt_type = ex['event']['event_type']
        if evt_type not in ontology_dict:
            continue 
        template = ontology_dict[evt_type]['template']
        # extract argument text 
        print('ontology_dict')
        print(ontology_dict[evt_type])
        print('template')
        print(template)
        print(ontology_dict[evt_type].keys())
        # print('input ex')
        # print(ex)
        predicted_args = extract_args_from_template(ex,template, ontology_dict)
        role_description = {}
        if 'role_description' in ontology_dict[evt_type]:
            role_description = ontology_dict[evt_type]['role_description']
        print('role_description')
        print(role_description)
        print(ontology_dict[evt_type]['role_types'])

        print('predicted_args')
        print(predicted_args)
        # get trigger 
        # extract argument span
        trigger_start = ex['event']['trigger']['start']
        trigger_end = ex['event']['trigger']['end']
        
        predicted_set = set() 
        for argname in predicted_args:
            for entity in predicted_args[argname]:# this argument span is inclusive, FIXME: this might be problematic 
                arg_span = find_arg_span(entity, context_words, 
                    trigger_start, trigger_end, head_only=args.head_only, doc=doc) 
                
                if arg_span:# if None means hullucination
                    
                    predicted_set.add((arg_span[0], arg_span[1], evt_type, argname))

                else:
                    new_entity = []
                    for w in entity:
                        if w == 'and' and len(new_entity) >0:
                            arg_span = find_arg_span(new_entity, context_words, trigger_start, trigger_end,
                            head_only=args.head_only, doc=doc)
                            if arg_span: predicted_set.add((arg_span[0], arg_span[1], evt_type, argname))
                            new_entity = []
                        else:
                            new_entity.append(w)
                    
                    if len(new_entity) >0: # last entity
                        arg_span = find_arg_span(new_entity, context_words, trigger_start, trigger_end, 
                        head_only=args.head_only, doc=doc)
                        if arg_span: predicted_set.add((arg_span[0], arg_span[1], evt_type, argname))
                                  
        # get gold spans         
        gold_set = set() 
        gold_canonical_set = set() # set of canonical mention ids, singleton mentions will not be here 
        for arg in ex['event']['arguments']:
            argname = arg['role']
            entity_id = arg['entity_id']
            span = get_entity_span(ex, entity_id)
            span = (span[0], span[1]-1)
            span = clean_span(ex, span)
            # clean up span by removing `a` `the`
            if args.head_only and span[0]!=span[1]:
                span = find_head(span[0], span[1], doc=doc) 
            
            gold_set.add((span[0], span[1], evt_type, argname))
            if args.coref:
                if span in coref_mapping[doc_id]:
                    canonical_id = coref_mapping[doc_id][span]
                    gold_canonical_set.add((canonical_id, evt_type, argname))
        

        pred_arg_num += len(predicted_set)
        gold_arg_num += len(gold_set)
        # check matches
        for pred_arg in predicted_set:
            print(pred_arg)
            arg_start, arg_end, event_type, role = pred_arg
            gold_idn = {item for item in gold_set
                        if item[0] == arg_start and item[1] == arg_end
                        and item[2] == event_type}
            if gold_idn:
                arg_idn_num += 1
                gold_class = {item for item in gold_idn if item[-1] == role}
                if gold_class:
                    arg_class_num += 1
            elif args.coref:# check coref matches 
                arg_start, arg_end, event_type, role = pred_arg
                span = (arg_start, arg_end)
                if span in coref_mapping[doc_id]:
                    canonical_id = coref_mapping[doc_id][span]
                    gold_idn_coref = {item for item in gold_canonical_set 
                        if item[0] == canonical_id and item[1] == event_type}
                    if gold_idn_coref:
                        arg_idn_coref_num +=1 
                        gold_class_coref = {item for item in gold_idn_coref
                        if item[2] == role}
                        if gold_class_coref:
                            arg_class_coref_num +=1 
            
    # print(pred_arg_num)
    # print(gold_arg_num)
    # print(arg_idn_num)
        
    if args.head_only:
        print('Evaluation by matching head words only....')
    
    
    role_id_prec, role_id_rec, role_id_f = compute_f1(
        pred_arg_num, gold_arg_num, arg_idn_num)
    role_prec, role_rec, role_f = compute_f1(
        pred_arg_num, gold_arg_num, arg_class_num)

    
    print('Role identification: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
        role_id_prec * 100.0, role_id_rec * 100.0, role_id_f * 100.0))
    print('Role: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
        role_prec * 100.0, role_rec * 100.0, role_f * 100.0))

    if args.coref:
        role_id_prec, role_id_rec, role_id_f = compute_f1(
        pred_arg_num, gold_arg_num, arg_idn_num + arg_idn_coref_num)
        role_prec, role_rec, role_f = compute_f1(
            pred_arg_num, gold_arg_num, arg_class_num + arg_class_coref_num)

        
        print('Coref Role identification: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
            role_id_prec * 100.0, role_id_rec * 100.0, role_id_f * 100.0))
        print('Coref Role: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
            role_prec * 100.0, role_rec * 100.0, role_f * 100.0))



                    




