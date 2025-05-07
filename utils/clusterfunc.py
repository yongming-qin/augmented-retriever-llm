import argparse
import os
import json
import random
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans

from collections import defaultdict
from utils.inference import (
    run_embeddings,
    clustering_prompt
)

#https://github.com/wyu97/GenRead/blob/main/clusterfunc.py
def readfiles(infile):
    if infile.endswith('json'):
        lines = json.load(open(infile, 'r', encoding='utf8'))
    elif infile.endswith('jsonl'):
        lines = open(infile, 'r', encoding='utf8').readlines()
        lines = [json.loads(l) for l in lines]
    else:
        raise NotImplementedError

    if len(lines[0]) == 1 and lines[0].get('prompt'):
        lines = lines[1:]  ## skip prompt line

    return lines[:200]


''' step1: generate embeddings for each question-document pair'''


def step1(client, few_shot_examples, outfile=None, engine="gpt-4-32k", test=False, return_embeddings=False):



    # inlines = readfiles(infile)
    #
    # kept_idx = []
    # for idx, line in enumerate(inlines):
    #
    #     answers = line['answer']
    #     passage = line['output'][0]
    #
    #     if has_answer(answers, passage):
    #         kept_idx.append(idx)
    #
    # inlines = [l for i, l in enumerate(inlines) if i in kept_idx]
    #print(f'number of lines: {len(few_shot_examples)}')

    if outfile is not None:
        if os.path.exists(outfile):
            with open(outfile, 'r') as f:
                num_lines = len(f.readlines())
            outfile = open(outfile, 'a', encoding='utf8')
            # inlines = inlines[num_lines:]
        else:  # not os.path.exists(outfile)
            outfile = open(outfile, 'a', encoding='utf8')


    ## generate embeddings by batch
    # random.shuffle(inlines)
    index = 0
    if test == False:
        pbar = tqdm(total=len(few_shot_examples))
        pbar.update(index)
    embed = []

    while index < len(few_shot_examples):
        inputs, emb_inputs = [], []

        for _ in range(20):
            if index >= len(few_shot_examples): break

            line = few_shot_examples[index]
            inputs.append(line)
            # question = line['question']
            # passage = line['output'][0]
            # emb_input = ' '.join([question, passage])
            emb_inputs.append(line)
            index += 1
        #print("index", index)
        emebddings = run_embeddings(client, emb_inputs, engine)
        for i in range(len(emebddings)):
            embed.append(emebddings[i])
        #print("embedding shape", len(emebddings[0]))
        if test:
            return emebddings
        if outfile is not None:
            for line, emb in zip(inputs, emebddings):
                #line['embedding'] = emb
                outfile.write(json.dumps(emb) + '\n')

        pbar.update(20)

    pbar.close()
    if outfile is not None:
        outfile.close()
    if return_embeddings:
        return embed



''' step2: K-means clustering '''
def step2(infile, outfile, n_clusters=10):

    inlines = [json.loads(l) for l in open(infile, 'r').readlines()]
    matrix = np.vstack([l for l in inlines])
    print(f'embedding matrix: {matrix.shape}')
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    kmeans.fit(matrix)
    labels = kmeans.labels_

    assert len(inlines) == len(labels)
    # for line, label in zip(inlines, labels):
    #     line['label'] = str(label)
    #     del line['embedding']
    #
    # with open(outfile, 'w') as outfile:
    #     for line in inlines:
    #         outfile.write(json.dumps(line) + '\n')
    return kmeans, labels


''' step3: sample in-context demonstrations '''


def step3(infile, outfile, prompt):
    inlines = [json.loads(l) for l in open(infile, 'r').readlines()]

    cluster2examples = defaultdict(list)
    for _, line in enumerate(inlines):
        clusterid = line['label']
        cluster2examples[clusterid].append(line)

    with open(outfile, 'w') as outfile:
        for cid, ls in cluster2examples.items():
            random.shuffle(ls)
            cluster_prompt = clustering_prompt(ls[:5], prompt)
            outfile.write(json.dumps({
                'type': 'question answering',
                'task': 'step1',
                'pid': f'c-{cid}',
                'prompt': cluster_prompt,
            }) + '\n')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--dataset", default=None, type=str, required=True,
                        help="dataset name: [nq, tqa, webq, wizard, fever, fm2]",
                        )
    parser.add_argument("--engine", default='text-davinci-002', type=str, required=False,
                        help="text-davinci-002 (used in our experiments), code-davinci-002",
                        )
    parser.add_argument("--pid", default='1', type=str, required=False,
                        help="prompt id used in the first step, default=1",
                        )

    args = parser.parse_args()

    if args.dataset in ['nq', 'webq', 'tqa', 'twiki']:
        datatype = 'question answering'
    elif args.dataset in ['fever', 'fm2']:
        datatype = 'fact checking'
    elif args.dataset in ['wow']:
        datatype = 'dialogue system'
    else:  # other task type?
        raise NotImplementedError

    infolder = f'backgrounds-greedy-{args.engine}/{args.dataset}'
    infile = f'{infolder}/{args.dataset}-train-p{args.pid}.jsonl'
    outfolder = f'embeddings-greedy-{args.engine}/{args.dataset}'
    os.makedirs(outfolder, exist_ok=True)
    embfile = f'{outfolder}/{args.dataset}-train-embeddings.jsonl'
    clsfile = f'{outfolder}/{args.dataset}-train-clusters.jsonl'
    promptfile = f'{outfolder}/{args.dataset}-cluster-prompts.jsonl'

    step1(infile, embfile)  # step1: generate embeddings
    step2(embfile, clsfile)  # step2: k-means cluster

    promptlines = open(f'inprompts/regular.jsonl', 'r').readlines()
    for line in promptlines:
        line = json.loads(line)

        if line['type'] == datatype and line['task'] == 'step1':
            prompt = line['prompt']
            step3(clsfile, promptfile, prompt)
            break  ## only use the first prompt