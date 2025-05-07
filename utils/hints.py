



from utils.gpt4_aug import collect_samples, filter_responses
import numpy as np
import pandas as pd
import os
import torch

def get_embs_for_sents(examples_train, embeddings, cluster_id) -> dict:
    sents_dct = {}
    emb_dct = {}
    embed_temp = {}

    for i in range(len(cluster_id)):
        if cluster_id[i] in sents_dct.keys():
            sents_dct[cluster_id[i]].append(examples_train[i])
            embed_temp[cluster_id[i]].append(embeddings[i])
        else:
            sents_dct[cluster_id[i]] = [examples_train[i]]
            embed_temp[cluster_id[i]] = [embeddings[i]]

    for label in sents_dct.keys():
        emb_dct[label] = {'emb': embed_temp[label], 'sent': sents_dct[label]}
    return emb_dct


def calculate_outliers(examples_train, embeddings, cluster_id) -> dict:
    embs_dct = get_embs_for_sents(examples_train, embeddings, cluster_id)
    mean_dct = {}
    pandas_dct = {'label': [], 'distance': [], 'text': []}

    # calculate mean vector per label
    for label in embs_dct:
        mean_dct[label] = np.array(embs_dct[label]['emb']).mean(axis=0)

    # calculate distance from the mean vector per label
    for label in embs_dct:
        mean_emb = mean_dct[label]
        for (sent_emb, sent) in zip(embs_dct[label]['emb'], embs_dct[label]['sent']):
            dist = np.linalg.norm(mean_emb - sent_emb)
            pandas_dct['label'].append(label)
            pandas_dct['distance'].append(dist)
            pandas_dct['text'].append(sent)
    return pd.DataFrame.from_dict(pandas_dct), embs_dct


def get_seed_sentences_per_labels(outliers_df, dct_phrases, no_samples=3, random=False) -> dict:
    dct_seeds_per_label = {}
    for label in dct_phrases.keys():
        #no_samples = len(dct_phrases[label])
        if random:
            sub_outlier_df = outliers_df[outliers_df['label'] == label].sample(frac=1)
        else:
            sub_outlier_df = outliers_df[outliers_df['label'] == label].sort_values(by=['distance'], ascending=False)
            #sub_outlier_df = outliers_df[outliers_df['label'] == label].sort_values(by=['distance'], ascending=True)
        dct_seeds_per_label[label] = list(sub_outlier_df.head(no_samples)['text'])
    return dct_seeds_per_label

# use random=True for ablated version
def get_hint_sentences_per_labels(df_outliers, no_samples, dct_phrases, random=False):
    dct_hints_per_sample = {}
    for label in dct_phrases.keys():
        for phrase in dct_phrases[label]:
            sub_df = df_outliers[df_outliers['seed'] == phrase]
            if random:
                sub_df = sub_df.sample(frac=1)
            else:
                sub_df = sub_df.sort_values(by=['distance'], ascending=False)
            dct_hints_per_sample[phrase] = list(sub_df.head(no_samples)['text'])
    return dct_hints_per_sample

def hint_aug(args, examples_train, embeddings, cluster_id):
    path = f"cluster_results/hint_responses_{args.cluster_type}_{args.n_clusters}_{args.train_number}_{args.aug_th}_{args.seed_sample_num}.pt"
    if os.path.exists(path):
        dct_responses = torch.load(path)
        print(f"Directly load responses from {path}")
        #print("all response", dct_responses)
    else:

        df_outliers, dct_phrases = calculate_outliers(examples_train, embeddings, cluster_id)

        # df_merged = df_outliers.merge(fb_0, how='inner', on='text').drop_duplicates()[
        #     ['label_x', 'text', 'distance', 'seed']]
        # df_merged = df_merged.rename(columns={'label_x': 'label'})

        dct_phrases = get_seed_sentences_per_labels(df_outliers, dct_phrases, no_samples=args.seed_sample_num)
        default_prompt = """Paraphrase this original question and statement 3 times with the same format as the original phrase. Original phrase: "{}".
        """



        dct_final_prompts = {}
        default_hint_prompt = '"{}".'

        for key in dct_phrases:
            dct_final_prompts[key] = []
            for hints in dct_phrases[key]:
                #hints = dct_hints_per_sample[phrase]
                # str_hints = []
                # for hint in hints:
                #     str_hints.append(default_hint_prompt.format(hint))
                # final_hint_str = "\n".join(str_hints)
                dct_final_prompts[key].append((default_prompt.format(hints)))

        # dct_final_prompts = {}
        #
        # for key in dct_phrases:
        #     dct_final_prompts[key] = []
        #     for phrase in dct_phrases[key]:
        #         dct_final_prompts[key].append((default_prompt.format(phrase), phrase))
        dct_responses = collect_samples(dct_final_prompts)
        torch.save(dct_responses,
                   f'cluster_results/hint_responses_{args.cluster_type}_{args.n_clusters}_{args.train_number}_{args.aug_th}_{args.seed_sample_num}.pt')
        print(
            f"save response to cluster_results/hint_responses_{args.cluster_type}_{args.n_clusters}_{args.train_number}_{args.aug_th}_{args.seed_sample_num}.pt.pt")
    fb_0 = filter_responses(dct_responses)
    #print(fb_0)
    new_sample = []
    for sentence in fb_0['text']:
        #print(sentence)
        new_sample.append(sentence)
    print("extended train num", len(new_sample))
    return new_sample
    #exit(0)


def seed_aug(args, dct_phrases):
    path = f"cluster_results/seed_responses_{args.cluster_type}_{args.n_clusters}_1000.pt"
    if os.path.exists(path):
        dct_responses = torch.load(path)
        print(f"Directly load responses from {path}")
        #print("all response", dct_responses)
    else:

        # df_outliers, dct_phrases = calculate_outliers(examples_train, embeddings, cluster_id)
        #
        # # df_merged = df_outliers.merge(fb_0, how='inner', on='text').drop_duplicates()[
        # #     ['label_x', 'text', 'distance', 'seed']]
        # # df_merged = df_merged.rename(columns={'label_x': 'label'})
        #
        # dct_phrases = get_seed_sentences_per_labels(df_outliers, dct_phrases, no_samples=args.seed_sample_num)
        default_prompt = """Paraphrase this original question and statement 3 times with the same format as the original phrase. Original phrase: "{}".
        """



        dct_final_prompts = {}
        default_hint_prompt = '"{}".'

        for key in dct_phrases:
            dct_final_prompts[key] = []
            for hints in dct_phrases[key]:
                #hints = dct_hints_per_sample[phrase]
                # str_hints = []
                # for hint in hints:
                #     str_hints.append(default_hint_prompt.format(hint))
                # final_hint_str = "\n".join(str_hints)
                dct_final_prompts[key].append((default_prompt.format(hints)))

        # dct_final_prompts = {}
        #
        # for key in dct_phrases:
        #     dct_final_prompts[key] = []
        #     for phrase in dct_phrases[key]:
        #         dct_final_prompts[key].append((default_prompt.format(phrase), phrase))
        dct_responses = collect_samples(dct_final_prompts)
        torch.save(dct_responses,
                   f'cluster_results/seed_responses_{args.cluster_type}_{args.n_clusters}_1000.pt')
        print(
            f"save response to cluster_results/seed_responses_{args.cluster_type}_{args.n_clusters}_1000.pt")
    fb_0 = filter_responses(dct_responses)
    #print(fb_0)
    new_sample = []
    for sentence in fb_0['text']:
        #print(sentence)
        new_sample.append(sentence)
    print("extended train num", len(new_sample))
    return new_sample
    #exit(0)


