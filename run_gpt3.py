import os
import json
import argparse
import random
import time

from base_prompt import *
from model import *
from utilities import extract_prediction, normalize_answer, _strip_string, read_json_all, last_boxed_only_string
from algorithm import init_algorithm
from retriever.bm25_retriever import bm25_retrieve
from utils.clusterfunc import step1, step2, step3
from utils.hints import hint_aug, seed_aug
import numpy as np
import torch
import torch.nn.functional as F
import openai
import os
import copy

import httpx
from openai import AzureOpenAI
from retriever.bm25_retriever import bm25_retrieve

os.environ["APP_CLIENT_ID"] = "long-tail-knowledge-app"
os.environ["APP_CLIENT_SECRET"] = "zyAeZTrIEJli3cqyVD2jvTDia6Ua"

from llm_idam_token_generator.idam_token_generator import get_llm_access_token

# OpenAI Endpoint details
OPENAI_ENDPOINT = "https://openai-llm-frontdoor-hma7evbthrd4cugn.a01.azurefd.net"
OPENAI_DEPLOYMENT_MODEL = "gpt-4-32k-beta"
OPENAI_AZURE_API_VERSION = "2023-12-01-preview"
APIM_KEY = "8b96051ed6b84e4dad762fdc9f8c809e"


#openai.api_key = os.getenv("OPENAI_API_KEY")
#openai.api_key = "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"

client = AzureOpenAI(
            api_key="xxx",
            # This is not playing any role, but required as per OpenAI sdk. So any random could be passed.
            azure_endpoint=OPENAI_ENDPOINT,
            azure_deployment=OPENAI_DEPLOYMENT_MODEL,
            api_version=OPENAI_AZURE_API_VERSION,
            http_client=httpx.Client(verify=False),
            default_headers={
                'Authorization': f'Bearer {get_llm_access_token()}',
                'Content-Type': 'application/json',
                'Ocp-Apim-Subscription-Key': f'{APIM_KEY}'
            })


def load_data(args):
    if 'medqa' in args.data_root_test:
        problems_test = [json.loads(line) for line in open(args.data_root_test, 'r')]
        problems_train = [json.loads(line) for line in open(args.data_root_train, 'r')]
        problems = problems_test + problems_train
        test_pids = list(i for i in range(len(problems_test)))
        train_pids = list(i for i in range(len(problems_test), len(problems_test) + len(problems_train)))
        if args.data_root_vali is not None:
            problems_vali = [json.loads(line) for line in open(args.data_root_vali, 'r')]
            problems += problems_vali
    elif 'MATH' in args.data_root_test:
        problems_test = read_json_all(args.data_root_test)
        if 'pth' in args.data_root_train:
            problems_train = torch.load(args.data_root_train)
            # problems_train = problems_train[1744:]
        else:
            problems_train = read_json_all(args.data_root_train)
        # dup_data = torch.load("../data/MATH/train/algebra/problems_train_duplicate.pth")
        problems = problems_test + problems_train
        # print("original problem #",len(problems_train))
        # print("aug problem #", len(dup_data))
        # exit(0)
        # test problem ids
        test_pids = list(i for i in range(len(problems_test)))
        train_pids = list(i for i in range(len(problems_test), len(problems_test) + len(problems_train)))
    elif args.data_root_test in ['ethos-national_origin', 'blimp-anaphor_number_agreement', 'tweet_eval-irony', 'wino_grande', 'race-middle', 'proto_qa', 'race-high', 'kilt_hotpotqa', 'ade_corpus_v2-classification', 'hate_speech18', 'tweet_eval-stance_hillary', 'tweet_eval-offensive', 'kilt_ay2', 'squad-with_context', 'tweet_eval-sentiment', 'ethos-religion', 'wikisql', 'squad-no_context', 'ethos-race', 'tweet_eval-stance_climate', 'hate_speech_offensive', 'kilt_nq', 'tweet_eval-hate', 'tweet_eval-emotion', 'ethos-sexual_orientation', 'hatexplain', 'kilt_fever', 'blimp-ellipsis_n_bar_2', 'ethos-gender', 'ade_corpus_v2-dosage', 'ethos-disability', 'tweet_eval-stance_atheism', 'kilt_zsre', 'blimp-sentential_negation_npi_licensor_present', 'tweet_eval-stance_feminist', 'kilt_trex', 'tweet_eval-stance_abortion', 'ethos-directed_vs_generalized', 'blimp-sentential_negation_npi_scope', 'tweet_eval-emoji']:
        with open('../data/metaicl/task_data_splits.json', 'r', encoding='utf-8') as json_file:
            train_test_split = json.load(json_file)
        problems_test = train_test_split[args.data_root_test]['test']
        problems_train = train_test_split[args.data_root_test]['train']
        problems = problems_test + problems_train
        test_pids = list(i for i in range(len(problems_test)))
        train_pids = list(i for i in range(len(problems_test), len(problems_test) + len(problems_train)))
    else:
        problems_test = json.load(open(args.data_root_test))
        problems_train = json.load(open(args.data_root_train))
        problems = {**problems_test, **problems_train}

        # test problem ids
        test_pids = list(problems_test.keys())
        train_pids = list(problems_train.keys())
    if args.test_number < len(test_pids) and args.test_number > 0:
        test_pids = random.sample(test_pids, args.test_number)
    #test_pids = test_pids[:args.test_number] if args.test_number > 0 else test_pids
    if args.test_pids_ckpt:
        test_pids = torch.load(args.test_pids_ckpt)
    print(f"number of test problems: {len(test_pids)}\n")



    print(f"original cand set numbner {len(train_pids)}")
    if args.cand_ckpt:
        cand_pids = torch.load(args.cand_ckpt)
        if 'MATH' in args.data_root_test:
            cand_pids = [i + len(problems_test) for i in cand_pids]
    else:
        if args.cand_number < len(train_pids):
            cand_pids = random.sample(train_pids, args.cand_number)  # random sample
        else:
            cand_pids = train_pids
    cand_pids = [i for i in cand_pids if i not in test_pids]

    return problems, test_pids, cand_pids


def get_gpt_output(prompt, args):
    if "gpt4" in args.engine:
        # client = AzureOpenAI(
        #     api_key="xxx",
        #     # This is not playing any role, but required as per OpenAI sdk. So any random could be passed.
        #     azure_endpoint=OPENAI_ENDPOINT,
        #     azure_deployment=OPENAI_DEPLOYMENT_MODEL,
        #     api_version=OPENAI_AZURE_API_VERSION,
        #     http_client=httpx.Client(verify=False),
        #     default_headers={
        #         'Authorization': f'Bearer {get_llm_access_token()}',
        #         'Content-Type': 'application/json',
        #         'Ocp-Apim-Subscription-Key': f'{APIM_KEY}'
        #     })
        if 'pubmed' in args.data_root_test:
            user_prompt = "Please answer yes or no or maybe for the question."
        elif 'medqa' in args.data_root_test or args.data_root_test in ['ethos-national_origin',
                                                                           'blimp-anaphor_number_agreement',
                                                                           'tweet_eval-irony', 'wino_grande',
                                                                           'race-middle', 'proto_qa', 'race-high',
                                                                           'kilt_hotpotqa',
                                                                           'ade_corpus_v2-classification',
                                                                           'hate_speech18', 'tweet_eval-stance_hillary',
                                                                           'tweet_eval-offensive', 'kilt_ay2',
                                                                           'squad-with_context', 'tweet_eval-sentiment',
                                                                           'ethos-religion', 'wikisql',
                                                                           'squad-no_context', 'ethos-race',
                                                                           'tweet_eval-stance_climate',
                                                                           'hate_speech_offensive', 'kilt_nq',
                                                                           'tweet_eval-hate', 'tweet_eval-emotion',
                                                                           'ethos-sexual_orientation', 'hatexplain',
                                                                           'kilt_fever', 'blimp-ellipsis_n_bar_2',
                                                                           'ethos-gender', 'ade_corpus_v2-dosage',
                                                                           'ethos-disability',
                                                                           'tweet_eval-stance_atheism', 'kilt_zsre',
                                                                           'blimp-sentential_negation_npi_licensor_present',
                                                                           'tweet_eval-stance_feminist', 'kilt_trex',
                                                                           'tweet_eval-stance_abortion',
                                                                           'ethos-directed_vs_generalized',
                                                                           'blimp-sentential_negation_npi_scope',
                                                                           'tweet_eval-emoji']:
            user_prompt = "Please choose from all the options follow the given example."
        else:
            user_prompt = "Follow the given examples and answer the question following the same format."
        response = client.chat.completions.create(
            model=OPENAI_DEPLOYMENT_MODEL,
            # The deployment name you chose when you deployed the GPT-35-Turbo or GPT-4 model.
            messages=[{
                "role": "system", "content": prompt
            },
                {
        "role": "user", "content": user_prompt
    }],
            temperature=0.0, max_tokens=args.max_tokens, top_p=1, frequency_penalty=args.frequency_penalty, presence_penalty=args.presence_penalty)
        output = response.choices[0].message.content
        if output is not None:
            if output.startswith("\n\n"):
                output = output[2:]
            output = output.split("\n")[0]
    else:
        response = openai.Completion.create(engine=args.engine,
                                            prompt=prompt,
                                            temperature=args.temperature,
                                            max_tokens=args.max_tokens,
                                            top_p=args.top_p,
                                            frequency_penalty=args.frequency_penalty,
                                            presence_penalty=args.presence_penalty,
                                            stop=["\n"])
        output = response["choices"][0]["text"].strip()


    return output


def get_result_file(args):
    result_path = f"{args.output_root}/{args.model}"
    os.makedirs(result_path, exist_ok=True)

    result_file = "{}/{}_{}_{}_{}_seed_{}.json".format(result_path, args.label, args.test_split, args.prompt_format,
                                                       args.shot_number, args.seed)

    return result_file


def save_results(result_file, acc, correct, count, cand_pids, args, results):
    data = {}
    data['acc'] = acc
    data['correct'] = correct
    data['count'] = count
    data['cand_pids'] = cand_pids
    data['args'] = vars(args)
    data['results'] = results

    with open(result_file, 'w') as f:
        json.dump(data, f, indent=2, separators=(',', ': '))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_train', type=str, default='../data/tabmwp/problems_train.json')
    parser.add_argument('--data_root_test', type=str, default='../data/tabmwp/problems_test.json')
    parser.add_argument('--data_root_vali', type=str, default=None)
    parser.add_argument('--output_root', type=str, default='../results')
    parser.add_argument('--model', type=str, default='gpt3_rl')
    parser.add_argument('--option_inds', type=list,
                        default=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q",
                                 "R", "S", "T", "U", "V", "W", "X", "Y", "Z"])
    parser.add_argument('--batch_size',
                        type=int,
                        default=50,
                        help='Policy network training batch size. Set to train_number by default.')

    # user options
    parser.add_argument('--label', type=str, default='exp0')
    parser.add_argument('--test_split', type=str, default='test', choices=['dev', 'dev1k', 'test', 'test1k'])
    parser.add_argument('--test_number', type=int, default=500, help='GPT-3 is expensive. -1 for the whole test set')
    parser.add_argument('--save_every', type=int, default=10, help='Save the result with every n examples.')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument(
        '--prompt_format',
        type=str,
        default='Q-A',
        choices=['T-A', 'Q-A', 'Q-AS', 'Q-SA', 'TQ-A', 'TQ-AS', 'TQ-SA', 'QT-A', 'QT-AS', 'QT-SA', 'QTS-A', 'TQS-A', 'SQ-A', 'SQ-'],
        help='prompt format template')
    parser.add_argument('--shot_number', type=int, default=2, help='Number of n-shot training examples.')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    # GPT-3 settings
    parser.add_argument('--engine', type=str, default='gpt4', choices=['text-davinci-002', 'ada', 'gpt4'])
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_tokens',
                        type=int,
                        default=512,
                        help='The maximum number of tokens allowed for the generated answer.')
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)

    # Policy Model settings
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--model_config',
                        type=str,
                        default='bert-base-uncased',
                        choices=['distilbert-base-uncased', 'bert-base-uncased'])
    parser.add_argument('--cand_number', type=int, default=100, help='Number of candidate prompts.')
    parser.add_argument('--embedding_size', type=int, default=128, help='Policy network final layer hidden state size.')
    parser.add_argument('--ckpt_root', type=str, default='../checkpoints')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--ckpt_context', type=str, default=None)
    parser.add_argument('--ckpt_lossnet', type=str, default=None)
    parser.add_argument('--cand_ckpt', type=str, default=None, help="cand_pids root. cluster_results/cand_erm7_140.pt for pubmed")
    parser.add_argument('--test_pids_ckpt', type=str, default=None, help="test pids. cluster_results/false_medqa_val_Q-A_0_random_seed_1.pth")
    parser.add_argument('--train_ckpt', type=str, default=None,
                        help="cand_pids root. pubmed: cluster_results/wrong_80_1000.pt. medqa: cluster_results/wrong_cluster_medqa_40_500.pt")
    parser.add_argument('--val_ckpt', type=str, default=None,
                        help="cand_pids root. pubmed: cluster_results/val_80_1000.pt. medqa: cluster_results/val_cluster_medqa_40_500.pt")

    # Method
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate of policy network.')
    parser.add_argument('--algorithm', type=str, default='ERM',
                        choices=['ERM', 'DRNN', 'ARM-CML', 'ARM-BN', 'ARM-LL', 'DANN', 'MMD'])
    parser.add_argument('--adapt_bn', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=0.0, help='hyperparameter for MMD regularization.')
    parser.add_argument('--score_th', type=float, default=-1, help='threshold for retriever score..')

    #aug
    parser.add_argument('--cluster_type', type=str, default=None, help="kmeans")
    parser.add_argument('--n_clusters', type=int, default=5, help="The number of clusters.")
    parser.add_argument('--aug_method', type=str, default=None, help="hints/seed")
    parser.add_argument('--preselection', action='store_true')
    parser.add_argument('--select_number', type=int, default=50, help='select_number from preselection pool.')
    parser.add_argument('--get_false_test_pids', action='store_true', help='get the wrong/right number from results')

    args = parser.parse_args()
    args.meta_batch_size = args.batch_size
    return args


if __name__ == '__main__':

    args = parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    # https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  # CPU random seed
    torch.cuda.manual_seed(args.seed)  # GPU random seed
    torch.backends.cudnn.benchmark = True

    algorithm = init_algorithm(args)
    # problems, test question ids, candidate prompt pids, RL training pids
    problems, pids, cand_pids = load_data(args)
    # #extract mmd for problems from different domain (remove later)
    # feature_model = SentenceTransformer('all-MiniLM-L6-v2')
    # domain_num = int(len(cand_pids)/args.cand_number)
    # print('domain_num', domain_num)
    # for i in range(domain_num):
    #     Problem = []
    #     for j in range(i*args.cand_number, (i+1)*args.cand_number):
    #         Problem.append(problems[cand_pids[j]]['problem'])
    #     embedding_all  = feature_model.encode(Problem, convert_to_tensor=True)
    #     embedding_mean = torch.mean(embedding_all, dim=0)
    #     print('embedding_all size', embedding_all.shape)
    #     torch.save(embedding_all, f'../distribution/domain{str(i)}.pth')
    # Problem = []
    # for j in range(len(pids)):
    #     Problem.append(problems[pids[j]]['problem'])
    # embedding_all = feature_model.encode(Problem, convert_to_tensor=True)
    # embedding_mean = torch.mean(embedding_all, dim=0)
    # print('embedding_all size', embedding_all.shape)
    # torch.save(embedding_all, f'../distribution/test_domain.pth')
    #
    #
    # exit(0)

    result_file = get_result_file(args)

    # load the check point
    if os.path.exists(result_file):
        print("# The result file exists! We will load the learned check point!!!")
        check_point = json.load(open(result_file))
        results = check_point['results']
        if args.get_false_test_pids:
            false_test_pids = []
            correct_test_pids = []
            if args.test_pids_ckpt:
                result_key = pids
            else:
                result_key = results.keys()
            for pid in result_key:
                if results[pid]["true_false"] == False:
                    false_test_pids.append(pid)
                else:
                    correct_test_pids.append(pid)
            print("false test num: {}".format(len(false_test_pids)))
            print("accuracy for test_pids: {}".format(len(correct_test_pids) / len(result_key)))

            # false_test_file = "cluster_results/false_{}_{}_{}_{}_{}_seed_{}.pth".format(args.label, args.test_split,
            #                                                                             args.prompt_format,
            #                                                                             args.shot_number,
            #                                                                             args.retrieve_method, args.seed)
            # correct_test_file = "cluster_results/correct_{}_{}_{}_{}_{}_seed_{}.pth".format(args.label, args.test_split,
            #                                                                                 args.prompt_format,
            #                                                                                 args.shot_number,
            #                                                                                 args.retrieve_method,
            #                                                                                 args.seed)
            # torch.save(false_test_pids, false_test_file)
            # torch.save(correct_test_pids, correct_test_file)
            # print(f"save false_test_pids to {false_test_file}")
            exit(0)
    else:
        results = {}

    total = len(pids)
    check_count = len(results)  # number of existing results
    correct = 0  # number of correct results




    print("candidate prompts: ")
    print("===========")
    cand_examples = []
    for pid in cand_pids:
        example = create_example_from_pid(pid, problems, args, test=True)  # CHECK !!!
        #print(example)
        #print("===========")
        cand_examples.append(example)
    if args.train_ckpt:
        train_pids = torch.load(args.train_ckpt)
        train_pids = [i for i in train_pids if i not in pids]
        val_examples = []
        for pid in train_pids:
            val_example = create_example_from_pid(pid, problems, args, test=True)  # CHECK !!!
            # print(example)
            # print("===========")
            val_examples.append(val_example)
        if args.val_ckpt:
            val_pids = torch.load(args.val_ckpt)
            correct_val_pids = [i for i in val_pids if i not in train_pids]
            correct_val_examples = []
            for pid in correct_val_pids:
                correct_val_example = create_example_from_pid(pid, problems, args, test=True)  # CHECK !!!
                # print(example)
                # print("===========")
                correct_val_examples.append(correct_val_example)

    if args.aug_method == 'hints':
        aug_label = []
        for i in cluster_num.keys():
            if cluster_num[i] <= args.aug_th:
                aug_label.append(i)
        print("aug label", aug_label)
        cluster_train_id = [i for i in range(len(cluster_id)) if cluster_id[i] in aug_label]

        # #subsample cluster_train_id
        # cluster_train_id = random.sample(cluster_train_id, 60)


        examples_train_aug = [cand_examples[i] for i in cluster_train_id]
        embeddings = [embeddings[i] for i in cluster_train_id]
        cluster_id_aug = [cluster_id[i] for i in cluster_train_id]

        # cluster_num = {}
        # for i in cluster_id_aug:
        #     if i not in cluster_num.keys():
        #         cluster_num[i] = 1
        #     else:
        #         cluster_num[i] = cluster_num[i] + 1
        # print("after reducing samples:", cluster_num)

        print("len train", len(examples_train_aug))
        print("len embeddings", len(embeddings))
        print("len cluster id", len(cluster_id_aug))
        new_samples = hint_aug(args, examples_train_aug, embeddings, cluster_id_aug)
    elif args.aug_method == 'seed':
        seed_sentence = torch.load(
            f"cluster_results/seed_sentence_{args.cluster_type}_{args.n_clusters}_1000.pt")
        print(
            f"Load seed sentence from cluster_results/seed_sentence_{args.cluster_type}_{args.n_clusters}_1000.pt")
        print("generate augmented samples using seed sentences.")
        new_samples = seed_aug(args, seed_sentence)
    else:
        new_samples = []
    #print("len(examples_train)", len(examples_train)

    #generate new pids
    new_pids = []
    for i in range(len(new_samples)):
        if i not in cand_pids:
            new_pids.append(i)
        else:
            print("id already exists!")
            exit(0)

    #combine cand samples and augmented samples
    cand_examples.extend(new_samples)
    cand_pids.extend(new_pids)
    for i in range(len(new_samples)):
        problems.update({new_pids[i]: {'new_samples': new_samples[i]}})
    print("Extended cand size", len(cand_pids))
    test_examples = []
    for pid in pids:
        if 'medqa' in args.data_root_test or args.data_root_test in ['ethos-national_origin',
                                                                           'blimp-anaphor_number_agreement',
                                                                           'tweet_eval-irony', 'wino_grande',
                                                                           'race-middle', 'proto_qa', 'race-high',
                                                                           'kilt_hotpotqa',
                                                                           'ade_corpus_v2-classification',
                                                                           'hate_speech18', 'tweet_eval-stance_hillary',
                                                                           'tweet_eval-offensive', 'kilt_ay2',
                                                                           'squad-with_context', 'tweet_eval-sentiment',
                                                                           'ethos-religion', 'wikisql',
                                                                           'squad-no_context', 'ethos-race',
                                                                           'tweet_eval-stance_climate',
                                                                           'hate_speech_offensive', 'kilt_nq',
                                                                           'tweet_eval-hate', 'tweet_eval-emotion',
                                                                           'ethos-sexual_orientation', 'hatexplain',
                                                                           'kilt_fever', 'blimp-ellipsis_n_bar_2',
                                                                           'ethos-gender', 'ade_corpus_v2-dosage',
                                                                           'ethos-disability',
                                                                           'tweet_eval-stance_atheism', 'kilt_zsre',
                                                                           'blimp-sentential_negation_npi_licensor_present',
                                                                           'tweet_eval-stance_feminist', 'kilt_trex',
                                                                           'tweet_eval-stance_abortion',
                                                                           'ethos-directed_vs_generalized',
                                                                           'blimp-sentential_negation_npi_scope',
                                                                           'tweet_eval-emoji']:
            pid = int(pid)
        example = create_example_from_pid(pid, problems, args, test=True)  # CHECK !!!
        test_examples.append(example)

    # ======================================================= INFERENCE ===============================================
    if args.ckpt:
        ckpt_path = os.path.join(args.ckpt_root, args.ckpt)
        if args.ckpt_context:
            ckpt_context_path = os.path.join(args.ckpt_root, args.ckpt_context)
            if os.path.exists(ckpt_context_path):
                algorithm.context_net.linear.load_state_dict(torch.load(ckpt_context_path))
                print("context model loaded")
            else:
                print(f"The ckpt path for [{ckpt_context_path}] does not exist!")
        if args.ckpt_lossnet:
            ckpt_lossnet_path = os.path.join(args.ckpt_root, args.ckpt_lossnet)
            if os.path.exists(ckpt_lossnet_path):
                algorithm.learned_loss_net.load_state_dict(torch.load(ckpt_lossnet_path))
                print("Loss net loaded")
            else:
                print(f"The ckpt path for [{ckpt_lossnet_path}] does not exist!")

        if os.path.exists(ckpt_path):
            algorithm.model.linear.load_state_dict(torch.load(ckpt_path))
            print("Policy model loaded")
        else:
            print(f"The ckpt path for [{ckpt_path}] does not exist!")  # CHECK
            #exit()



    else:
        print(f"!!! Load the pre-traind model instead!")  # CHECK
        #exit()

    algorithm.model.eval()
        # Calculate the embeddings for candidate examples only one time!
    #with torch.no_grad():
    cand_embedding = algorithm.predict(cand_examples)
    if args.train_ckpt:
        val_embedding = algorithm.predict(val_examples)
        if args.val_ckpt:
            correct_val_embedding = algorithm.predict(correct_val_examples)
    # num_batch = math.ceil(len(cand_examples) / args.batch_size)
    # cand_embedding = []
    # for batch_i in range(num_batch):
    #     cand_batch = cand_examples[batch_i * args.batch_size:(batch_i + 1) * args.batch_size]
    #     if args.gamma != 0:
    #         cand_embedding_i, _ = algorithm.predict(cand_batch)
    #     #update the policy model with test sample
    #     #_,_ = algorithm.predict(test_examples)
    #     else:
    #         cand_embedding_i = algorithm.predict(cand_batch)
    #     cand_embedding.append(cand_embedding_i)
    #     #_ = algorithm.predict(test_examples)
    #     # print("cand_embedding:", cand_embedding.shape)  # [cand_num x emb_size]
    # cand_embedding = torch.cat(cand_embedding)
    # print("cand_embedding shape", cand_embedding.shape)
    # _ = algorithm.predict(test_examples)
    wrong_max_scores = []
    correct_max_scores = []
    wrong_max_scores_true = []
    correct_max_scores_true = []
    shot_len_avg = []
    if args.preselection:
        original_cand_pids = copy.deepcopy(cand_pids)
    with torch.no_grad():
        for i, pid in enumerate(pids):
            if 'medqa' in args.data_root_test or args.data_root_test in ['ethos-national_origin',
                                                                           'blimp-anaphor_number_agreement',
                                                                           'tweet_eval-irony', 'wino_grande',
                                                                           'race-middle', 'proto_qa', 'race-high',
                                                                           'kilt_hotpotqa',
                                                                           'ade_corpus_v2-classification',
                                                                           'hate_speech18', 'tweet_eval-stance_hillary',
                                                                           'tweet_eval-offensive', 'kilt_ay2',
                                                                           'squad-with_context', 'tweet_eval-sentiment',
                                                                           'ethos-religion', 'wikisql',
                                                                           'squad-no_context', 'ethos-race',
                                                                           'tweet_eval-stance_climate',
                                                                           'hate_speech_offensive', 'kilt_nq',
                                                                           'tweet_eval-hate', 'tweet_eval-emotion',
                                                                           'ethos-sexual_orientation', 'hatexplain',
                                                                           'kilt_fever', 'blimp-ellipsis_n_bar_2',
                                                                           'ethos-gender', 'ade_corpus_v2-dosage',
                                                                           'ethos-disability',
                                                                           'tweet_eval-stance_atheism', 'kilt_zsre',
                                                                           'blimp-sentential_negation_npi_licensor_present',
                                                                           'tweet_eval-stance_feminist', 'kilt_trex',
                                                                           'tweet_eval-stance_abortion',
                                                                           'ethos-directed_vs_generalized',
                                                                           'blimp-sentential_negation_npi_scope',
                                                                           'tweet_eval-emoji']:
                pid = int(pid)
            count = i + 1  # number of current results
            problem = problems[pid]
            # if 'solution' not in problems[pid].keys():
            #     _, _, answer = problems[pid]['answer'].partition("\n#### ")
            # else:
            #     answer = problems[pid]['answer']
            if 'tabmwp' in args.data_root_test or 'medqa' in args.data_root_test:
                answer = problem['answer']
            elif 'gsm8k' in args.data_root_test:
                _, _, answer = problem['answer'].partition("\n#### ")
            elif 'MATH' in args.data_root_test:
                answer = remove_boxed(last_boxed_only_string(problem["solution"]))
            elif 'pubmed' in args.data_root_test:
                answer = problem['final_decision']
            elif "output" in problem.keys():
                answer = problem['output']
            else:
                raise Exception("The dataset does not exist!")
            if "options" in problem.keys():
                if 'medqa' in args.data_root_test:
                    options = []
                    for o in problems[pid]['options'].keys():
                        options.append(problems[pid]['options'][o])
                else:
                    options = problems[pid]['options']
            elif "choices" in problem.keys():
                options = problems[pid]['choices']
            else:
                options = None
            if 'unit' in problems[pid].keys():
                unit = problems[pid]['unit']
            else:
                unit = None
            if str(pid) in results:
                pid = str(pid)
                output = results[pid]["output"]
                shot_len_avg.append(len(results[pid]["shot_pids"]))
            else:

                example = create_example_from_pid(pid, problems, args, test=True)
                if args.preselection:
                    Cand_example = bm25_retrieve(example, cand_examples, n=args.select_number)
                    cand_idx = bm25_retrieve(example, cand_examples, n=args.select_number, return_index=True).tolist()
                    cand_pids = [original_cand_pids[c] for c in cand_idx]
                    cand_embedding = algorithm.predict(Cand_example)
                # if i < 10:
                if args.gamma != 0:
                    ctxt_embedding, _ = algorithm.predict([example], test=True)
                else:
                    ctxt_embedding = algorithm.predict([example], test=True)

                # else:
                #     with torch.no_grad():
                #         if args.gamma != 0:
                #             ctxt_embedding, _ = algorithm.predict([example], test=True)
                #         else:
                #             ctxt_embedding = algorithm.predict([example], test=True)
                # print("ctxt_embedding:", ctxt_embedding.shape)  # [1 x emb_size]
                # algorithm.model.eval()
                # Calculate the embeddings for candidate examples only one time!
                # if count > 10:
                #     with torch.no_grad():
                #         if args.gamma != 0:
                #             cand_embedding, _ = algorithm.predict(cand_examples, test=True)
                #             # update the policy model with test sample
                #             # _,_ = algorithm.predict(test_examples)
                #         else:
                #             cand_embedding = algorithm.predict(cand_examples, test=True)
                # if count < 10:
                # cand_embedding = []
                # with torch.no_grad():
                #     for batch_i in range(num_batch):
                #         cand_batch = cand_examples[batch_i * args.batch_size:(batch_i + 1) * args.batch_size]
                #         if args.gamma != 0:
                #             cand_embedding_i, _ = algorithm.predict(cand_batch, test=True)
                #         # update the policy model with test sample
                #         # _,_ = algorithm.predict(test_examples)
                #         else:
                #             cand_embedding_i = algorithm.predict(cand_batch, test=True)
                #         cand_embedding.append(cand_embedding_i)
                #         # _ = algorithm.predict(test_examples)
                #         # print("cand_embedding:", cand_embedding.shape)  # [cand_num x emb_size]
                #     cand_embedding = torch.cat(cand_embedding)

                # print("cand_embedding shape", cand_embedding.shape)

                scores = F.softmax(torch.mm(ctxt_embedding, cand_embedding.t()), dim=1)[0]  # [cand_num]
                # print(scores.shape)
                scores = scores.cpu().detach().numpy().tolist()
                score_th = args.score_th
                if args.train_ckpt:
                    val_scores = F.softmax(torch.mm(val_embedding, cand_embedding.t()), dim=1)  # [cand_num]
                    val_scores = val_scores.cpu().detach().numpy()
                    # print(len(scores), val_scores.shape)
                    val_mean, val_std = np.mean(val_scores, axis=0), np.std(val_scores, axis=0)
                    # print(f"get mean {val_mean} and std {val_std} for validation data!")
                    score_th = (args.score_th - val_mean) + scores
                    # print(f"The adjust score_th is {score_th}")
                if args.val_ckpt:
                    correct_val_scores = F.softmax(torch.mm(ctxt_embedding, correct_val_embedding.t()), dim=1)[
                        0]  # [cand_num]
                    # print(scores.shape)
                    correct_val_scores = correct_val_scores.cpu().detach().numpy().tolist()

                shot_pids = []
                # if max(scores) > score_th:
                cand_ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:args.shot_number]
                for cid in cand_ids[::-1]:
                    if args.train_ckpt is None:
                        if scores[cid] > score_th:
                            shot_pids.append(cand_pids[cid])
                    else:
                        if scores[cid] > score_th[cid]:
                            shot_pids.append(cand_pids[cid])
                # print(scores[cand_ids[-1]])

                # if len(shot_pids) > 0:
                #     print(len(shot_pids))
                # shot_pids = [cand_pids[cid] for cid in cand_ids[::-1]]
                # print("shot_pids:", shot_pids)
                shot_len_avg.append(len(shot_pids))

                prompt_no_test = build_prompt(problems, shot_pids, pid, args, include_test=False)

                #        if pid in results:
                #            output = results[pid]["output"]
                #        else:

                # if args.duplicate_sample > 0:
                #     # let gpt generate nearly duplicate samples
                #     new_sample = get_gpt3_output(prompt_no_test, args,
                #                                  user_prompt=f'Please generate {args.duplicate_sample} nearly duplicate samples as these samples use the same format.')
                #     #print(f"generate duplicate sample: {new_sample}")
                #     #cluster new_pid with original id
                #     new_pid = len(problems)
                #     temp, _, solution = new_sample.partition("Answer: ")
                #     _, _, question = temp.partition("Question: ")
                #     problems.append({})
                #     problems[new_pid]["solution"] = solution
                #     problems[new_pid]['question'] = question
                #     shot_pids.append(new_pid)
                prompt = build_prompt(problems, shot_pids, pid, args)  # generate the prompt input

                # if str(pid) in results:
                #     output = results[str(pid)]["output"]
                # else:
                output = get_gpt_output(prompt, args)  # generate the output by GPT-3

            if 'tabmwp' in args.data_root_test or 'medqa' in args.data_root_test:
                # the core prediction in the output
                if output:
                    prediction = extract_prediction(output, options, args.option_inds)
                    prediction_norm = normalize_answer(prediction, unit)
                else:
                    prediction = output
                    prediction_norm = output

                # normalize the number in the text
                answer_norm = normalize_answer(answer, unit)

            elif 'pubmed' in args.data_root_test:
                answer_norm = answer
                prediction = extract_prediction(output, options, args.option_inds)
                prediction_norm = prediction
            else:
                # the core prediction in the output
                if output:

                    prediction = remove_boxed(last_boxed_only_string(output))

                    if not prediction:
                        prediction = extract_prediction(output, options, args.option_inds)
                else:
                    prediction = output

                # normalize the number in the text
                if answer:
                    # answer = normalize_answer(answer, unit)
                    # prediction = normalize_answer(prediction, unit)
                    answer_norm = _strip_string(answer)
                    # answer_norm = answer
                    if prediction:
                        prediction_norm = _strip_string(prediction)
                    else:
                        prediction_norm = prediction

            if answer:
                if str(pid) not in results:
                    # save the results
                    results[pid] = {}

                    results[pid]["shot_pids"] = shot_pids
                    results[pid]["prompt"] = prompt
                    results[pid]["answer"] = answer
                    results[pid]["answer_norm"] = answer_norm
                    results[pid]["output"] = output
                    results[pid]["prediction"] = prediction
                    results[pid]["prediction_norm"] = prediction_norm
                    # if args.duplicate_sample > 0:
                    #     results[pid]["duplicate_sample"] = new_sample
                else:
                    shot_pids = results[pid]["shot_pids"]
                    prompt = results[pid]["prompt"]
                    answer = results[pid]["answer"]
                    answer_norm = results[pid]["answer_norm"]
                    output = results[pid]["output"]
                    prediction = results[pid]["prediction"]
                    prediction_norm = results[pid]["prediction_norm"]



                # correct or not
                if prediction_norm:
                    if answer_norm.lower() in prediction_norm.lower():
                        correct += 1
                        results[pid]["true_false"] = True
                        # if args.train_ckpt:
                        #     correct_max_scores.append(max(val_scores))
                        #     if args.val_ckpt:
                        #         correct_max_scores_true.append(max(correct_val_scores))
                    else:
                        results[pid]["true_false"] = False
                        # if args.train_ckpt:
                        #     wrong_max_scores.append(max(val_scores))
                        #     if args.val_ckpt:
                        #         wrong_max_scores_true.append(max(correct_val_scores))
                else:
                    results[pid]["true_false"] = False

                acc = correct / (i + 1) * 100

                if args.debug or i < 10:
                    print("\n##################################")
                    print(prompt, "\n")
                    print("[A] labeled answer (normalized):\t", answer_norm)
                    print("[P] predicted answer (normalized):\t", prediction_norm)
                    print("[Acc]:\t", results[pid]["true_false"])
                    print("")
                    print("[A] labeled answer:\t", answer)
                    print("[P] predicted answer:\t", prediction)
                    print("[P] generated output:\t", output)

                if count % args.save_every == 0 or count == total:
                    avg_len = sum(shot_len_avg)/len(shot_len_avg)
                    if count >= check_count:
                        # have new outputs
                        print(f"{count}/{total}, correct: {correct}, acc: {round(acc, 2)}%, avg shot number {avg_len}, saved to {result_file}")
                        save_results(result_file, acc, correct, count, cand_pids, args, results)
                    else:
                        # no new outputs, just print the accuracy
                        print(f"{count}/{total}, correct: {correct}, acc: {round(acc, 2)}%, avg shot number {avg_len}")
                    # if args.train_ckpt:
                    #     if len(wrong_max_scores) > 0:
                    #         print(f"wrong max score for false val: {sum(wrong_max_scores)/len(wrong_max_scores)}, wrong max score for correct val: {sum(wrong_max_scores_true)/len(wrong_max_scores_true)}")
                    #     if len(correct_max_scores) > 0:
                    #         print(f"correct max score for false val: {sum(correct_max_scores) / len(correct_max_scores)}, correct max score for correct val: {sum(correct_max_scores_true) / len(correct_max_scores_true)}")
                    #
                    #












