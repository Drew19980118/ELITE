# -*- encoding:utf-8 -*-
"""
  This script provides a k-BERT exmaple for classification.
"""
import sys
import torch
import json
import random
import argparse
import collections
import torch.nn as nn
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils.tokenizer import * 
from uer.model_builder import build_model
from uer.utils.optimizers import  BertAdam
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from uer.targets.mlm_target import WordMlmTarget, RelMlmTarget, EntMlmTarget
from brain import KnowledgeGraph
from multiprocessing import Process, Pool
import numpy as np

class RelevanceScoreBertClassifier(nn.Module):
    def __init__(self, args, model):
        super(RelevanceScoreBertClassifier, self).__init__()
        self.embedding = model.embedding
        self.encoder = model.encoder
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, args.hidden_size)
        print("[RelevanceScoreBertClassifier] is loading")
    def forward(self, src, mask):
        """
        Args:
            src: [batch_size x seq_length]
            label: [batch_size]
            mask: [batch_size x seq_length]
        """
        # src/mask_src Embedding.
        emb = self.embedding(src)
        output = self.encoder(emb, mask)
        output = self.output_layer_2(torch.tanh(self.output_layer_1(output)))
        return output


class BertClassifier(nn.Module):
    def __init__(self, args, model):
        super(BertClassifier, self).__init__()
        self.embedding = model.embedding
        self.encoder = model.encoder
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.word_rel_ent_mlm_loss = WordMlmTarget(args, len(args.vocab))
        self.rel_mlm_loss = RelMlmTarget(args, len(args.vocab))
        self.ent_mlm_loss = EntMlmTarget(args, len(args.vocab))
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, args.labels_num)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()
        self.use_vm = False if args.no_vm else True
        print("[BertClassifier] use visible_matrix: {}".format(self.use_vm))

    def forward(self, id, src, label, mask, type, tgt_word, tgt_rel, tgt_ent, mask_src, mask_rel, mask_ent, concept_ent_pairs, edge_idx, pos=None, vm=None):
        """
        Args:
            src: [batch_size x seq_length]
            label: [batch_size]
            mask: [batch_size x seq_length]
        """
        # src/mask_src Embedding.
        emb = self.embedding(id, src, mask, type, concept_ent_pairs, edge_idx, pos, True)
        if mask_src != None:
            mask_emb = self.embedding(id, mask_src, mask, type, concept_ent_pairs, edge_idx, pos, False)
            #mask_rel_emb = self.embedding(mask_rel, mask, type, pos)
            #mask_ent_emb = self.embedding(mask_ent, mask, type, pos)

        # src/mask_src Encoder.
        if not self.use_vm:
            vm = None
        output = self.encoder(id, concept_ent_pairs, edge_idx, emb, mask, True, vm)
        if mask_src != None:
            mask_output = self.encoder(id, concept_ent_pairs, edge_idx, mask_emb, mask, False, vm)
            #mask_rel_output = self.encoder(mask_rel_emb, mask, vm)
            #mask_ent_output = self.encoder(mask_ent_emb, mask, vm)
            # Target.
            word_rel_ent_mlm_loss, _, _ = self.word_rel_ent_mlm_loss(mask_output, tgt_word)
            #rel_mlm_loss, _, _ = self.rel_mlm_loss(mask_rel_output, tgt_rel)
            #ent_mlm_loss, _, _ = self.ent_mlm_loss(mask_ent_output, tgt_ent)
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        loss = self.criterion(self.softmax(logits.view(-1, self.labels_num)), label.view(-1))
        if mask_src != None:
            loss = loss + 0.1*word_rel_ent_mlm_loss
        return loss, logits

def add_knowledge_worker(params):

    p_id, sentences, columns, kg, vocab, args, model = params

    sentences_num = len(sentences)
    dataset = []
    for line_id, line in enumerate(sentences):
        if line_id % 10000 == 0:
            print("Progress of process {}: {}/{}".format(p_id, line_id, sentences_num))
            sys.stdout.flush()
        line = line.strip().split('\t')
        try:
            if len(line) == 2:
                label = int(line[columns["label"]])
                text = CLS_TOKEN + line[columns["text_a"]]
                tokens, mask_tokens, mask_rel_tokens, mask_ent_tokens, pos, vm, type, tgt_word, tgt_rel, tgt_ent, id, concept_ent_pairs, edge_idx, _ = kg.add_knowledge_with_vm(line_id, [text], vocab, model, add_pad=True, max_length=args.seq_length)
                tokens = tokens[0]
                mask_tokens = mask_tokens[0]
                mask_rel_tokens = mask_rel_tokens[0]
                mask_ent_tokens = mask_ent_tokens[0]
                pos = pos[0]
                vm = vm[0].astype("bool")
                type = type[0]
                tgt_word = tgt_word[0]
                tgt_rel = tgt_rel[0]
                tgt_ent = tgt_ent[0]
                # text_node = text_node[0]
                id = id[0]
                concept_ent_pairs = concept_ent_pairs[0]
                edge_idx = edge_idx[0]

                token_ids = [vocab.get(t) for t in tokens]
                mask_token_ids = [vocab.get(t) for t in mask_tokens]
                mask_rel_token_ids = [vocab.get(t) for t in mask_rel_tokens]
                mask_ent_token_ids = [vocab.get(t) for t in mask_ent_tokens]
                mask = [1 if t != PAD_TOKEN else 0 for t in tokens]

                dataset.append((token_ids, mask_token_ids, mask_rel_token_ids, mask_ent_token_ids, label, mask, pos, vm, type, tgt_word, tgt_rel, tgt_ent, id, concept_ent_pairs, edge_idx))
            
            elif len(line) == 3:
                label = int(line[columns["label"]])
                text = CLS_TOKEN + line[columns["text_a"]] + SEP_TOKEN + line[columns["text_b"]] + SEP_TOKEN

                tokens, pos, vm, type, _ = kg.add_knowledge_with_vm([text], add_pad=True, max_length=args.seq_length)
                tokens = tokens[0]
                pos = pos[0]
                vm = vm[0].astype("bool")
                type = type[0]

                token_ids = [vocab.get(t) for t in tokens]
                mask = []
                seg_tag = 1
                for t in tokens:
                    if t == PAD_TOKEN:
                        mask.append(0)
                    else:
                        mask.append(seg_tag)
                    if t == SEP_TOKEN:
                        seg_tag += 1

                dataset.append((token_ids, label, mask, pos, vm, type))
            
            elif len(line) == 4:  # for dbqa
                qid=int(line[columns["qid"]])
                label = int(line[columns["label"]])
                text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
                text = CLS_TOKEN + text_a + SEP_TOKEN + text_b + SEP_TOKEN

                tokens, pos, vm, type, _ = kg.add_knowledge_with_vm([text], add_pad=True, max_length=args.seq_length)
                tokens = tokens[0]
                pos = pos[0]
                vm = vm[0].astype("bool")
                type = type[0]

                token_ids = [vocab.get(t) for t in tokens]
                mask = []
                seg_tag = 1
                for t in tokens:
                    if t == PAD_TOKEN:
                        mask.append(0)
                    else:
                        mask.append(seg_tag)
                    if t == SEP_TOKEN:
                        seg_tag += 1
                
                dataset.append((token_ids, label, mask, pos, vm, type, qid))
            else:
                pass
            
        except Exception as e:
            print("Error line: ", line)
            print("Error line_id: ", line_id)
            print(f"An error occurred: {e}")
    return dataset


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--relevance_model_path", default=None, type=str,
                        help="Path of the relevance score model.")
    parser.add_argument("--output_model_path", default="./models/classifier_model.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--vocab_path", default="./models/google_vocab.txt", type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path of the trainset.")
    parser.add_argument("--dev_path", type=str, required=True,
                        help="Path of the devset.") 
    parser.add_argument("--test_path", type=str, required=True,
                        help="Path of the testset.")
    parser.add_argument("--config_path", default="./models/google_config.json", type=str,
                        help="Path of the config file.")

    # Model options.
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=256,
                        help="Sequence length.")
    parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
                                                   "cnn", "gatedcnn", "attn", \
                                                   "rcnn", "crnn", "gpt", "bilstm"], \
                                                   default="bert", help="Encoder type.")

    parser.add_argument("--relevance_encoder", default="relevance", help="relevane encoder type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    # Subword options.
    parser.add_argument("--subword_type", choices=["none", "char"], default="none",
                        help="Subword feature type.")
    parser.add_argument("--sub_vocab_path", type=str, default="models/sub_vocab.txt",
                        help="Path of the subword vocabulary file.")
    parser.add_argument("--subencoder", choices=["avg", "lstm", "gru", "cnn"], default="avg",
                        help="Subencoder type.")
    parser.add_argument("--sub_layers_num", type=int, default=2, help="The number of subencoder layers.")

    # Tokenizer options.
    parser.add_argument("--tokenizer", choices=["bert", "char", "word", "space"], default="bert",
                        help="Specify the tokenizer." 
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Word tokenizer supports online word segmentation based on jieba segmentor."
                             "Space tokenizer segments sentences into words according to space."
                             )

    # Optimizer options.
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")

    # Training options.
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=25,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")

    # Evaluation options.
    parser.add_argument("--mean_reciprocal_rank", action="store_true", help="Evaluation metrics for DBQA dataset.")

    # kg
    parser.add_argument("--kg_name", required=True, help="KG name or path")
    parser.add_argument("--workers_num", type=int, default=1, help="number of process for loading dataset")
    parser.add_argument("--no_vm", action="store_true", help="Disable the visible_matrix")

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Count the number of labels.
    labels_set = set()
    columns = {}
    with open(args.train_path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            try:
                line = line.strip().split("\t")
                if line_id == 0:
                    for i, column_name in enumerate(line):
                        columns[column_name] = i
                    continue
                label = int(line[columns["label"]])
                labels_set.add(label)
            except:
                pass
    args.labels_num = len(labels_set) 

    # Load vocabulary.
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    # Build bert model.
    # A pseudo target is added.
    args.target = "bert"
    model, relevance_model = build_model(args)

    # Load or initialize parameters.
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)
        relevance_model.load_state_dict(torch.load(args.relevance_model_path, map_location=torch.device('cpu')), strict=False)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if 'gamma' not in n and 'beta' not in n:
                p.data.normal_(0, 0.02)
    
    # Build classification model.
    model = BertClassifier(args, model)

    relevance_model = RelevanceScoreBertClassifier(args, relevance_model)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model = model.to(device)
    relevance_model = relevance_model.to(device)
    
    # Datset loader.
    def batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms, type_ids, id, concept_ent_pairs, edge_idx, tgt_word, tgt_rel, tgt_ent, mask_input_ids, mask_rel_ids, mask_ent_ids):
        instances_num = input_ids.size()[0]
        for i in range(instances_num // batch_size):
            input_ids_batch = input_ids[i*batch_size: (i+1)*batch_size, :]
            if mask_input_ids == None:
                mask_input_ids_batch = None
                mask_rel_ids_batch = None
                mask_ent_ids_batch = None
                tgt_word_batch = None
                tgt_rel_batch = None
                tgt_ent_batch = None
            else:
                mask_input_ids_batch = mask_input_ids[i*batch_size: (i+1)*batch_size, :]
                mask_rel_ids_batch = mask_rel_ids[i * batch_size: (i + 1) * batch_size, :]
                mask_ent_ids_batch = mask_ent_ids[i * batch_size: (i + 1) * batch_size, :]
                tgt_word_batch = tgt_word[i * batch_size: (i + 1) * batch_size, :]
                tgt_rel_batch = tgt_rel[i * batch_size: (i + 1) * batch_size, :]
                tgt_ent_batch = tgt_ent[i * batch_size: (i + 1) * batch_size, :]
            label_ids_batch = label_ids[i*batch_size: (i+1)*batch_size]
            mask_ids_batch = mask_ids[i*batch_size: (i+1)*batch_size, :]
            pos_ids_batch = pos_ids[i*batch_size: (i+1)*batch_size, :]
            vms_batch = vms[i*batch_size: (i+1)*batch_size]
            type_ids_batch = type_ids[i*batch_size: (i+1)*batch_size, :]
            id_batch = id[i * batch_size: (i + 1) * batch_size]
            concept_ent_pairs_batch = concept_ent_pairs[i*batch_size: (i+1)*batch_size]
            edge_idx_batch = edge_idx[i*batch_size: (i+1)*batch_size]
            yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch, type_ids_batch, tgt_word_batch, tgt_rel_batch, tgt_ent_batch, mask_input_ids_batch, mask_rel_ids_batch, mask_ent_ids_batch, id_batch, concept_ent_pairs_batch, edge_idx_batch
        if instances_num > instances_num // batch_size * batch_size:
            input_ids_batch = input_ids[instances_num//batch_size*batch_size:, :]
            if mask_input_ids == None:
                mask_input_ids_batch = None
                mask_rel_ids_batch = None
                mask_ent_ids_batch = None
                tgt_word_batch = None
                tgt_rel_batch = None
                tgt_ent_batch = None
            else:
                mask_input_ids_batch = mask_input_ids[instances_num // batch_size * batch_size:, :]
                mask_rel_ids_batch = mask_rel_ids[instances_num // batch_size * batch_size:, :]
                mask_ent_ids_batch = mask_ent_ids[instances_num // batch_size * batch_size:, :]
                tgt_word_batch = tgt_word[instances_num // batch_size * batch_size:, :]
                tgt_rel_batch = tgt_rel[instances_num // batch_size * batch_size:, :]
                tgt_ent_batch = tgt_ent[instances_num // batch_size * batch_size:, :]
            label_ids_batch = label_ids[instances_num//batch_size*batch_size:]
            mask_ids_batch = mask_ids[instances_num//batch_size*batch_size:, :]
            pos_ids_batch = pos_ids[instances_num//batch_size*batch_size:, :]
            vms_batch = vms[instances_num // batch_size * batch_size:]
            type_ids_batch = type_ids[instances_num//batch_size*batch_size:, :]
            id_batch = id[i * batch_size: (i + 1) * batch_size]
            concept_ent_pairs_batch = concept_ent_pairs[i * batch_size: (i + 1) * batch_size]
            edge_idx_batch = edge_idx[i * batch_size: (i + 1) * batch_size]

            yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch, type_ids_batch, tgt_word_batch, tgt_rel_batch, tgt_ent_batch, mask_input_ids_batch, mask_rel_ids_batch, mask_ent_ids_batch, id_batch, concept_ent_pairs_batch, edge_idx_batch

    # Build knowledge graph.
    if args.kg_name == 'none':
        spo_files = []
    else:
        #spo_files = args.kg_name.split(',')
        spo_files = [args.kg_name]
    kg = KnowledgeGraph(spo_files=spo_files, predicate=False)

    def read_dataset(path, model, workers_num=1):

        print("Loading sentences from {}".format(path))
        sentences = []
        with open(path, mode='r', encoding="utf-8") as f:
            for line_id, line in enumerate(f):
                if line_id == 0:
                    continue
                sentences.append(line)
        sentence_num = len(sentences)

        print("There are {} sentence in total. We use {} processes to inject knowledge into sentences.".format(sentence_num, workers_num))
        if workers_num > 1:
            params = []
            sentence_per_block = int(sentence_num / workers_num) + 1
            for i in range(workers_num):
                params.append((i, sentences[i*sentence_per_block: (i+1)*sentence_per_block], columns, kg, vocab, args))
            pool = Pool(workers_num)
            res = pool.map(add_knowledge_worker, params)
            pool.close()
            pool.join()
            dataset = [sample for block in res for sample in block]
        else:
            params = (0, sentences, columns, kg, vocab, args, model)
            dataset = add_knowledge_worker(params)

        return dataset

    # Evaluation function.
    def evaluate(args, is_test, metrics='Acc'):
        if is_test:
            dataset = read_dataset(args.test_path, relevance_model, workers_num=args.workers_num)
        else:
            dataset = read_dataset(args.dev_path, relevance_model, workers_num=args.workers_num)

        input_ids = torch.LongTensor([sample[0] for sample in dataset])
        label_ids = torch.LongTensor([sample[4] for sample in dataset])
        mask_ids = torch.LongTensor([sample[5] for sample in dataset])
        pos_ids = torch.LongTensor([example[6] for example in dataset])
        vms = [example[7] for example in dataset]
        type_ids = torch.LongTensor([example[8] for example in dataset])
        # tgt_word = torch.LongTensor([example[9] for example in dataset])
        # tgt_rel = torch.LongTensor([example[10] for example in dataset])
        # tgt_ent = torch.LongTensor([example[11] for example in dataset])
        id = [example[12] for example in dataset]
        concept_ent_pairs = [example[13] for example in dataset]
        edge_idx = [example[14] for example in dataset]

        batch_size = args.batch_size
        instances_num = input_ids.size()[0]
        if is_test:
            print("The number of evaluation instances: ", instances_num)

        correct = 0
        # Confusion matrix.
        confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

        model.eval()

        if not args.mean_reciprocal_rank:
            for i, (input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch, type_ids_batch, tgt_word_batch, tgt_rel_batch, tgt_ent_batch, mask_input_ids_batch, mask_rel_ids_batch, mask_ent_ids_batch, id_batch, concept_ent_pairs_batch, edge_idx_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms, type_ids, id, concept_ent_pairs, edge_idx, tgt_word=None, tgt_rel=None, tgt_ent=None, mask_input_ids=None, mask_rel_ids=None, mask_ent_ids=None)):

                vms_batch = torch.LongTensor(vms_batch)

                input_ids_batch = input_ids_batch.to(device)
                label_ids_batch = label_ids_batch.to(device)
                mask_ids_batch = mask_ids_batch.to(device)
                pos_ids_batch = pos_ids_batch.to(device)
                vms_batch = vms_batch.to(device)
                type_ids_batch = type_ids_batch.to(device)
                # tgt_word_batch = tgt_word_batch.to(device)
                # tgt_rel_batch = tgt_rel_batch.to(device)
                # tgt_ent_batch = tgt_ent_batch.to(device)

                with torch.no_grad():
                    try:
                        loss, logits = model(id_batch, input_ids_batch, label_ids_batch, mask_ids_batch, type_ids_batch, tgt_word_batch, tgt_rel_batch, tgt_ent_batch, mask_input_ids_batch, mask_rel_ids_batch, mask_ent_ids_batch, concept_ent_pairs_batch, edge_idx_batch, pos_ids_batch, vms_batch)
                    except:
                        print(input_ids_batch)
                        print(input_ids_batch.size())
                        print(vms_batch)
                        print(vms_batch.size())

                logits = nn.Softmax(dim=1)(logits)
                pred = torch.argmax(logits, dim=1)
                gold = label_ids_batch
                for j in range(pred.size()[0]):
                    try:
                        confusion[pred[j], gold[j]] += 1
                    except Exception as e:
                        print(f"An error occurred: {e}")
                        print("confusion shape", confusion.shape)
                        print("logits shape", logits.shape)
                        print("pred shape", pred.shape)
                        print("gold shape", gold.shape)
                correct += torch.sum(pred == gold).item()

            if is_test:
                print("Confusion matrix:")
                print(confusion)
                print("Report precision, recall, and f1:")

            for i in range(confusion.size()[0]):
                denominator_precision = confusion[i, :].sum().item()
                denominator_recall = confusion[:, i].sum().item()
                if denominator_precision == 0:
                    p = 0.0
                else:
                    p = confusion[i, i].item() / denominator_precision
                if denominator_recall == 0:
                    r = 0.0
                else:
                    r = confusion[i, i].item() / denominator_recall
                # p = confusion[i,i].item()/confusion[i,:].sum().item()
                # r = confusion[i,i].item()/confusion[:,i].sum().item()
                f1 = 2*p*r / (p+r)
                if i == 1:
                    label_1_f1 = f1
                print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i,p,r,f1))
            print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct/len(dataset), correct, len(dataset)))
            if metrics == 'Acc':
                return correct/len(dataset)
            elif metrics == 'f1':
                return label_1_f1
            else:
                return correct/len(dataset)
        else:
            for i, (input_ids_batch, label_ids_batch,  mask_ids_batch, pos_ids_batch, vms_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms)):

                vms_batch = torch.LongTensor(vms_batch)

                input_ids_batch = input_ids_batch.to(device)
                label_ids_batch = label_ids_batch.to(device)
                mask_ids_batch = mask_ids_batch.to(device)
                pos_ids_batch = pos_ids_batch.to(device)
                vms_batch = vms_batch.to(device)

                with torch.no_grad():
                    loss, logits = model(input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch)
                logits = nn.Softmax(dim=1)(logits)
                if i == 0:
                    logits_all=logits
                if i >= 1:
                    logits_all=torch.cat((logits_all,logits),0)

            order = -1
            gold = []
            for i in range(len(dataset)):
                qid = dataset[i][-1]
                label = dataset[i][1]
                if qid == order:
                    j += 1
                    if label == 1:
                        gold.append((qid,j))
                else:
                    order = qid
                    j = 0
                    if label == 1:
                        gold.append((qid,j))

            label_order = []
            order = -1
            for i in range(len(gold)):
                if gold[i][0] == order:
                    templist.append(gold[i][1])
                elif gold[i][0] != order:
                    order=gold[i][0]
                    if i > 0:
                        label_order.append(templist)
                    templist = []
                    templist.append(gold[i][1])
            label_order.append(templist)

            order = -1
            score_list = []
            for i in range(len(logits_all)):
                score = float(logits_all[i][1])
                qid=int(dataset[i][-1])
                if qid == order:
                    templist.append(score)
                else:
                    order = qid
                    if i > 0:
                        score_list.append(templist)
                    templist = []
                    templist.append(score)
            score_list.append(templist)

            rank = []
            pred = []
            print(len(score_list))
            print(len(label_order))
            for i in range(len(score_list)):
                if len(label_order[i])==1:
                    if label_order[i][0] < len(score_list[i]):
                        true_score = score_list[i][label_order[i][0]]
                        score_list[i].sort(reverse=True)
                        for j in range(len(score_list[i])):
                            if score_list[i][j] == true_score:
                                rank.append(1 / (j + 1))
                    else:
                        rank.append(0)

                else:
                    true_rank = len(score_list[i])
                    for k in range(len(label_order[i])):
                        if label_order[i][k] < len(score_list[i]):
                            true_score = score_list[i][label_order[i][k]]
                            temp = sorted(score_list[i],reverse=True)
                            for j in range(len(temp)):
                                if temp[j] == true_score:
                                    if j < true_rank:
                                        true_rank = j
                    if true_rank < len(score_list[i]):
                        rank.append(1 / (true_rank + 1))
                    else:
                        rank.append(0)
            MRR = sum(rank) / len(rank)
            print("MRR", MRR)
            return MRR

    # Training phase.
    print("Start training.")
    trainset = read_dataset(args.train_path, relevance_model, workers_num=args.workers_num)
    print("Shuffling dataset")
    random.shuffle(trainset)
    instances_num = len(trainset)
    batch_size = args.batch_size

    print("Trans data to tensor.")
    print("input_ids")
    input_ids = torch.LongTensor([example[0] for example in trainset])
    print("mask_input_ids")
    mask_input_ids = torch.LongTensor([example[1] for example in trainset])
    print("mask_rel_ids")
    mask_rel_ids = torch.LongTensor([example[2] for example in trainset])
    print("mask_ent_ids")
    mask_ent_ids = torch.LongTensor([example[3] for example in trainset])
    print("label_ids")
    label_ids = torch.LongTensor([example[4] for example in trainset])
    print("mask_ids")
    mask_ids = torch.LongTensor([example[5] for example in trainset])
    print("pos_ids")
    pos_ids = torch.LongTensor([example[6] for example in trainset])
    print("vms")
    vms = [example[7] for example in trainset]
    print("type")
    type_ids = torch.LongTensor([example[8] for example in trainset])
    print("tgt_word")
    tgt_word = torch.LongTensor([example[9] for example in trainset])
    print("tgt_rel")
    tgt_rel = torch.LongTensor([example[10] for example in trainset])
    print("tgt_ent")
    tgt_ent = torch.LongTensor([example[11] for example in trainset])
    print("id")
    id = [example[12] for example in trainset]
    print("concept_ent_pairs")
    concept_ent_pairs = [example[13] for example in trainset]
    print("edge_idx")
    edge_idx = [example[14] for example in trainset]

    train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup, t_total=train_steps)

    total_loss = 0.
    result = 0.0
    best_result = 0.0
    for epoch in range(1, args.epochs_num+1):
        model.train()
        for i, (input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch, type_ids_batch, tgt_word_batch, tgt_rel_batch, tgt_ent_batch, mask_input_ids_batch, mask_rel_ids_batch, mask_ent_ids_batch, id_batch, concept_ent_pairs_batch, edge_idx_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms, type_ids, id, concept_ent_pairs, edge_idx, tgt_word, tgt_rel, tgt_ent, mask_input_ids, mask_rel_ids, mask_ent_ids)):
            model.zero_grad()

            vms_batch = torch.LongTensor(vms_batch)

            print("运行了batch", i)
            input_ids_batch = input_ids_batch.to(device)
            mask_input_ids_batch = mask_input_ids_batch.to(device)
            mask_rel_ids_batch = mask_rel_ids_batch.to(device)
            mask_ent_ids_batch = mask_ent_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
            vms_batch = vms_batch.to(device)
            type_ids_batch = type_ids_batch.to(device)
            tgt_word_batch = tgt_word_batch.to(device)
            tgt_rel_batch = tgt_rel_batch.to(device)
            tgt_ent_batch = tgt_ent_batch.to(device)

            loss, _ = model(id_batch, input_ids_batch, label_ids_batch, mask_ids_batch, type_ids_batch, tgt_word_batch, tgt_rel_batch, tgt_ent_batch, mask_input_ids_batch, mask_rel_ids_batch, mask_ent_ids_batch, concept_ent_pairs_batch, edge_idx_batch, pos=pos_ids_batch, vm=vms_batch)
            if torch.cuda.device_count() > 1:
                loss = torch.mean(loss)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i+1, total_loss / args.report_steps))
                sys.stdout.flush()
                total_loss = 0.
            loss.backward()
            optimizer.step()

        print("Start evaluation on dev dataset.")
        result = evaluate(args, False)
        if result > best_result:
            best_result = result
            save_model(model, args.output_model_path)
        else:
            continue

        print("Start evaluation on test dataset.")
        evaluate(args, True)

    # Evaluation phase.
    print("Final evaluation on the test dataset.")
    evaluate(args, True)

if __name__ == "__main__":
    main()
