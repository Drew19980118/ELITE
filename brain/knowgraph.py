# coding: utf-8
"""
KnowledgeGraph
"""
import os
import random

import h5py

import brain.config as config
import pkuseg
import numpy as np
import torch
import uer.utils.constants as const
import math
import pickle
import pretrain_emb
import torch.nn as nn

class KnowledgeGraph(object):
    """
    spo_files - list of Path of *.spo files, or default kg name. e.g., ['HowNet']
    """

    def __init__(self, spo_files, predicate=True):
        self.predicate = predicate
        self.spo_file_paths = [config.KGS.get(f, f) for f in spo_files]
        self.lookup_table_CnDbpedia = self._create_lookup_table_CnDbpedia(self.spo_file_paths[0])
        # self.lookup_table_HowNet = self._create_lookup_table_HowNet(self.spo_file_paths[1])
        self.segment_vocab = list(self.lookup_table_CnDbpedia.keys()) + config.NEVER_SPLIT_TAG
        self.tokenizer = pkuseg.pkuseg(model_name="default", postag=False, user_dict=self.segment_vocab)
        self.special_tags = set(config.NEVER_SPLIT_TAG)

    def _create_lookup_table_CnDbpedia(self, spo_path):
        lookup_table = {}
        #for spo_path in self.spo_file_paths:
        print("[KnowledgeGraph] Loading spo from {}".format(spo_path))
        with open(spo_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    subj, pred, obje = line.strip().split("\t")
                except:
                    print("[KnowledgeGraph] Bad spo:", line, "from {}".format(spo_path))
                if self.predicate:
                    value = pred + obje
                else:
                    value = obje
                if subj in lookup_table.keys():
                    lookup_table[subj].add(value)
                else:
                    lookup_table[subj] = set([value])
        return lookup_table

    def _create_lookup_table_HowNet(self, spo_path):
        lookup_table = {}
        #for spo_path in self.spo_file_paths:
        print("[KnowledgeGraph] Loading spo from {}".format(spo_path))
        with open(spo_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    subj, pred, obje = line.strip().split("\t")
                except:
                    print("[KnowledgeGraph] Bad spo:", line, "from {}".format(spo_path))
                if self.predicate:
                    value = pred + obje
                else:
                    value = obje
                if subj in lookup_table.keys():
                    lookup_table[subj].add(value)
                else:
                        lookup_table[subj] = set([value])
        return lookup_table

    def add_knowledge_with_vm(self, id, sent_batch, vocab, model, max_entities=config.MAX_ENTITIES, add_pad=True, max_length=128):
        """
        input: sent_batch - list of sentences, e.g., ["abcd", "efgh"]
        return: know_sent_batch - list of sentences with entites embedding
                position_batch - list of position index of each character.
                visible_matrix_batch - list of visible matrixs
                seg_batch - list of segment tags
        """
        split_sent_batch = [self.tokenizer.cut(sent) for sent in sent_batch]
        know_sent_batch = []
        know_sent_mask_batch = []
        know_sent_mask_rel_batch = []
        know_sent_mask_ent_batch = []
        position_batch = []
        type_batch = []
        visible_matrix_batch = []
        seg_batch = []
        tgt_word_list_batch = []
        tgt_rel_list_batch = []
        tgt_ent_list_batch = []
        id_batch = []
        id_batch.append(id)
        file_path = 'pretrain_emb/load_sentence_emb.h5'
        for split_sent in split_sent_batch:
            with h5py.File(file_path, "r") as h5file:
                dataset_name = "embeddings"
                if dataset_name in h5file:
                    dataset = h5file[dataset_name]
                    text_node = torch.tensor(dataset[id])
            # create tree
            sent_tree = []
            pos_idx_tree = []
            abs_idx_tree = []
            pos_idx = -1
            abs_idx = -1
            abs_idx_src = []
            for token in split_sent:
                entities_CnD = list(self.lookup_table_CnDbpedia.get(token, []))[:max_entities]
                # entities_HowNet = list(self.lookup_table_HowNet.get(token, []))[:max_entities]
                entities = entities_CnD
                entities_save = []
                for ent in entities:
                    ent_list = [x for x in ent]
                    if len(ent_list) < max_length:
                        pad_num = max_length - len(ent_list)
                        ent_list += [config.PAD_TOKEN] * pad_num
                    else:
                        ent_list = ent_list[:max_length]
                    mask = [1 if t != const.PAD_TOKEN else 0 for t in ent_list]
                    ent_list = [vocab.get(t) for t in ent_list]
                    ent_node = model(torch.LongTensor(ent_list).unsqueeze(0), torch.LongTensor(mask))
                    ent_node = ent_node.mean(1)
                    query = ent_node
                    key = text_node
                    value = text_node
                    scores = (query * key)
                    scores = torch.matmul(scores, value.t())
                    prob = torch.sigmoid(scores)
                    if prob >= 0.5:
                        entities_save.append(ent)
                sent_tree.append((token, entities))

                if token in self.special_tags:
                    token_pos_idx = [pos_idx+1]
                    token_abs_idx = [abs_idx+1]
                else:
                    token_pos_idx = [pos_idx+i for i in range(1, len(token)+1)]
                    token_abs_idx = [abs_idx+i for i in range(1, len(token)+1)]
                abs_idx = token_abs_idx[-1]

                entities_pos_idx = []
                entities_abs_idx = []
                for ent in entities:
                    ent_pos_idx = []
                    count_pos = 0
                    for count in range(len(ent)):
                        if ent[count] == ',':
                            ent_pos_idx.append('[SPLIT]')
                        else:
                            count_pos += 1
                            ent_pos_idx.append(token_pos_idx[-1] + count_pos)
                    entities_pos_idx.append(ent_pos_idx)
                    ent_abs_idx = []
                    count_abs = 0
                    for count in range(len(ent)):
                        if ent[count] == ',':
                            ent_abs_idx.append('[SPLIT]')
                        else:
                            count_abs += 1
                            ent_abs_idx.append(abs_idx + count_abs)
                    abs_idx = ent_abs_idx[-1]
                    entities_abs_idx.append(ent_abs_idx)

                pos_idx_tree.append((token_pos_idx, entities_pos_idx))
                pos_idx = token_pos_idx[-1]
                abs_idx_tree.append((token_abs_idx, entities_abs_idx))
                abs_idx_src += token_abs_idx

            # Fetch ent and ent_idx
            concept_ent_pairs_batch = []
            edge_idx_batch = []
            concept_ent_pairs = []
            start_idx = 0
            edge_idx = []
            edge_idx_i = []
            edge_idx_j = []
            for concept, all_ent in abs_idx_tree:
                if len(all_ent) > 0:
                    concept_ent_pairs.append(concept)
                    start_idx += 1
                    edge_idx_i.append(0)
                    edge_idx_j.append(start_idx)
                    for _ in range (len(all_ent)):
                        edge_idx_i.append(start_idx)
                    for ent in all_ent:
                        concept_ent_pairs.append(ent)
                        start_idx += 1
                        edge_idx_j.append(start_idx)
            edge_idx = [edge_idx_i] + [edge_idx_j]
            edge_idx = torch.tensor(edge_idx)

            # Construct type
            type = []
            for token, rel_ent in abs_idx_tree:
                for i in range(len(token)):
                    type.append(1)
                if len(rel_ent) == 0:
                    continue
                else:
                    for i in range(len(rel_ent)):
                        split_flag = False
                        for j in range(len(rel_ent[i])):
                            if rel_ent[i][j] == '[SPLIT]':
                                split_flag = True
                            elif not (rel_ent[i][j] == '[SPLIT]') and split_flag == False:
                                type.append(10)
                            else:
                                type.append(20)

            # Get know_sent and pos
            know_sent = []
            pos = []
            seg = []
            for i in range(len(sent_tree)):
                word = sent_tree[i][0]
                if word in self.special_tags:
                    know_sent += [word]
                    seg += [1]
                else:
                    add_word = list(word)
                    know_sent += add_word
                    seg += [1] * len(add_word)
                pos += pos_idx_tree[i][0]
                for j in range(len(sent_tree[i][1])):
                    add_word = list(sent_tree[i][1][j])
                    add_word = [x for x in add_word if x != ',']
                    know_sent += add_word
                    seg += [1] * len(add_word)
                    add_pos = list(pos_idx_tree[i][1][j])
                    add_pos = [x for x in add_pos if x != '[SPLIT]']
                    pos += add_pos

            token_num = len(know_sent)

            # Construct word mask
            know_sent_mask = know_sent[:]
            tgt_word_list = [0] * len(know_sent)
            candidate_word_pos = [tup[0] for tup in abs_idx_tree[1:]]
            random.shuffle(candidate_word_pos)
            percent = 0.15
            num_elements_to_extract = int(len(candidate_word_pos) * percent)
            pred_word_pos_idx = candidate_word_pos[:num_elements_to_extract]
            all_label_word_id = []
            for pred_candidate_word_pos_idx in pred_word_pos_idx:
                label_word_id = []
                #if random.random() < 0.8:
                for pred_candidate_char_pos_idx in pred_candidate_word_pos_idx:
                    label_word_id.append(vocab.get(know_sent[pred_candidate_char_pos_idx]))
                    know_sent_mask[pred_candidate_char_pos_idx] = '[MASK]'
                all_label_word_id.append(label_word_id)

            flat_pred_word_pos_idx = [idx for sub_pred_word_pos_idx in pred_word_pos_idx for idx in sub_pred_word_pos_idx]
            flat_label_word_id = [label for sub_label_word_id in all_label_word_id for label in sub_label_word_id]
            for index, pred_word_pos_idx in enumerate(flat_pred_word_pos_idx):
                tgt_word_list[pred_word_pos_idx] = flat_label_word_id[index]

            # # Construct rel and ent mask
            know_sent_rel_mask = know_sent[:]
            know_sent_ent_mask = know_sent[:]
            tgt_rel_list = [0] * len(know_sent)
            tgt_ent_list = [0] * len(know_sent)
            candidate_ent_pos = [tup[1] for tup in abs_idx_tree[1:] if len(tup[1]) != 0]
            candidate_ent_pos = [rel_pos for sub_candidate_rel_pos in candidate_ent_pos for rel_pos in sub_candidate_rel_pos]
            random.shuffle(candidate_ent_pos)
            if len(candidate_ent_pos) > 0:
                candidate_rel_pos_idx = [rel_pos for rel_pos in candidate_ent_pos[0]]
                if len(candidate_ent_pos) > 1:
                     candidate_rel_pos_idx = candidate_rel_pos_idx + [rel_pos for rel_pos in candidate_ent_pos[1]]
                label_rel_id = []
                for rel_pos in candidate_rel_pos_idx:
                    label_rel_id.append(vocab.get(know_sent[rel_pos]))
                    know_sent_ent_mask[rel_pos] = '[MASK]'
                for index, rel_pos in enumerate(candidate_rel_pos_idx):
                    tgt_ent_list[rel_pos] = label_rel_id[index]

            # Calculate visible matrix
            visible_matrix = np.zeros((token_num, token_num))
            for item in abs_idx_tree:
                src_ids = item[0]
                for id in src_ids:
                    visible_abs_idx = abs_idx_src + [idx for ent in item[1] for idx in ent]
                    update_visible_abs_idx = [x for x in visible_abs_idx if x != '[SPLIT]']
                    visible_matrix[id, update_visible_abs_idx] = 1
                for ent in item[1]:
                    update_ent = [x for x in ent if x != '[SPLIT]']
                    for id in update_ent:
                        visible_abs_idx = update_ent + src_ids
                        visible_matrix[id, visible_abs_idx] = 1

            src_length = len(know_sent)
            if len(know_sent) < max_length:
                pad_num = max_length - src_length
                know_sent += [config.PAD_TOKEN] * pad_num
                know_sent_mask += [config.PAD_TOKEN] * pad_num
                know_sent_rel_mask += [config.PAD_TOKEN] * pad_num
                know_sent_ent_mask += [config.PAD_TOKEN] * pad_num
                seg += [0] * pad_num
                pos += [max_length - 1] * pad_num
                type += [0] * pad_num
                tgt_word_list += [0] * pad_num
                tgt_rel_list += [0] * pad_num
                tgt_ent_list += [0] * pad_num
                visible_matrix = np.pad(visible_matrix, ((0, pad_num), (0, pad_num)), 'constant')  # pad 0
            else:
                know_sent = know_sent[:max_length]
                know_sent_mask = know_sent_mask[:max_length]
                know_sent_rel_mask = know_sent_rel_mask[:max_length]
                know_sent_ent_mask = know_sent_ent_mask[:max_length]
                seg = seg[:max_length]
                pos = pos[:max_length]
                type = type[:max_length]
                visible_matrix = visible_matrix[:max_length, :max_length]
                tgt_word_list = tgt_word_list[:max_length]
                tgt_rel_list = tgt_rel_list[:max_length]
                tgt_ent_list = tgt_ent_list[:max_length]

            know_sent_batch.append(know_sent)
            know_sent_mask_batch.append(know_sent_mask)
            know_sent_mask_rel_batch.append(know_sent_rel_mask)
            know_sent_mask_ent_batch.append(know_sent_ent_mask)
            position_batch.append(pos)
            visible_matrix_batch.append(visible_matrix)
            seg_batch.append(seg)
            type_batch.append(type)
            tgt_word_list_batch.append(tgt_word_list)
            tgt_rel_list_batch.append(tgt_rel_list)
            tgt_ent_list_batch.append(tgt_ent_list)
            concept_ent_pairs_batch.append(concept_ent_pairs)
            edge_idx_batch.append(edge_idx)

        return know_sent_batch, know_sent_mask_batch, know_sent_mask_rel_batch, know_sent_mask_ent_batch, position_batch, visible_matrix_batch, type_batch, tgt_word_list_batch, tgt_rel_list_batch, tgt_ent_list_batch, id_batch, concept_ent_pairs_batch, edge_idx_batch, seg_batch

