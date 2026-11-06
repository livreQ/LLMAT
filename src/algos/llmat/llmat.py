import warnings

warnings.filterwarnings("ignore")

from transformers import logging
from transformers.tokenization_utils_base import BatchEncoding

logging.set_verbosity_error()

import torch
from torch import nn, optim
import math
import pandas as pd
from benchmarks.MAT.data_processor import DataProcessor
from peft import PeftModel
import tqdm
from contextlib import nullcontext
from copy import deepcopy
from transformers import get_scheduler
from typing import *
from botorch.posteriors import DeterministicPosterior
import numpy as np
from src.algos.llmat.node import Node
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.nn.init as init
import random
import pingouin as pg
from scipy import stats
from src.utils.helpers import trace_times


class FTDataset(Dataset):

    def __init__(self, dataframe, cluster_col=None):
        self.dataframe = dataframe
        self.cluster_col = cluster_col
        print("DataFrame columns:", self.dataframe.columns)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        # print("Row:", row)

        # # Extract features and targets from the batch (list of DataFrame rows)
        features = torch.tensor(row["features"])

        SMILES = row["SMILES"]

        #Include other columns if needed (e.g., metadata, clusters)

        if "Entry Number" in self.dataframe.columns:
            index = row["Entry Number"]
            targets = torch.tensor(row["targets_transformed"])
            clusters = torch.tensor(row[self.cluster_col])
            return {
                "Entry Number": index,
                "features": features,
                "targets": targets,
                "SMILES": SMILES,
                "clusters": clusters,
            }
        else:
            targets = torch.tensor(row["targets"])
            clusters = torch.tensor(row["clusters"])
            labels = torch.tensor(row["labels"])
            return {
                "features": features,
                "targets": targets,
                "SMILES": SMILES,
                "labels": labels,
                "clusters": clusters,
            }


class LLMAT(object):
    """
    Amortized Learning of Partition Hierarchies (ALPHA) for Bayesian Optimization.
    Density ratio estimation with the classification head and the LoRA weights.
    """

    def __init__(
            self,
            peft_model: PeftModel,
            training_set: pd.DataFrame,
            data_processor: DataProcessor,
            head_cfg: dict,  # type: ignore
            lora_cfg: dict,  # type: ignore
            device: str = "cuda",
            dtype: str = "float32",
            append_eos: bool = True,
            gamma: float = 0.33,  # quantile
            acf_type: str = "pi",
            lmbda: float = 1,  # ucb weight
            eta: float = 0.5,  # meta learning rate
            tree_depth: int = 3,
            finetuning: bool = False,
            reinit: bool = False,
            threshold: float = 0.3,  # threshold for partitioning
            alpha: float = 1.0,  # for ei, pi, etc...
            feature_dim: int = 1024,  # feature dimension
            selection_type: str = "uct",  # "uct" or "greedy"
    ):

        # ========== Misc ==========
        self.dtype = dtype
        self.ptdtype = {"float32": torch.float32,\
                        "bfloat16": torch.bfloat16,\
                        "float16": torch.float16}[dtype]
        self.ctx = (nullcontext() if device == "cpu" or device == "mps" \
                                  else torch.amp.autocast(device_type="cuda", dtype=self.ptdtype))
        self.enable_grad_scaler = dtype in ["float16", "bfloat16"]
        self.finetuning = finetuning  # peft or
        self.reinit = reinit  # reinitialize the classifier head after each iteration
        # ========== PEFT ===========

        self.append_eos = append_eos
        self.peft_model = peft_model.to(device, dtype=self.ptdtype)
        self.feature_dim = feature_dim
        #self.feature_dim = peft_model.base_model.feature_dim
        self.last_hidden_size = 50
        self.clf_head = nn.Sequential(nn.Linear(self.feature_dim, self.last_hidden_size), \
                                  nn.ReLU(),
                                  nn.Linear(self.last_hidden_size, 1))
        self.reg_head = nn.Sequential(nn.Linear(self.feature_dim, self.last_hidden_size), \
                                nn.ReLU(),
                                nn.Linear(self.last_hidden_size, 1))
        #   nn.Sigmoid())
        self.lora_cfg = lora_cfg
        self.training_set = training_set
        self.data_processor = data_processor
        self.device = device
        ## ========= meta-learning ===========
        self.theta = self.clf_head.state_dict()
        self.eta = eta
        ## ========= MCTS ===========
        self.lmbda = lmbda
        self.tree_depth = tree_depth
        self.selection_type = selection_type
        self.nodes = []
        self.head_cfg = head_cfg
        root = Node(parent=None, reset_id=True, clf=self.clf_head.state_dict())
        self.nodes.append(root)
        self.ROOT = root
        self.CURT = self.ROOT
        self.LEAF_SAMPLE_SIZE = head_cfg["leaf_sample_size"]
        self.threshold = threshold
        self.node_train_logs = []

        ## ========= BO ===========
        self.gamma = gamma
        self.acf_type = acf_type
        self.alpha = alpha  # for ei, pi, etc...
        # ========= init clf head =======
        self.init_weights_xavier()

    def freeze_base_model(self):
        # Freeze base model parameters
        for param in self.peft_model.base_model.parameters():
            param.requires_grad = False

    def freeze_clf_head(self):
        """Freeze the classifier head (only train LoRA layers)."""
        for param in self.clf_head.parameters():
            param.requires_grad = False

    def freeze_lora(self):
        """Freeze the LoRA layers (only train the classifier)."""
        for param in self.peft_model.parameters():
            param.requires_grad = False
        """Freeze the regression head."""
        for param in self.reg_head.parameters():
            param.requires_grad = False

        # for n, p in self.peft_model.named_parameters():
        #     if "lora" in n:
        #         p.requires_grad = False

    def unfreeze_clf_head(self):
        """Unfreeze the classifier head."""
        for param in self.clf_head.parameters():
            param.requires_grad = True

        # for name, param in self.clf_head.named_parameters():
        #     if param.grad is None:
        #         print(f"No gradient for {name}")

    def unfreeze_lora(self):
        """Unfreeze the LoRA layers."""
        for param in self.peft_model.parameters():
            param.requires_grad = True
        """unfreeze the regression head."""
        for param in self.reg_head.parameters():
            param.requires_grad = True
        # for n, p in self.peft_model.named_parameters():
        #     if "lora" in n:
        #         p.requires_grad = True

    def init_weights_xavier(self):
        for param in self.clf_head.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)  # Xavier initialization

    def restore_finetuning(self):
        """
            Restore the initial LoRA state for finetuning.
        """
        self._initial_lora_state = {}
        self._initial_reg_head = self.reg_head.state_dict()
        for name, param in self.peft_model.named_parameters():
            if 'lora' in name:
                self._initial_lora_state[name] = param.data.cpu().clone()

    def reset_lora(self):
        for name, param in self.peft_model.named_parameters():
            if 'lora' in name and name in self._initial_lora_state:
                param.data.copy_(self._initial_lora_state[name].to(param.device))

    def get_traindata_weight(self, train_dataset=None):
        if train_dataset is None:
            train_dataset = self.data_processor.dataset
        # print(self.training_set)
        y_reg = train_dataset['targets']
        y_cluster = train_dataset['clusters']
        if "features" in train_dataset:
            x = train_dataset['features']
        else:
            x = train_dataset['SMILES']
        x_ori = train_dataset['SMILES']
        tau = np.quantile(y_reg, q=1 - self.gamma)  #np.quantile(y_reg, q=1 - self.gamma)
        y_class = np.greater_equal(y_reg, tau)
        print("train length:", len(y_reg))
        print("tau:", tau)
        print("-----------------")
        print("reg targets:", y_reg)
        print("class labels:", y_class)
        print("------- finish get weight----------")

        return x_ori, x, y_class, y_cluster, y_reg, tau

    def _weighted_vi_loss(self, y_class, y_reg, y_pred, tau):
        # print(y_class)
        # print("y_pred:", y_pred.requires_grad)  # Check if `y_pred` has `requires_grad=True`

        positive_mask = y_class == 1
        negative_mask = y_class == 0
        # negative_mask = torch.ones_like(y_class)
        if self.acf_type == "ei":
            sample_weight = (y_reg - tau)[positive_mask].to(self.device, torch.float32)  #.to(self.device, torch.float32)
        elif self.acf_type == "pi":
            sample_weight = torch.ones_like(positive_mask)[positive_mask]  #.to(self.device, torch.float32)
        else:
            sample_weight = torch.ones_like(positive_mask)[positive_mask]  #.to(self.device, torch.float32)
        # sample_weight = sample_weight.to(self.device, torch.float32)

        if sample_weight.sum() == 0:
            print("sample weight is zero, using uniform weights")
            sample_weight = torch.ones_like(positive_mask)[positive_mask].to(self.device, torch.float32)
        else:
            sample_weight = sample_weight / sample_weight.sum()
        #print("sample weight", sample_weight)
        y_pred = y_pred.to(self.device)  #.to(self.device, torch.float32)#logits
        #print("logits:", y_pred)
        #print("prob:", F.sigmoid(y_pred))
        # print(sample_weight)
        # Compute log probabilities
        # print(positive_mask)
        log_prob_positive = self.gamma * torch.mul(sample_weight, F.logsigmoid(y_pred[positive_mask])).mean()
        log_prob_negative = F.logsigmoid(-y_pred[negative_mask]).mean()
        #print("gamma", self.gamma)
        # Final loss
        loss = log_prob_positive + log_prob_negative
        loss = -loss  # Negate because we want to minimize
        #print("Loss requires grad:", loss.requires_grad)  # Should also be
        return loss

    def _weighted_vi_loss2(self, y_class, y_reg, y_pred, tau):
        # print(y_class)
        # print("y_pred:", y_pred.requires_grad)  # Check if `y_pred` has `requires_grad=True`

        positive_mask = y_class == 1
        negative_mask = y_class == 0
        # negative_mask = torch.ones_like(y_class)
        if self.acf_type == "ei":
            pos_weight = (y_reg - tau)[positive_mask].to(self.device, torch.float32)  #.to(self.device, torch.float32)
            pos_weight = pos_weight**self.alpha
        elif self.acf_type == "pi":
            pos_weight = torch.ones_like(positive_mask)  #.to(self.device, torch.float32)
        else:
            pos_weight = torch.ones_like(positive_mask)  #.to(self.device, torch.float32)
        # sample_weight = sample_weight.to(self.device, torch.float32)

        if pos_weight.sum() == 0:
            print("sample weight is zero, using uniform weights")
            pos_weight = torch.ones_like(positive_mask)[positive_mask].to(self.device, torch.float32)
        else:
            pos_weight = pos_weight / pos_weight.mean()

        neg_weight = torch.ones_like(negative_mask)[negative_mask].to(self.device, torch.float32)
        w = torch.cat([self.gamma * pos_weight, (1 - self.gamma) * neg_weight], dim=0)
        w = w / torch.mean(w)
        pos_weight = w[positive_mask].to(self.device, torch.float32)
        neg_weight = w[negative_mask].to(self.device, torch.float32)

        #print("sample weight", sample_weight)
        y_pred = y_pred.to(self.device)  #.to(self.device, torch.float32)#logits
        #print("logits:", y_pred)
        #print("prob:", F.sigmoid(y_pred))
        # print(sample_weight)
        # Compute log probabilities
        # print(positive_mask)
        log_prob_positive = torch.mul(pos_weight, F.logsigmoid(y_pred[positive_mask])).mean()
        log_prob_negative = torch.mul(neg_weight, F.logsigmoid(-y_pred[negative_mask])).mean()
        #log_prob_positive_neg = torch.mul(pos_weight, F.logsigmoid(-y_pred[positive_mask])).mean()
        # log(1-C(x)), logsigmoid = log(1/(1+exp(-x)))
        #print("gamma", self.gamma)
        # Final loss
        loss = log_prob_positive + log_prob_negative  #+ log_prob_positive_neg
        loss = -loss  # Negate because we want to minimize
        #print("Loss requires grad:", loss.requires_grad)  # Should also be
        return loss

    def _weighted_vi_meta_loss(self, sample_weight, y_pred):

        #print("sample weight", sample_weight)
        y_pred = y_pred.to(self.device)  #.to(self.device, torch.float32)#logits
        log_prob_positive = self.gamma * torch.mul(sample_weight, F.logsigmoid(y_pred)).mean()
        log_prob_negative = F.logsigmoid(-y_pred).mean()
        #   log_prob_negative = F.logsigmoid(-y_pred[negative_mask]).mean()
        # log(1-C(x)), logsigmoid = log(1/(1+exp(-x)))
        #print("gamma", self.gamma)
        # Final loss
        loss = log_prob_positive + log_prob_negative
        loss = -loss  # Negate because we want to minimize
        #print("Loss requires grad:", loss.requires_grad)  # Should also be
        return loss

    def get_weighted_dataset(self, train_dataset, tau):
        """
            patch the positive samples and all the samples with positive labels and negative labels respectively. 
        """
        assert self.acf_type in ["pi", "ei"]
        assert train_dataset is not None
        # train_dataset = self.data_processor.dataset
        # print(self.training_set)
        y_reg = train_dataset['targets'].tolist()
        y_cluster = train_dataset['clusters'].tolist()
        x_ori = train_dataset['SMILES'].tolist()

        if self.acf_type == "ei":
            X_ori, Y_reg, Y_cluster = np.hstack(x_ori), np.hstack(y_reg), np.hstack(y_cluster)
            # tau = np.quantile(Y_reg, q=self.gamma)
            z = np.greater(Y_reg, tau)
            x_ori1, y_reg1, y_cluster1 = X_ori[z], Y_reg[z], Y_cluster[z]
            x_ori0, y_reg0, y_cluster0 = X_ori, Y_reg, Y_cluster
            w1 = (tau - Y_reg)[z]
            w1 = w1 / np.mean(w1)
            w0 = 1 - np.zeros_like(z)

            x_ori_new = np.concatenate([x_ori1, x_ori0], axis=0)
            # y_class = np.concatenate([y_class1, y_class0], axis=0)
            y_cluster_new = np.concatenate([y_cluster1, y_cluster0], axis=0)
            y_reg_new = np.concatenate([y_reg1, y_reg0], axis=0)
            s1 = x_ori1.shape[0]
            s0 = x_ori0.shape[0]

            w = np.concatenate([w1 * (s1 + s0) / s1, w0 * (s1 + s0) / s0], axis=0)
            w = w / np.mean(w)
            return x_ori_new, y_cluster_new, y_reg_new, w

        if self.acf_type == "pi":
            # x, y = np.vstack(self.features), np.hstack(self.targets)
            x_ori_new, y_reg_new, y_cluster_new = np.hstack(x_ori), np.hstack(y_reg), np.hstack(y_cluster)
            # tau = np.quantile(Y_reg, q=self.gamma)
            y_class = np.greater(Y_reg, tau)
            w = np.ones_like(y_class)
            return x_ori_new, y_cluster_new, y_reg_new, w

    def get_weighted_dataset2(self, train_dataset, tau):
        """
            non-patch way
        """
        assert self.acf_type in ["pi", "ei"]
        assert train_dataset is not None
        # train_dataset = self.data_processor.dataset
        # print(self.training_set)
        y_reg = train_dataset['targets'].tolist()
        y_cluster = train_dataset['clusters'].tolist()
        x_ori = train_dataset['SMILES'].tolist()

        X_ori, Y_reg, Y_cluster = np.hstack(x_ori), np.hstack(y_reg), np.hstack(y_cluster)
        # tau = np.quantile(Y_reg, q=self.gamma)
        z = np.greater(Y_reg, tau)

        positive_mask = z == True

        if self.acf_type == "ei":
            pos_weight = np.where(Y_reg > tau, Y_reg - tau, 0)  #(Y_reg - tau)[positive_mask].to(self.device, torch.float32)  #.to(self.device, torch.float32)
            pos_weight = pos_weight**self.alpha
            pos_weight = pos_weight
        elif self.acf_type == "pi":
            pos_weight = torch.ones_like(positive_mask)  #.to(self.device, torch.float32)
        else:
            pos_weight = torch.ones_like(positive_mask)  #.to(self.device, torch.float32)
        # sample_weight = sample_weight.to(self.device, torch.float32)

        assert pos_weight.sum() != 0
        pos_weight = pos_weight / pos_weight.mean()

        if self.acf_type == "ei":
            pos_weight = self.gamma * pos_weight

        return X_ori, Y_cluster, Y_reg, pos_weight

    def pad_dataframe(self, df, batch_size):
        dataset_size = len(df)
        remainder = dataset_size % batch_size

        if remainder == 0:
            return df

        samples_to_add = batch_size - remainder

        # Randomly sample rows to duplicate
        random_indices = np.random.choice(df.index, size=samples_to_add, replace=True)
        padding_rows = df.loc[random_indices]

        # Concatenate original DataFrame with padding rows
        padded_df = pd.concat([df, padding_rows], ignore_index=True)

        return padded_df

    def construct_metaset(self, node):
        """
        Construct the meta dataset for the given node.
        """
        data_dict = {}
        x_ori, y_cluster, y_reg, w = self.get_weighted_dataset(node.dataset, node.tau)
        #print(x_ori, y_cluster, y_reg, w)
        #print(x_ori.shape, y_cluster.shape, y_reg.shape, w.shape)
        data_dict["SMILES"] = x_ori
        data_dict["targets"] = y_reg
        data_dict["clusters"] = y_cluster
        data_dict["weights_0"] = w
        train_dataset = pd.DataFrame(data_dict)

        return self.pad_dataframe(train_dataset, self.lora_cfg['batch_size'])

    def construct_mn_metaset(self):
        """
         construct the meta dataset for all the nodes
        """
        data_dict = {}
        cnt = 0
        for node in self.nodes:
            if node.clf is None:
                continue
            x_ori, y_cluster, y_reg, pos_w = self.get_weighted_dataset2(self.ROOT.dataset, node.tau)
            print(x_ori, y_cluster, y_reg, pos_w)
            print(x_ori.shape, y_cluster.shape, y_reg.shape, pos_w.shape)
            if cnt == 0:
                data_dict["SMILES"] = x_ori
                # data_dict["features"] = x
                data_dict["targets"] = y_reg
                data_dict["clusters"] = y_cluster
                # data_dict["labels_" + str(node.id)] = y_class
                # data_dict["weights"] = pos_w
                data_dict["weights_" + str(node.id)] = pos_w
            else:
                # data_dict["labels_" + str(node.id)] = y_class
                data_dict["weights_" + str(node.id)] = pos_w
            cnt += 1
        print("data_dict:", data_dict)
        train_dataset = pd.DataFrame(data_dict)
        return train_dataset

    def train_model(self):
        if self.finetuning:
            self.freeze_base_model()
        # train local binary classifiers for partitioning and acquisition functions simutaneously
        self.tree_construction()

    def cal_acqfs(self, cands, path):
        """
            Compute the acqf values of the candidates falling into the leaf node.
            cands: all the unobserved candidates within the selected clusters, pandas dataframe
            path: list of tuples (node, choice) representing the path to the leaf node
        """
        self.freeze_clf_head()
        if self.finetuning:
            self.freeze_lora()
        # print("=========candidates before selection=====")
        # print(cands['SMILES'])

        # if self.finetuning:
        #     # print(cands)
        #     #assert '__index_level_0__' in cands
        #     # index_col = "targets"  #'__index_level_0__'
        # else:
        index_col = 'Entry Number'
        # print(cands['targets'])
        total = len(cands[index_col].tolist())
        assert len(path) > 0
        # we remember all the candidates indexes on the nodes along the selected path
        # this is used to avoid final void promising candidates
        node_acqfs = {}

        for node, choice in path:
            print("node id ", node.id)
            cands_idx = cands[index_col].tolist()
            # print(cands_idx)
            if len(cands_idx) == 0:
                print("no valid candidates in this node")
                node_acqfs[node.id] = (node.tau, 0, cands_idx, torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]))
                return node_acqfs

            assert len(cands[index_col].tolist()) > 0

            #print("==========candidates before selection============")
            # print(cands['SMILES'])
            #print(cands['targets'])

            targets = cands["targets"]  #y_reg
            # print("==========targets============")
            # print(targets)
            ###  predict the candidates fall in current node
            self.clf_head.load_state_dict(node.clf)
            if self.finetuning and node.parent is None:
                # only forward onece at the root node
                feat = self.peft_model.forward_features(cands.to(self.device))
                cands["features"] = feat
            else:
                print("already forward features")
                feat = cands["features"]
            logits = self.clf_head(feat.to(self.device))
            # print("logits:", pred)
            cx = torch.nn.functional.sigmoid(logits)  # pred_probs
            # print("==========pred sigmoids============")
            # print(cx)
            acq_vals = (cx / (1 - cx))
            ratio = len(cands_idx) / float(total)
            print("nb of cands of node {}:{}".format(node.id, len(cands_idx)))
            print("ratio compare to root node:", ratio)
            # store the candidates indexes and acqfs on the current node
            node_acqfs[node.id] = (node.tau, ratio, cands_idx, acq_vals, targets.unsqueeze(0), cx, cands['features'])
            #======= partition the candidates to the selected kid node =======
            if choice == 0:  # left kid
                idx = torch.where(cx > node.threshold)[0]
            else:  # right kid
                idx = torch.where(cx <= node.threshold)[0]

            if self.finetuning:
                cands = BatchEncoding({key: value[idx.to(value.device)] for key, value in cands.items()})
            else:
                idx = idx.tolist()
                cands = {
                    'Entry Number': cands['Entry Number'][idx],
                    'features': cands['features'][idx],
                    'targets': cands['targets'][idx],
                    'SMILES': [cands['SMILES'][i] for i in idx],
                    'clusters': cands['clusters'][idx]
                }
            # print("======candidates data after selection=====")
            # # print(cands['SMILES'])
            # print(cands['targets'])
            # cands_idx = cands[index_col].tolist()

        assert len(cands_idx) <= total
        assert len(cands_idx) > 0
        #print("=========candidates after selection=====")
        # print(cands['SMILES'])
        #print(cands['targets'])
        cands_idx = cands[index_col].tolist()
        ratio = len(cands_idx) / float(total)
        #print("ratio compare to root node:", ratio)
        return node_acqfs

    def condition_on_observations(self, obs):
        ### will retrain
        print("====before add new data====")
        print(self.training_set)
        self.training_set = self.training_set._append(obs)
        print("====after add new data====")
        print(self.training_set)
        if self.reinit:
            self.init_weights_xavier()
        if self.finetuning:
            start_meta, end_meta = trace_times(None, None, device=self.device)
            self._meta_finetune_lora()
            print("time for meta training:", trace_times(start_meta, end_meta, self.device))
        self.train_model()

    def _meta_finetune_lora(self, root_only=True, train_reg=True, train_meta=True):
        """
        Finetune the LoRA layers of the model in a meta fasion.
        """
        cfg = self.lora_cfg
        model = self.peft_model.to(self.device)
        self.reg_head.load_state_dict(self._initial_reg_head)  # load initial reg head
        self.clf_head.load_state_dict(self.theta)  # load meta-model theta
        self.reg_head = self.reg_head.to(self.device)
        self.clf_head = self.clf_head.to(self.device)
        self.freeze_clf_head()  # freeze the classifier head
        self.unfreeze_lora()  # unfreeze the LoRA layers
        # for n, p in model.named_parameters():
        #     if "lora" not in n:
        #         p.requires_grad = False
        self.reset_lora()
        lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora" in n]
        # head_params = [p for n, p in self.reg_head.named_parameters() if p.requires_grad]
        print("meta learning rate:", cfg["lr"])
        print("meta batch size:", cfg["batch_size"])
        if root_only:
            meta_train_dataset = self.construct_metaset(self.ROOT)
        else:
            meta_train_dataset = self.construct_mn_metaset()


        train_loader = self.data_processor.get_dataloader(meta_train_dataset,\
                            batch_size=cfg["batch_size"],
                            shuffle=True,
                            append_eos=self.append_eos)
        num_training_steps = cfg["n_epochs"] * len(train_loader)
        print("number of training steps:", num_training_steps)

        optimizer_lora = optim.AdamW(lora_params, lr=float(cfg["lr"]), weight_decay=5e-4)
        if train_reg:
            optimizer_head = torch.optim.Adam(self.reg_head.parameters(), lr=float(cfg["lr"]), weight_decay=5e-4)
            scheduler_head = get_scheduler(
                name="linear",
                optimizer=optimizer_head,
                num_warmup_steps=0,  #10,  #num_training_steps,
                num_training_steps=num_training_steps,
            )
            reg_func = nn.MSELoss(reduction="none")  # use MSELoss for regression

        loss_func = nn.BCEWithLogitsLoss(reduction="none")  # use BCEWithLogitsLoss for binary classification

        scheduler_lora = get_scheduler(name="linear",\
                                       optimizer=optimizer_lora,
                                       # num_warmup_steps=0.06*num_training_steps,  # Following the warmup ratio in LoRA paper
                                       num_warmup_steps=0,
                                       num_training_steps=num_training_steps)

        scaler = torch.cuda.amp.GradScaler(enabled=self.enable_grad_scaler)
        print("======== start meta finetuning ==========")
        print("meta train dataset:", meta_train_dataset)
        for _ in tqdm.trange(cfg["n_epochs"], position=1, leave=False, desc="[Training]", colour="blue"):
            # pass all the training data through the tree in a batch fashion
            for batch in train_loader:
                model.train()
                with self.ctx:
                    #### forward use original x as input, only calculate once
                    feat = model(batch.to(self.device))
                    cnt_nodes = 0
                    avg_loss = 0
                    for node in self.nodes:
                        if node.clf is None:
                            continue
                        if root_only and node.parent is not None:
                            # only finetune lora given the root node
                            continue
                        print("node tau:", node.tau)
                        print("node threshold:", node.threshold)
                        if train_meta:
                            if not root_only:
                                # only finetune lora given the root node
                                # self.clf_head.load_state_dict(node.clf)
                                clf_temp = deepcopy(self.clf_head)
                                clf_temp.load_state_dict(node.clf)
                                y_pred = clf_temp(feat)
                                # loss = self._weighted_vi_meta_loss()
                                pos_weight = batch["weights_" + str(node.id)].to(self.device, non_blocking=True)
                                weighted_loss = self._weighted_vi_meta_loss(pos_weight, y_pred)
                                # loss = self._weighted_vi_loss(y_class, y_reg, y_pred, node.tau)  # weigh
                            else:
                                y_pred = self.clf_head(feat.to(self.device))
                                y_reg = batch["targets"].to(self.device, non_blocking=True)  #.to(self.device, torch.float32, non_blocking=True)
                                y_class = torch.greater_equal(y_reg, self.ROOT.tau).unsqueeze(-1).float()
                                losses = loss_func(y_pred, y_class)  # shape: [B]
                                weighted_loss = (losses * batch["weights_0"]).mean()
                            avg_loss += weighted_loss

                        if train_reg:
                            y_reg = batch["targets"].to(self.device, non_blocking=True)  #.to(self.device, torch.float32, non_blocking=True)
                            y_pred2 = self.reg_head(feat.to(self.device))
                            loss_reg = reg_func(y_pred2, y_reg)  # shape: [B]
                            avg_loss += cfg["reg"] * loss_reg.mean()

                        # if use_pred:
                        #     y_class = torch.greater_equal(y_pred, node.threshold).unsqueeze(-1).float()
                        # else:
                        cnt_nodes += 1
                    avg_loss /= cnt_nodes
                if train_meta or train_reg:
                    scaler.scale(avg_loss).backward()
                    if float(cfg["grad_clip"]) != 0.0:
                        scaler.unscale_(optimizer_lora)
                        torch.nn.utils.clip_grad_norm_(lora_params, float(cfg["grad_clip"]))

                    if train_meta:
                        print("meta train loss (clf):", weighted_loss.item())
                    if train_reg:
                        print("meta train loss (reg):", loss_reg.mean().item())

                    scaler.step(optimizer_lora)
                    if train_reg:
                        scaler.step(optimizer_head)
                    scaler.update()
                    scheduler_lora.step()
                    if train_reg:
                        scheduler_head.step()
                    optimizer_lora.zero_grad(set_to_none=True)
                    if train_reg:
                        optimizer_head.zero_grad(set_to_none=True)
                    print("meta train loss:", avg_loss.item())

        model.eval()
        print("======== end meta finetuning ==========")
        # return model

    def _get_node_depth(self, node):
        depth = 0
        cur = node
        while cur.parent is not None:
            depth += 1
            cur = cur.parent
        return depth

    def _finetune_head_clf(self, train_dataset, node):
        tau = node.tau
        print("======== start finetuning node " + str(node.id) + " ==========")
        cfg = self.head_cfg
        print(cfg)
        print(cfg["lr"])
        ft_dataset = FTDataset(train_dataset)
        train_loader = DataLoader(
            ft_dataset,
            batch_size=cfg["batch_size"],
            shuffle=True,
            # collate_fn=collate_fn,
            # collate_fn=DataCollatorWithPadding(self.data_processor.tokenizer),
        )

        self.freeze_lora()  # freeze the LoRA layers
        # load the meta-model theta
        if node.parent is not None:
            self.clf_head.load_state_dict(node.parent.clf)
        else:
            self.clf_head.load_state_dict(self.theta)
        self.clf_head = self.clf_head.to(self.device)
        print(self.device)
        # assert self.device == "cuda"
        self.unfreeze_clf_head()  # unfreeze the classifier head

        num_training_steps = cfg["n_epochs"] * len(train_loader)
        # optimizer_head = optim.AdamW(self.clf_head.parameters(), lr=float(cfg["lr"]))  #1e-3 weight_decay=5e-4'
        lr = float(cfg["lr"])
        # if self.finetuning:
        #     lr = float(cfg["lr"]) * len(train_dataset) / 10.0
        optimizer_head = torch.optim.Adam(self.clf_head.parameters(), lr=lr, weight_decay=5e-4)
        scheduler_head = get_scheduler(
            name="cosine",
            optimizer=optimizer_head,
            num_warmup_steps=10,  #num_training_steps,
            num_training_steps=num_training_steps,
        )
        ######---------finue-tune classifier head--------######
        pred = []
        p_samples = pd.DataFrame()
        n_samples = pd.DataFrame()
        node_depth = self._get_node_depth(node)
        epoch_losses = []
        for epoch in tqdm.trange(cfg["n_epochs"], position=1, leave=False, desc="[Training]", colour="blue"):
            avg_loss = []

            for batch in train_loader:
                # print(batch)
                y_class = batch["labels"].to(self.device, dtype=torch.float32, non_blocking=True)
                y_reg = batch["targets"].to(self.device, dtype=torch.float32, non_blocking=True)
                # labels = torch.greater(labels, self.tau)
                # labels = torch.tensor(labels, dtype=torch.float32)
                # print("reg", y_reg)
                # # print("pred", y_pred)
                # print("class label", y_class)
                # print(y_class)
                with self.ctx:
                    optimizer_head.zero_grad()
                    y_pred = self.clf_head(batch["features"].to(self.device)).squeeze(1)  # logits
                    # print("Parameters in optimizer:")
                    # for group in optimizer_head.param_groups:
                    #     print(group["params"])

                    # print(batch["features"])

                    # feat = model.forward_features(data)
                    # print(outputs.shape, labels.shape); input()
                    # print("=========")
                    # print(batch)
                    # print("reg", y_reg)
                    # print("pred", y_pred)
                    # print("class label", y_class)
                    loss = self._weighted_vi_loss(y_class, y_reg, y_pred, tau)  # weights
                    # print("loss:", loss)
                    avg_loss.append(loss.item())
                    loss.backward()
                    # for name, param in self.clf_head.named_parameters():
                    #     if param.grad is None:
                    #         print(f"No gradient for {name}")
                    optimizer_head.step()
                if epoch == cfg["n_epochs"] - 1:
                    # pred += (y_pred > 0).int()
                    pred = F.sigmoid(y_pred)
                    print("======= predicted sigmoids of train data ====")
                    print("predicted prob", pred)
                    pos_inds = torch.where(pred >= node.threshold)[0]
                    neg_inds = torch.where(pred < node.threshold)[0]
                    # print(pos_inds, neg_inds)
                    batch['features'] = batch['features'].tolist()
                    pd_batch = pd.DataFrame(batch)
                    # print(pd_batch[["targets", "labels"]])
                    p_samples = pd.concat([p_samples, pd_batch.iloc[pos_inds.cpu().numpy()]], ignore_index=True)
                    n_samples = pd.concat([n_samples, pd_batch.iloc[neg_inds.cpu().numpy()]], ignore_index=True)
                    # print("positive samples:", p_samples)

                # scaler.scale(loss).backward()
                # scaler.step(optimizer_head)
                # scaler.update()
            scheduler_head.step()
            epoch_loss = np.mean(avg_loss)
            epoch_losses.append(epoch_loss)
            print("clf train loss:", epoch_loss)
        pos_mean = float(p_samples["targets"].mean()) if len(p_samples) > 0 else float("nan")
        neg_mean = float(n_samples["targets"].mean()) if len(n_samples) > 0 else float("nan")
        self.node_train_logs.append({
            "node_id": node.id,
            "depth": node_depth,
            "n_samples": len(train_dataset),
            "epoch_losses": epoch_losses,
            "pos_mean": pos_mean,
            "neg_mean": neg_mean,
            "n_pos": len(p_samples),
            "n_neg": len(n_samples),
        })
        print("positive samples:", p_samples[["targets", "labels"]])
        print("negative samples:", n_samples[["targets", "labels"]])
        print("======== end finetuning ==========")
        # if len(np.unique(pred > node.threshold)) == 1:
        if len(p_samples) == 0 or len(n_samples) == 0:
            print("not splited")
            # print(pred)
            return False, p_samples, n_samples
        else:
            return True, p_samples, n_samples

        # node.clf = self.clf_head.state_dict()

    def init_nodes_with_train_data(self):
        print("======== init root node ==========")
        # evaluate the nodes in previous iteration
        self.ROOT.obj_counter = 0
        for node in self.nodes:
            if node is not None:
                node.clear_data()
        self.nodes.clear()
        # self.training_set is the list of pd.Series
        print("========= raw training set =========")
        print(self.training_set)
        print(self.training_set.columns)
        if self.finetuning:
            print("fineuntuning, re-extract features from the training set")
            train_loader = self.data_processor.get_dataloader(self.training_set,\
                                                        batch_size=self.lora_cfg["batch_size"],
                                                        shuffle=False,
                                                        append_eos=self.append_eos)
            # this modifies self.data_processor.dataset

            features = []
            for data in train_loader:
                # print(data)
                with torch.no_grad():
                    feat = self.peft_model.forward_features(data)
                    features.append(feat)
            train_dataset = pd.DataFrame({
                "features": torch.cat(features, dim=0).tolist(),
                "SMILES": self.training_set["SMILES"],
                "targets": self.data_processor.dataset['targets'],  # tranformed targets in data_processor
                "clusters": self.data_processor.dataset["clusters"],
            })
        else:
            train_dataset = pd.DataFrame({
                "features": self.training_set["features"],
                "SMILES": self.training_set["SMILES"],
                "targets": self.training_set['targets_transformed'],
                "clusters": self.training_set[self.data_processor.cluster_col],
            })
        # print(torch.cat(features, dim=0))
        # print(self.data_processor.dataset['labels'].squeeze())
        # if self.clustering_type == "kmeans":
        #     cluster_col = "clusters"
        # else:
        #     cluster_col = "llm_clusters"

        # train_dataset = pd.DataFrame(self.training_set)
        # # print(len(train_dataset), torch.cat(features, dim=0).shape)
        # train_dataset["features"] = torch.cat(features, dim=0).tolist()
        # print(train_dataset)

        print("========= processed training set with features =========")
        print(train_dataset)
        print(train_dataset.drop(columns=["features"]))
        print(train_dataset.columns)
        # train_dataset = self.data_processor.dataset
        # train_dataset["features"] = torch.cat(features, dim=0)
        new_root = Node(parent=None, reset_id=True, tau=None, clf=None, is_splitable=True, threshold=self.threshold)
        self.nodes.append(new_root)
        self.ROOT = new_root
        self.CURT = self.ROOT
        self.ROOT.update_dataset(train_dataset)
        print("======== finish init root node ==========")
        # update the root with the training data

    def get_leaf_status(self):
        status = []
        n_nodes = len(self.nodes)
        for node in self.nodes:
            if node.is_leaf() == True and node.n > self.LEAF_SAMPLE_SIZE \
                and node.is_splitable == True:
                n_nodes += 2
                if n_nodes <= np.power(2, self.tree_depth) - 1:
                    status.append(True)
                else:
                    status.append(False)
            else:
                status.append(False)
        return np.array(status)

    def is_splitable(self):
        status = self.get_leaf_status()
        print("leaf status:", status)
        if True in status:
            return True
        else:
            return False

    def get_split_idx(self):
        splitable_nodes_idx = np.argwhere(self.get_leaf_status() == True).reshape(-1)
        return splitable_nodes_idx

    def train_and_split(self, node):
        """
        Train the classifier head and split the node 
        """
        print(f">>>>>>>> train node {node.id} >>>>>>>>>")
        assert node.n >= 2
        x_ori, x, y_class, y_cluster, y_reg, tau = self.get_traindata_weight(node.dataset)

        # print("tau:", tau)

        train_dataset = pd.DataFrame({
            "SMILES": x_ori,
            "features": x,
            "targets": y_reg,
            "clusters": y_cluster,
            "labels": y_class,
        })
        node.update_dataset(train_dataset)
        node.tau = tau
        is_splitable, p_samples, n_samples = self._finetune_head_clf(train_dataset, node)
        node.clf = deepcopy(self.clf_head.state_dict())
        node.is_splitable = is_splitable
        if is_splitable:
            print(">>>>>>> node is splittable >>>>>")
        else:
            print(">>>>>>> node is not splittable >>>>>")
        assert len(p_samples) + len(n_samples) == node.n
        return p_samples, n_samples

    def tree_construction(self):
        # the node will bifurcate into a good and a bad kid
        #### initialize the root node with the training data ###
        # For finetuning, will process the training data with the peft_model and get the features
        print("======== init nodes, if finetuning, extract and store features ==========")
        start_feature, end_feature = trace_times(start=None, end=None, device=self.device)
        self.init_nodes_with_train_data()
        print("time for prepare features:", trace_times(start_feature, end_feature, self.device))
        #########################################
        start_tree, end_tree = trace_times(start=None, end=None, device=self.device)
        print("======== train weighted classifiers for each feasible node ==========")
        assert self.ROOT.n == len(self.training_set)
        assert len(self.nodes) == 1
        self.theta = self.clf_head.state_dict()
        # self.theta = self.theta.to(self.device)
        while self.is_splitable():
            to_split = self.get_split_idx()
            #print("==>to split:", to_split, " total:", len(self.nodes) )
            for nidx in to_split:
                parent = self.nodes[nidx]
                assert parent.n >= self.LEAF_SAMPLE_SIZE
                assert parent.is_splitable == True
                # print("spliting node:", parent.get_name(), len(parent.bag))
                good_kid_data, bad_kid_data = self.train_and_split(parent)

                ##### check device #####
                # update meta-model theta
                if self.finetuning:
                    if parent.parent is None:
                        ## only update theta with the root node model
                        for name in parent.clf:
                            self.theta[name] = (1 - self.eta) * self.theta[name].to(self.device) + self.eta * parent.clf[name].to(self.device)
                else:
                    for name in parent.clf:
                        self.theta[name] = (1 - self.eta) * self.theta[name].to(self.device) + self.eta * parent.clf[name].to(self.device)

                #creat two kids, assign the data, and push into lists
                assert len(good_kid_data) + len(bad_kid_data) == parent.n
                if len(good_kid_data) == 0 or len(bad_kid_data) == 0:
                    print("no split")
                    continue
                good_kid = Node(parent=parent, reset_id=False, tau=None, clf=None, is_splitable=True, threshold=self.threshold)
                # _ , _ = self.train_and_split(good_kid)
                bad_kid = Node(parent=parent, reset_id=False, tau=None, clf=None, is_splitable=True, threshold=self.threshold)
                good_kid.update_dataset(good_kid_data)
                bad_kid.update_dataset(bad_kid_data)

                parent.update_kids(good_kid=good_kid, bad_kid=bad_kid)

                self.nodes.append(good_kid)
                self.nodes.append(bad_kid)

            #print("continue split:", self.is_splitable())
        print("time for training node classfiers:", trace_times(start_tree, end_tree, self.device))
        # self.print_tree()

    def reset_to_root(self):
        self.CURT = self.ROOT

    def greedy_select(self):
        """Select the leaf node with the highest mean value (no exploration bonus)."""
        self.reset_to_root()
        curt_node = self.ROOT
        path = []
        while curt_node.is_leaf() == False:
            scores = []
            for i in curt_node.kids:
                scores.append(i.get_vbar())
            scores = np.nan_to_num(scores, nan=0)
            choice = np.random.choice(np.argwhere(scores == np.amax(scores)).reshape(-1), 1)[0]
            path.append((curt_node, choice))
            print("current node:", curt_node.get_name())
            print("greedy scores =>", scores)
            curt_node = curt_node.kids[choice]
            print("=>", curt_node.get_name(), end=' ')
        print("")
        if len(path) == 0:
            path.append((curt_node, None))
        return curt_node, path

    def select(self):
        """
            select the leaf node and the path to it.
            selection_type="uct": UCT (mean + exploration bonus)
            selection_type="greedy": greedy (mean value only, no exploration)
        """
        if self.selection_type == "greedy":
            return self.greedy_select()

        self.reset_to_root()
        curt_node = self.ROOT
        path = []

        while curt_node.is_leaf() == False:
            UCT = []
            for i in curt_node.kids:
                UCT.append(i.get_uct2(self.lmbda))
                print("node mean and variance")
                print(i.v_bar, i.v_var)
                print(i.dataset["targets"])
            # print(UCT)
            UCT = np.nan_to_num(UCT, nan=0)
            choice = np.random.choice(np.argwhere(UCT == np.amax(UCT)).reshape(-1), 1)[0]
            path.append((curt_node, choice))
            print("current node:", curt_node.get_name())
            print("UCT =>", UCT)
            curt_node = curt_node.kids[choice]
            print("next node:", curt_node.get_name())
        if len(path) == 0:
            path.append((curt_node, None))
        print("finish select")
        return curt_node, path

    # def backpropogate(self, leaf, acc):
    #     curt_node = leaf
    #     while curt_node is not None:
    #         assert curt_node.n > 0
    #         curt_node.x_bar = (curt_node.v_bar * curt_node.n + acc) / (curt_node.n + 1)
    #         curt_node.n += 1
    #         curt_node = curt_node.parent

    def cal_cluster_uct(self, node_sta, train_sta, n, lmbda=0.1):
        """
            If any cluster does not exist in the leaf node, consider its UCT based on root node data.
            If both leaf and root node data do not contain the cluster, assign a large UCT value.
            To make sure this cluster can be observed.
        """
        cluster_uct = []
        max_uct = -1000000
        max_cluster = []
        for cluster in range(5):
            if cluster in node_sta["clusters"].values:
                n_i_leaf = node_sta[node_sta["clusters"] == cluster]["freq"].values[0]
                # print(n_i)
                n_i = train_sta[train_sta["clusters"] == cluster]["freq"].values[0]
                v_i = node_sta[node_sta["clusters"] == cluster]["avg"].values[0]
                # print(v_i, n_i)
                uct_i = v_i + 2 * lmbda * math.sqrt(2 * np.log(n) / n_i_leaf) + n_i_leaf / float(n_i)
            elif cluster in train_sta["clusters"].values:
                v_i = train_sta[train_sta["clusters"] == cluster]["avg"].values[0]
                n_i = train_sta[train_sta["clusters"] == cluster]["freq"].values[0]
                uct_i = v_i + 2 * lmbda * math.sqrt(2 * np.log(n) / n_i)
            else:
                uct_i = float(100000)
            if max_uct < uct_i:
                max_uct = uct_i
            cluster_uct.append(uct_i)
        max_cluster = random.choice([index for index, value in enumerate(cluster_uct) if value == max_uct])
        return cluster_uct, max_cluster

    def get_cluster(self, node):
        """
            Get the cluster with the highest UCT value.
        """
        print(node.dataset)

        node_sta = node.dataset.groupby('clusters').agg(
            freq=('clusters', 'size'),
            avg=('targets', 'mean'),
        ).reset_index()
        train_sta = self.ROOT.dataset.groupby('clusters').agg(
            freq=('clusters', 'size'),
            avg=('targets', 'mean'),
        ).reset_index()
        cluster_uct, max_cluster = self.cal_cluster_uct(node_sta, train_sta, node.n)
        print(node_sta)
        print(train_sta)
        print("======== cluster uct =======")
        print(cluster_uct)
        # max_idx = node.dataset["targets"].idxmax()
        #k = min(int(node.n * 0.5), 3)
        #max_inds = node.dataset["targets"].nlargest(k).index.tolist()
        #print("======== node maximals ========")
        #for idx in max_inds:
        #    print(node.dataset.iloc[idx])
        #max_idx = random.choice(max_inds)
        #cluster = node.dataset.iloc[max_idx]["clusters"]
        # cluster = random.choice([0, 1, 2, 3, 4])
        return max_cluster

    def get_clusters(self, node=None, pval=0.01, n_clusters=5):
        print(">>>> select clusters >>>>>")
        # print(">>>> leaf node data >>>>>")
        # print(node.dataset)
        print(">>>> all data >>>>")
        print(self.ROOT.dataset[["targets", "clusters"]])

        # node_sta = node.dataset.groupby('clusters').agg(
        #     freq=('clusters', 'size'),
        #     avg=('targets', 'mean'),
        # ).reset_index()
        # train_sta = self.ROOT.dataset.groupby('clusters').agg(
        #     freq=('clusters', 'size'),
        #     avg=('targets', 'mean'),
        # ).reset_index()

        # Welch's ANOVA
        welch_result = pg.welch_anova(dv='targets', between='clusters', data=self.ROOT.dataset)
        print(welch_result)
        posthoc = pg.pairwise_gameshowell(dv='targets', between='clusters', data=self.ROOT.dataset)
        print(posthoc)
        posthoc = posthoc[posthoc["pval"] < pval]
        to_remove = set()
        for index, row in posthoc.iterrows():
            if row["mean(A)"] > row["mean(B)"]:
                to_remove.add(row["B"])
            else:
                to_remove.add(row["A"])
        # to_remove = posthoc["B"].unique()
        clusters = [c for c in range(n_clusters) if c not in to_remove]
        return clusters
