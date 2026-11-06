import torch
import torch.utils.data as data_utils
import pandas as pd
from transformers import DataCollatorWithPadding, PreTrainedTokenizer
from typing import List, Union

from benchmarks.MAT.prompting import PromptBuilder


class MultiObjDataProcessor:
    """
    Data processor for multi-objective optimization tasks.
    Handles multiple objectives without using HuggingFace datasets multiprocessing
    to avoid serialization issues with lists.
    """

    def __init__(self, prompt_builder: PromptBuilder, num_outputs: int,
                 tokenizer: PreTrainedTokenizer, obj_cols: List[str], obj_str: str, 
                 clustering_type: str = "kmeans"):
        self.prompt_builder = prompt_builder
        self.num_outputs = num_outputs
        self.tokenizer = tokenizer
        self.obj_cols = obj_cols
        self.obj_str = obj_str
        self.target_col_transformed = [f"{col}_transformed" for col in obj_cols]
        # Add cluster_col attribute needed by LLMAT
        self.cluster_col = self.get_cluster_col(clustering_type)
    
    def get_cluster_col(self, clustering_type):
        """Get cluster column name based on clustering type."""
        if clustering_type == "kmeans":
            cluster_col = "cluster"
        elif clustering_type == "llms":
            cluster_col = "llm_cluster"
        else:
            cluster_col = "cluster"
        return cluster_col

    def get_dataloader(self, pandas_dataset: pd.DataFrame, batch_size=16,
                      max_seq_len=512, shuffle=False, append_eos=True) -> data_utils.DataLoader:
        """
        Create a DataLoader from pandas DataFrame for multi-objective tasks.
        Uses manual processing instead of dataset.map() to avoid multiprocessing issues.
        """

        # Process all rows manually
        processed_data = []

        for idx in range(len(pandas_dataset)):
            row = pandas_dataset.iloc[idx]

            # Get prompt and tokenize
            prompt = self.prompt_builder.get_prompt(row["SMILES"], self.obj_str)
            if append_eos:
                prompt += self.tokenizer.eos_token

            tokenized = self.tokenizer(prompt, truncation=True, max_length=max_seq_len)

            # Get targets (multi-objective)
            if all(col in pandas_dataset.columns for col in self.target_col_transformed):
                # Use transformed targets if available
                targets = [row[col] for col in self.target_col_transformed]
            else:
                # Use original targets
                targets = [row[col] for col in self.obj_cols]

            # Create data dict
            data_dict = {
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask'],
                'labels': targets,  # Multi-objective targets
            }

            processed_data.append(data_dict)

        # Create custom dataset
        dataset = MultiObjDataset(processed_data)

        # Create dataloader with custom collate function
        dataloader = data_utils.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn,
        )

        return dataloader

    def _collate_fn(self, batch):
        """
        Custom collate function for multi-objective data.
        Handles padding and stacking of tensors properly.
        """
        # Pad input_ids and attention_mask
        input_ids = [torch.tensor(item['input_ids']) for item in batch]
        attention_mask = [torch.tensor(item['attention_mask']) for item in batch]
        labels = [torch.tensor(item['labels'], dtype=torch.float32) for item in batch]

        # Pad sequences
        max_len = max(len(ids) for ids in input_ids)

        padded_input_ids = []
        padded_attention_mask = []

        for ids, mask in zip(input_ids, attention_mask):
            padding_length = max_len - len(ids)
            padded_input_ids.append(
                torch.cat([ids, torch.full((padding_length,), self.tokenizer.pad_token_id)])
            )
            padded_attention_mask.append(
                torch.cat([mask, torch.zeros(padding_length, dtype=torch.long)])
            )

        return {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(padded_attention_mask),
            'labels': torch.stack(labels),
        }


class MultiObjDataset(torch.utils.data.Dataset):
    """Simple dataset class for multi-objective data."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MultiRedoxDataProcessor(MultiObjDataProcessor):
    """
    Multi-objective processor for Redox dataset.
    Objectives: Minimize Ered, minimize Gsol
    """

    def __init__(self, prompt_builder: PromptBuilder, tokenizer: PreTrainedTokenizer, 
                 clustering_type: str = "kmeans"):
        super().__init__(
            prompt_builder=prompt_builder,
            num_outputs=2,
            tokenizer=tokenizer,
            obj_cols=["Ered", "Gsol"],
            obj_str="redox potential and solvation energy",
            clustering_type=clustering_type
        )


class MultiLaserDataProcessor(MultiObjDataProcessor):
    """
    Multi-objective processor for Laser dataset.
    Objectives: Maximize Fluorescence Oscillator Strength, maximize Electronic Gap
    """

    def __init__(self, prompt_builder: PromptBuilder, tokenizer: PreTrainedTokenizer,
                 clustering_type: str = "kmeans"):
        super().__init__(
            prompt_builder=prompt_builder,
            num_outputs=2,
            tokenizer=tokenizer,
            obj_cols=["Fluorescence Oscillator Strength", "Electronic Gap"],
            obj_str="fluorescence oscillator strength and electronic gap",
            clustering_type=clustering_type
        )
