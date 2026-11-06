import torch
import torch.utils.data as data_utils
import pandas as pd
from transformers import (
    DataCollatorWithPadding,
    RobertaTokenizer,
)
from transformers import PreTrainedTokenizer
from datasets import Dataset  # huggingface datasets
from typing import *

from benchmarks.MAT.prompting import PromptBuilder

from datasets.utils.logging import disable_progress_bar

disable_progress_bar()


class DataProcessor:
    """
    Base class for all Bayesian optimization datasets (always regression)
    """

    def __init__(self, prompt_builder: PromptBuilder, num_outputs: int, tokenizer: PreTrainedTokenizer, clustering_type: str):
        self.prompt_builder = prompt_builder
        self.num_outputs = num_outputs
        self.tokenizer = tokenizer
        self.dataset = None
        # To be defined in subclasses

        self.target_col = None
        self.target_col_transformed = None
        self.cluster_col = self.get_cluster_col(clustering_type)
        self.obj_str = None
        self.data_cols = None

    def get_cluster_col(self, clustering_type):
        if clustering_type == "kmeans":
            cluster_col = "cluster"
        elif clustering_type == "llms":
            cluster_col = "llm_cluster"
        else:
            cluster_col = "cluster"
        return cluster_col

    def get_dataloader(self, pandas_dataset: pd.DataFrame, batch_size=16, max_seq_len=512,\
                        shuffle=False, append_eos=True,) -> data_utils.DataLoader:

        dataset = Dataset.from_pandas(pandas_dataset)

        # print(dataset["SMILES"])

        def tokenize(row):
            prompt = self.prompt_builder.get_prompt(row["SMILES"], self.obj_str)
            if append_eos:
                prompt += self.tokenizer.eos_token
            # print(prompt)
            out = self.tokenizer(prompt, truncation=True, max_length=max_seq_len)
            # out["SMILES"] = [row["SMILES"]]
            # out["inputs"] = self._get_inputs(row)
            out["targets"] = self._get_targets(row)
            # initial dataset return target_col_transformed
            # after split to nodes, return targets (regression labels)
            if "clusters" in self.data_cols or self.cluster_col in self.data_cols:
                out["clusters"] = self._get_clusters(row)
            labels = self._get_labels(row)
            # out["weights"] = self._get_weights("weights", row)

            if "Entry Number" in self.data_cols:
                out["Entry Number"] = row["Entry Number"]

            if labels is not None:
                out["labels"] = labels
            else:
                out["labels"] = out["targets"]
            # for laplace, return targets as labels
            # for our algos, return binary labels as labels
            # vim log
            # print(labels)
            for col in self.data_cols:
                # print(col)
                if col not in out and "weights" in col:
                    out[col] = self._get_weights(col, row)
            return out

        self.data_cols = pandas_dataset.columns
        to_remove = list(set(self.data_cols).intersection(set(self._get_columns_to_remove())))
        # print(to_remove)
        # print(self.data_cols)
        dataset = dataset.map(tokenize, remove_columns=to_remove, num_proc=4)
        # print(dataset['targets'])
        # print(dataset)
        self.dataset = dataset
        data_loader = data_utils.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=DataCollatorWithPadding(self.tokenizer),
        )
        return data_loader

    def _get_targets(self, row: Union[pd.Series, dict]) -> torch.Tensor:
        """
        Arguments:
        ----------
        row: pd.Series containing one entry or a dictionary
            A single row of the raw dataset.

        Returns:
        --------
        targets: torch.Tensor
            Regression target(s). Shape (self.num_outputs,).
        """

        if "targets" in self.data_cols:
            return row["targets"]
        elif self.target_col_transformed not in self.data_cols:
            return [row[self.target_col]]
        elif isinstance(self.target_col_transformed, list):
            return [row[col] for col in self.target_col_transformed]
        else:
            return row[self.target_col_transformed]

    # def _get_inputs(self, row: Union[pd.Series, dict]) -> torch.Tensor:
    #     """
    #     Arguments:
    #     ----------
    #     row: pd.Series containing one entry or a dictionary
    #         A single row of the raw dataset.

    #     Returns:
    #     --------
    #     targets: torch.Tensor
    #         Regression target(s). Shape (self.num_outputs,).
    #     """
    #     if isinstance(self.target_col_transformed, list):
    #         return [row["SMILES"] for col in self.target_col_transformed]
    #     else:
    #         return [row["SMILES"]]

    def _get_clusters(self, row: Union[pd.Series, dict]) -> torch.Tensor:
        """
        Arguments:
        ----------
        row: pd.Series containing one entry or a dictionary
            A single row of the raw dataset.

        Returns:
        --------
        clusters: torch.Tensor
            clusters (s). Shape (self.num_outputs,).
        """
        if "clusters" in self.data_cols:
            return row["clusters"]
        elif isinstance(self.cluster_col, list):
            return [row[col] for col in self.cluster_col]
        else:
            return row[self.cluster_col]

    def _get_labels(self, row: Union[pd.Series, dict]) -> torch.Tensor:
        """
        Arguments:
        ----------
        row: pd.Series containing one entry or a dictionary
            A single row of the raw dataset.

        Returns:
        --------
        clusters: torch.Tensor
            clusters (s). Shape (self.num_outputs,).
        """
        if "labels" in self.data_cols:
            print("labels", self.data_cols, row["labels"])
            return row["labels"]
        else:
            return None
        # elif isinstance(self.cluster_col, list):
        #     return [row[col] for col in self.cluster_col]
        # else:
        #     return row[self.cluster_col]

    def _get_weights(self, weight_col, row: Union[pd.Series, dict]) -> torch.Tensor:
        """
        Arguments:
        ----------
        row: pd.Series containing one entry or a dictionary
            A single row of the raw dataset.

        Returns:
        --------
        clusters: torch.Tensor
            clusters (s). Shape (self.num_outputs,).
        """
        # if "weights" in self.data_cols:
        #     # print("labels", self.data_cols, row["labels"])
        #     return row["weights"]
        if weight_col in self.data_cols:
            # print(weight_col)
            return row[weight_col]
        else:
            print(f"Warning: {weight_col} not found in data_cols. Returning None.")
            return None

    def _get_columns_to_remove(self) -> List[str]:
        """
        Returns:
        --------
        cols: list of strs
            Columns to remove from the dataset
        """
        raise NotImplementedError


class RedoxDataProcessor(DataProcessor):
    """
    RangeIndex: 1407 entries, 0 to 1406
    Data columns (total 7 columns):
    #   Column                 Non-Null Count  Dtype
    --  ------                 --------------  -----
    0   Entry Number           1407 non-null   int64
    1   File Name              1407 non-null   object
    2   SMILES                 1407 non-null   object
    3   IUPAC Name             1407 non-null   object
    4   Ered                   1407 non-null   float64
    5   HOMO                   1407 non-null   float64
    6   Gsol                   1407 non-null   float64
    7   Absorption Wavelength  1407 non-null   float64
    dtypes: float64(4), int64(1), object(2)
    memory usage: 77.1+ KB

    Objective: Minimize Ered (secondary objective: minimize Gsol)
    """

    def __init__(self, prompt_builder, tokenizer, iupac=False, clustering_type="kmeans"):
        super().__init__(prompt_builder=prompt_builder, num_outputs=1, tokenizer=tokenizer, clustering_type=clustering_type)
        self.target_col = "Ered"
        self.target_col_transformed = self.target_col + "_transformed"
        self.obj_str = "redox potential"
        self.iupac = iupac

    def _get_columns_to_remove(self) -> List[str]:
        if self.iupac:
            columns = ["Entry Number", "File Name", "SMILES",\
                       "IUPAC Name", "HOMO", "Ered", "Gsol",\
                       "Absorption Wavelength", self.cluster_col, self.target_col_transformed]
        else:
            columns = ["Entry Number", "IUPAC Name", "File Name", "SMILES",\
                       "HOMO", "Ered", "Gsol",\
                       "Absorption Wavelength", self.cluster_col, self.target_col_transformed]
        return columns


class SolvationDataProcessor(DataProcessor):
    """
    RangeIndex: 1407 entries, 0 to 1406
    Data columns (total 7 columns):
    #   Column                 Non-Null Count  Dtype
    --  ------                 --------------  -----
    0   Entry Number           1407 non-null   int64
    1   File Name              1407 non-null   object
    2   SMILES                 1407 non-null   object
    3   IUPAC Name             1407 non-null   object
    4   Ered                   1407 non-null   float64
    5   HOMO                   1407 non-null   float64
    6   Gsol                   1407 non-null   float64
    7   Absorption Wavelength  1407 non-null   float64
    dtypes: float64(4), int64(1), object(2)
    memory usage: 77.1+ KB

    Objective: Minimize Ered (secondary objective: minimize Gsol)
    """

    def __init__(self, prompt_builder, tokenizer, iupac=False, clustering_type="kmeans"):
        super().__init__(prompt_builder=prompt_builder, num_outputs=1, tokenizer=tokenizer, clustering_type=clustering_type)
        self.target_col = "Gsol"
        self.target_col_transformed = self.target_col + "_transformed"
        self.obj_str = "solvation energy"
        self.iupac = iupac

    def _get_columns_to_remove(self) -> List[str]:
        if self.iupac:
            columns = ["Entry Number", "File Name", "SMILES",\
                       "IUPAC Name", "HOMO", "Ered", "Gsol",\
                       "Absorption Wavelength",self.cluster_col, self.target_col_transformed]
        else:
            columns = ["Entry Number", "File Name", "SMILES",\
                       "IUPAC Name", "HOMO", "Ered", "Gsol",\
                       "Absorption Wavelength",self.cluster_col, self.target_col_transformed]
        return columns


class KinaseDockingDataProcessor(DataProcessor):
    """
    Three datasets (10k, 50k, HTS) with same structure.

    RangeIndex:
        10k: 10,449 entries, 0 to 10448
        50k: 49,706 entries, 0 to 49,705
        HTS: 2,104,318 entries, 0 to 2,104,317

    Data columns (total 2 columns):
    #   Column                 Dtype
    --  ------                 -----
    0   SMILES                 object
    1   score                  float64

    dtypes: float64(1), object(1)

    Objective: Minimize the score
    """

    def __init__(self, prompt_builder, tokenizer, clustering_type="kmeans"):
        super().__init__(prompt_builder=prompt_builder, num_outputs=1, tokenizer=tokenizer, clustering_type=clustering_type)
        self.target_col = "score"
        self.target_col_transformed = self.target_col + "_transformed"
        self.obj_str = "docking score"

    # def _get_targets(self, row: Union[pd.Series, dict]) -> List[float]:
    #     return [row["score"]]

    def _get_columns_to_remove(self) -> List[str]:
        return ["SMILES", "score", self.cluster_col, self.target_col_transformed]


class AmpCDockingDataProcessor(DataProcessor):
    """
    RangeIndex: 96,214,206 entries, 0 to 96,214,205

    Data columns (total 2 columns):
    #   Column                 Dtype
    --  ------                 -----
    0   SMILES                 object
    1   score                  float64

    dtypes: float64(1), object(1)

    Objective: Minimize the score
    """

    def __init__(self, prompt_builder, tokenizer, clustering_type="kmeans"):
        super().__init__(prompt_builder=prompt_builder, num_outputs=1, tokenizer=tokenizer, clustering_type=clustering_type)
        self.target_col = "score"
        self.target_col_transformed = self.target_col + "_transformed"
        self.obj_str = "docking score"

    def _get_columns_to_remove(self) -> List[str]:
        return ["SMILES", "score", self.cluster_col, self.target_col_transformed]


class D4DockingDataProcessor(DataProcessor):
    """
    RangeIndex: 116,241,184 entries, 0 to 116,241,183

    Data columns (total 2 columns):
    #   Column                 Dtype
    --  ------                 -----
    0   SMILES                 object
    1   score                  float64

    dtypes: float64(1), object(1)

    Objective: Minimize the score
    """

    def __init__(self, prompt_builder, tokenizer, clustering_type="kmeans"):
        super().__init__(prompt_builder=prompt_builder, num_outputs=1, tokenizer=tokenizer, clustering_type=clustering_type)
        self.target_col = "score"
        self.target_col_transformed = self.target_col + "_transformed"
        self.obj_str = "docking score"

    def _get_columns_to_remove(self) -> List[str]:
        return ["SMILES", "score", self.cluster_col, self.target_col_transformed]


class PhotovoltaicsPCEDataProcessor(DataProcessor):
    """
    RangeIndex: 2,320,648 entries, 0 to 2,232,647

    Data columns (total 2 columns):
    #   Column                 Dtype
    --  ------                 -----
    0   SMILES                 object
    1   pce                    float64

    dtypes: float64(1), object(1)

    Objective: Maximize the PCE
    """

    def __init__(self, prompt_builder, tokenizer, clustering_type="kmeans"):
        super().__init__(prompt_builder=prompt_builder, num_outputs=1, tokenizer=tokenizer, clustering_type=clustering_type)
        self.target_col = "pce"
        self.target_col_transformed = self.target_col + "_transformed"
        self.obj_str = "power conversion efficiency"

    def _get_columns_to_remove(self) -> List[str]:
        return ["SMILES", "pce", self.cluster_col, self.target_col_transformed]


class LaserEmitterDataProcessor(DataProcessor):
    """
    RangeIndex: 182,858 entries, 0 to 182,857

    Data columns (total 2 columns):
    #   Column                              Dtype
    --  ------                              -----
    0   SMILES                              object
    1   Fluorescence Oscillator Strength    float64
    2   Electronic Gap                      float64

    dtypes: float64(2), object(1)

    Objective: Maximize the Fluorescence Oscillator Strength (secondary objective: maximize the Electronic Gap)
    """

    def __init__(self, prompt_builder, tokenizer, clustering_type="kmeans"):
        super().__init__(prompt_builder=prompt_builder, num_outputs=1, tokenizer=tokenizer, clustering_type=clustering_type)
        self.target_col = "Fluorescence Oscillator Strength"
        self.target_col_transformed = self.target_col + "_transformed"
        self.obj_str = "fluorescence oscillator strength"

    def _get_columns_to_remove(self) -> List[str]:
        return ["SMILES", "Fluorescence Oscillator Strength", self.cluster_col, self.target_col_transformed]  #, "Electronic Gap"]


class PhotoswitchDataProcessor(DataProcessor):
    """
    RangeIndex: 392 entries, 0 to 391

    Data columns (total 2 columns):
    #   Column                              Dtype
    --  ------                              -----
    0   SMILES                              object
    1   Pi-Pi* Transition Wavelength        float64

    dtypes: float64(1), object(1)

    Objective: Maximize the Pi–Pi* Transition Wavelength of the E isomer
    """

    def __init__(self, prompt_builder, tokenizer, clustering_type="kmeans"):
        super().__init__(prompt_builder=prompt_builder, num_outputs=1, tokenizer=tokenizer, clustering_type=clustering_type)
        self.target_col = "Pi-Pi* Transition Wavelength"
        self.target_col_transformed = self.target_col + "_transformed"
        self.obj_str = "Pi-Pi* Transition Wavelength"

    def _get_columns_to_remove(self) -> List[str]:
        return ["SMILES", "Pi-Pi* Transition Wavelength", self.cluster_col, self.target_col_transformed]


class MultiRedoxDataProcessor(DataProcessor):
    """
    RangeIndex: 1407 entries, 0 to 1406
    Data columns (total 7 columns):
    #   Column                 Non-Null Count  Dtype
    --  ------                 --------------  -----
    0   Entry Number           1407 non-null   int64
    1   File Name              1407 non-null   object
    2   SMILES                 1407 non-null   object
    3   IUPAC Name             1407 non-null   object
    4   Ered                   1407 non-null   float64
    5   HOMO                   1407 non-null   float64
    6   Gsol                   1407 non-null   float64
    7   Absorption Wavelength  1407 non-null   float64
    dtypes: float64(4), int64(1), object(2)
    memory usage: 77.1+ KB

    Objective: Minimize Ered, minimize Gsol
    """

    def __init__(self, prompt_builder, tokenizer, clustering_type="kmeans"):
        super().__init__(prompt_builder=prompt_builder, num_outputs=2, tokenizer=tokenizer, clustering_type=clustering_type)
        self.target_col = ["Ered", "Gsol"]
        self.target_col_transformed = [target_col + "_transformed" for target_col in self.target_col]
        self.obj_str = "redox potential and solvation energy"

    def _get_columns_to_remove(self) -> List[str]:
        columns = [
            "Entry Number",
            "File Name",
            "SMILES",
            "HOMO",
            "Ered",
            "Gsol",
            "Absorption Wavelength",
        ] + self.target_col_transformed
        return columns


class MultiLaserDataProcessor(DataProcessor):
    """
    Data columns (total 2 columns):
    #   Column                              Dtype
    --  ------                              -----
    0   SMILES                              object
    1   Fluorescence Oscillator Strength    float64
    2   Electronic Gap                      float64

    dtypes: float64(2), object(1)

    Objective: Maximize the Fluorescence Oscillator Strength (secondary objective: maximize the Electronic Gap)
    """

    def __init__(self, prompt_builder, tokenizer, clustering_type="kmeans"):
        super().__init__(prompt_builder=prompt_builder, num_outputs=2, tokenizer=tokenizer, clustering_type=clustering_type)
        self.target_col = ["Fluorescence Oscillator Strength", "Electronic Gap"]
        self.target_col_transformed = [target_col + "_transformed" for target_col in self.target_col]
        self.obj_str = "fluorescence oscillator strength and electronic gap"

    def _get_columns_to_remove(self) -> List[str]:
        return ["SMILES", "Fluorescence Oscillator Strength", "Electronic Gap", self.cluster_col] + self.target_col_transformed


if __name__ == "__main__":
    tok = RobertaTokenizer.from_pretrained("roberta-base")
    df = pd.read_csv("data/redox_mer.csv")

    dset = RedoxDataProcessor(None, tokenizer=tok)
    dataloader = dset.get_dataloader(df)

    for data in dataloader:
        # print(data.keys()); input()
        print(data.input_ids.shape, data.attention_mask.shape, data.targets.shape)
        input()
