#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : 2025
# File Name: prompt_cluster_api.py
# Description: API ablation for LLM clustering - supports Gemini and Qwen 3 APIs
"""
import os
import json
import numpy as np
import time

from openai import OpenAI
from benchmarks.MAT.prompt_cluster import PromptCluster


class PromptClusterAPI(PromptCluster):
    """
    Extends PromptCluster to support multiple LLM APIs for clustering ablation.
    Supports: GPT (OpenAI), Gemini (Google), Qwen 3 (DashScope).
    """

    def __init__(self, data_name, molecule_col, api_type="gpt"):
        super().__init__(data_name, molecule_col)
        self.api_type = api_type

    def set_gemini_api_key(self, api_key=None):
        if api_key is None:
            api_key = os.getenv('GOOGLE_API_KEY', '')
        os.environ["GOOGLE_API_KEY"] = api_key

    def set_qwen_api_key(self, api_key=None):
        if api_key is None:
            api_key = os.getenv('DASHSCOPE_API_KEY', '')
        os.environ["DASHSCOPE_API_KEY"] = api_key

    def query_gemini(self, prompt, model="gemini-2.5-pro", temperature=0.7, max_tokens=4000):
        """
        Sends a prompt to Google Gemini API and returns the response.
        """
        try:
            from google import genai
            # api_key = os.getenv('GOOGLE_API_KEY')
            # genai.configure(api_key=api_key)
            # gen_model = genai.GenerativeModel(model)
            # response = gen_model.generate_content(prompt, generation_config=genai.types.GenerationConfig(
            #     temperature=temperature,
            #     max_output_tokens=max_tokens,
            # ))
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )

            return response.text.strip()
        except Exception as e:
            print(f"Gemini API Error: {e}")
            return None

    def query_qwen(self, prompt, model="qwen3.5-plus-2026-02-15", temperature=0.7, max_tokens=4000):
        """
        Sends a prompt to Qwen 3 via DashScope OpenAI-compatible API.
        Uses the official OpenAI client with DashScope base_url.
        """
        try:
            client = OpenAI(
                api_key=os.getenv('DASHSCOPE_API_KEY'),
                base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            )
            completion = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            reply = completion.choices[0].message.content
            return reply.strip()
        except Exception as e:
            print(f"Qwen API Error: {e}")
            return None

    def query_api(self, prompt, max_tokens=4000):
        """Routes to the correct API based on api_type."""
        if self.api_type == "gemini":
            return self.query_gemini(prompt, max_tokens=max_tokens)
        elif self.api_type == "qwen3":
            return self.query_qwen(prompt, max_tokens=max_tokens)
        else:
            return self.query_gpt(prompt, max_tokens=max_tokens)

    def api_clustering(self, dataset, prompt_id=0):
        """
        Perform clustering using the specified API.
        Same logic as gpt_clustering but routes to different APIs
        and saves results in API-specific folders.
        """
        print(dataset)
        # Set API keys based on api_type
        if self.api_type == "gemini":
            self.set_gemini_api_key()
        elif self.api_type == "qwen3":
            self.set_qwen_api_key()
        else:
            api_key = os.getenv('OPENAI_API_KEY', '')
            self.set_api_key(api_key=api_key)

        prefix = self.get_prefix_prompt()
        print(self.molecule_col)
        molecules = dataset[self.molecule_col].to_list()
        batch_size = 200
        n_batches = int(np.ceil(len(molecules) / float(batch_size)))

        folder_path = f"benchmarks/MAT/data/llm_clustering_api/{self.api_type}/{self.data_name}/"

        for i in range(n_batches):
            if i == n_batches - 1:
                batch_mole = molecules[i * batch_size:]
            else:
                batch_mole = molecules[i * batch_size:(i + 1) * batch_size]
            formatted_data = "\n".join([f"{j+1}. {item}" for j, item in enumerate(batch_mole)])
            prompt = prefix + f"**Respond strictly with the counter and numerical cluster labels only. Do not include any additional text.**\nNow please cluster the following molecules:\n{formatted_data}\n"
            print(prompt)
            delay = 10.0
            result_file = folder_path + self.data_name + "_" + str(i) + "_result.npy"
            if os.path.exists(result_file):
                continue
            else:
                try:
                    time.sleep(delay)
                    response = self.query_api(prompt, max_tokens=4000)
                except Exception as e:
                    print(f"Rate limit or error hit: {e}. Waiting for {delay} seconds...")
                    time.sleep(delay)
                    response = self.query_api(prompt, max_tokens=4000)
            print(response)
            if not response:
                raise RuntimeError("API returned empty/invalid response")

            labels = response.strip().split("\n")
            results = []
            for j in range(len(batch_mole)):
                print(batch_mole[j], labels[j])
                tmp = labels[j].split()
                if len(tmp) == 1:
                    results.append((batch_mole[j], tmp[0]))
                else:
                    results.append((batch_mole[j], tmp[1]))

            if not os.path.exists(folder_path):
                print(f"folder '{folder_path}' not exist, making it ...")
                os.makedirs(folder_path)

            with open(folder_path + self.data_name + "_" + str(i) + '_data.json', 'w') as f:
                json.dump(results, f)
            outfile = folder_path + self.data_name + "_" + str(i) + "_result.npy"
            with open(outfile, 'wb') as outfile:
                np.save(outfile, results)

        # Aggregate cluster labels
        cluster_labels = []
        mole_cluster_dict = {}
        for i in range(n_batches):
            results = np.load(folder_path + self.data_name + "_" + str(i) + "_result.npy")
            if results[0][1] == "Cluster":
                print("batch", i, "has wrong results")
                batch_mole = molecules[i * batch_size:(i + 1) * batch_size]
                formatted_data = "\n".join([f"{j+1}. {item}" for j, item in enumerate(batch_mole)])
                prompt = prefix + f"**Respond strictly with the counter and numerical cluster labels only. Do not include any additional text.**\nNow please cluster the following molecules:\n{formatted_data}\n"
                print(prompt)
                response = self.query_api(prompt, max_tokens=4000)
                print(response)
                labels = response.split("\n")
                results = []
                for j in range(len(batch_mole)):
                    print(batch_mole[j], labels[j])
                    tmp = labels[j].split()
                    if len(tmp) == 1:
                        results.append((batch_mole[j], tmp[0]))
                    elif len(tmp) == 2:
                        results.append((batch_mole[j], tmp[1]))
                    elif len(tmp) == 3:
                        results.append((batch_mole[j], tmp[2]))
                    else:
                        results.append((batch_mole[j], tmp[-1]))
                with open(folder_path + self.data_name + "_" + str(i) + '_data.json', 'w') as f:
                    json.dump(results, f)
                outfile = folder_path + self.data_name + "_" + str(i) + "_result.npy"
                with open(outfile, 'wb') as outfile:
                    np.save(outfile, results)

            for mole, label in results:
                if "-" in label:
                    label = label.split("-")[1]
                    print("===============")
                    print(i)
                    print(label)
                elif int(label) < 0 or int(label) > 5:
                    print("===============")
                    print(i)
                    print(label)
                    label = np.random.choice(0, 5)
                cluster_labels.append(int(label))
                mole_cluster_dict[mole] = int(label)

        for mole in dataset[self.molecule_col].to_list():
            assert mole in mole_cluster_dict
            dataset.loc[dataset[self.molecule_col] == mole, "llm_cluster"] = mole_cluster_dict[mole]
        assert len(dataset) == len(cluster_labels)
        return dataset


if __name__ == "__main__":
    pass
