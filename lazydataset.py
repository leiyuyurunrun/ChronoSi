import torch
from torch.utils.data import Dataset
import pandas as pd
import transformers 
import numpy as np





class ChronoSightDataset(Dataset):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, per_usage=3):
        """
        Args:
            csv_file (str): Path to the CSV file containing the vectors and classes.
            tokenizer: Tokenizer to convert text to token IDs.
            max_vectors (int): Maximum number of <vector><class> pairs to include in each example.
        """
        self.vectors = np.loadtxt('data/combined_vectors.csv', delimiter=',', dtype=float)
        self.labels = np.loadtxt('data/combined_labels.csv', delimiter=',', dtype=str)
        self.tokenizer = tokenizer
        self.per_usage = per_usage
        self.special_tokens = {
            "vector_start": "<v_start>",
            "vector_end": "<v_end>",
            "vector_patch": "<v_patch>"
            # "class_start": "<class_start>",
            # "class_end": "<class_end>"
        }

        # Ensure new tokens are added to tokenizer
        self.tokenizer.add_tokens(list(self.special_tokens.values()))

    
    
    def _build_prompt(self,idx,classes):
        # Prepare human prompt with known pairs, excluding the last vector-class pair
        human_prompt = ("Please act as a classification agent to perform a few-shot learning task.\n"
                        "Here are some vector-class pairs for reference:\n")
        for c in  classes[:-1]:  # Exclude the last pair
            human_prompt += (f"{self.special_tokens['vector_start']} {self.special_tokens['vector_patch']} {self.special_tokens['vector_end']} - "
                            f"Class: {c}\n")

        # Add query for the new vector (last vector in the list)
        unique_eles = list(set(classes))
        human_prompt += (f"Given this {len(unique_eles)} class:")
        for i,ele in enumerate(unique_eles):
            human_prompt += (f" {i+1}: {ele}")
            
        human_prompt += (f"\npredict the class for this new vector:\n"
                        f"{self.special_tokens['vector_start']} {self.special_tokens['vector_patch']} {self.special_tokens['vector_end']} - Class:")

        # Prepare GPT response for the final vector-class pair
        response = (f"Based on the given pairs, the predicted class for "
                    f"{self.special_tokens['vector_start']} {self.special_tokens['vector_patch']} {self.special_tokens['vector_end']} is "
                    f"{classes[-1]}.")
        
        return human_prompt, response
    def __len__(self):
         return len(self.labels) // self.per_usage

    def __getitem__(self, idx):# 我现在在怀疑这里的浮点数能否进去，之前GraphGPT的话这里依然是以token的空值占位的，进去模型后才用换掉，换掉的方式是图传播，本来数据集不会只有inputids和labels,还会有别的属性，比如graph进去再换掉

       
        # Limit to max_vectors pairs
        start_idx = idx * self.per_usage
        end_idx = start_idx + self.per_usage

        # 获取对应的 vectors 和 classes 子集
        vectors = self.vectors[start_idx:end_idx]
        classes = self.labels[start_idx:end_idx]
        human_prompt, response =self._build_prompt(idx, classes)
     


        # Tokenize
        human_input_ids = self.tokenizer.encode(human_prompt, return_tensors="pt", truncation=True)
        gpt_output_ids = self.tokenizer.encode(response, return_tensors="pt", truncation=True)

        # Combine inputs and labels
        input_ids = torch.cat([human_input_ids, gpt_output_ids], dim=-1)
        labels = input_ids.clone()

        return {
            "input_ids": input_ids.squeeze(),
            "labels": labels.squeeze(),
            "vector_data": torch.tensor(vectors)
        }
        # 本来这里还要加一个额外的地方的向量，比如graph_data。然后直接放到dict中，data_dict = dict(input_ids=data_dict["input_ids"][0],
                            #  labels=data_dict["labels"][0])
