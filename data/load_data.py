import numpy as np
import pickle
import os
class Dataset:
    def __init__(self, data_dir: str, dataset_name: str, embedding_model: str):
        self.data_path = f"{data_dir}/{dataset_name}"
        self.embedding_model = embedding_model
        self.dataset_name = dataset_name
        self.embeddings = None
        
    
    def load_embeddings(self, embedding_server=None, recompute=False, num_samples=None):
        if recompute is False and os.path.exists(f"{self.data_path}/{self.embedding_model}_embeddings.npy"):
            embeddings = np.load(f"{self.data_path}/{self.embedding_model}_embeddings.npy")
            return embeddings
        elif recompute is True and embedding_server is not None:
            texts = self.load_texts()
            if num_samples is not None:
                texts = texts[:num_samples]
                embeddings = embedding_server.embed_text(texts)
                with open(f"{self.data_path}/{self.embedding_model}_embeddings_{num_samples}.npy", "wb") as f:
                    np.save(f, embeddings)
            else:
                embeddings = embedding_server.embed_text(texts)
                with open(f"{self.data_path}/{self.embedding_model}_embeddings.npy", "wb") as f:
                    np.save(f, embeddings)
            print(f"Embeddings saved to {self.data_path}/{self.embedding_model}_embeddings.npy")
            return embeddings
        else:
            raise ValueError("Embeddings not found and embedding server is not provided")
            
    
    def load_texts(self):
        with open(f"{self.data_path}/texts.pkl", "rb") as f:
            texts = pickle.load(f)
        return texts    
    
    def load_metadata(self):
        with open(f"{self.data_path}/metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        return metadata