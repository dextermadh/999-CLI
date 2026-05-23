import os
import json
import numpy as np

class EpisodeStore:
    def __init__(self, workspace_root: str, model_name: str = "all-MiniLM-L6-v2"):
        self.workspace_root = workspace_root
        self.memory_dir = os.path.join(workspace_root, ".999", "memory")
        os.makedirs(self.memory_dir, exist_ok=True)
        
        self.index_path = os.path.join(self.memory_dir, "episodes.index")
        self.data_path = os.path.join(self.memory_dir, "episodes.json")
        
        self.model = None
        self._has_model = False
        
        # Resilient lazy load embedding model to prevent memory crashes
        try:
            print(f"Loading memory embedding model {model_name}...")
            from sentence_transformers import SentenceTransformer
            import faiss
            self.model = SentenceTransformer(model_name)
            self.faiss = faiss
            self._has_model = True
            print("✓ Memory embedding model loaded successfully.")
        except Exception as e:
            print(f"⚠ Warning: Could not load memory embedding model ({str(e)}). Semantic memory retrieval will be bypassed.")

        # Initialize FAISS index
        self.dimension = 384 # Dimension for all-MiniLM-L6-v2
        if self._has_model and os.path.exists(self.index_path):
            try:
                self.index = self.faiss.read_index(self.index_path)
                with open(self.data_path, 'r') as f:
                    self.episodes = json.load(f)
            except Exception as e:
                print(f"⚠ Warning: Could not load FAISS memory index ({str(e)}). Initializing flat index.")
                if self._has_model:
                    self.index = self.faiss.IndexFlatL2(self.dimension)
                self.episodes = []
        else:
            if self._has_model:
                self.index = self.faiss.IndexFlatL2(self.dimension)
            self.episodes = [] # List of dicts: {"request": str, "solution": str}
            
    def add_episode(self, request: str, solution: str):
        """Adds a successful episode to memory."""
        if not self._has_model:
            return
            
        try:
            embedding = self.model.encode([request])[0]
            
            # Add to FAISS index
            self.index.add(np.array([embedding]).astype('float32'))
            
            # Add to data list
            self.episodes.append({"request": request, "solution": solution})
            
            # Save to disk
            self.faiss.write_index(self.index, self.index_path)
            with open(self.data_path, 'w') as f:
                json.dump(self.episodes, f, indent=2)
                
            print(f"Saved episode to memory. Total episodes: {len(self.episodes)}")
        except Exception as e:
            print(f"⚠ Warning: Could not save episode to memory: {str(e)}")
        
    def search_similar(self, query: str, k: int = 3) -> list:
        """Searches for similar past episodes."""
        if not self._has_model or not self.episodes:
            return []
            
        try:
            if self.index.ntotal == 0:
                return []
                
            query_embedding = self.model.encode([query])[0]
            distances, indices = self.index.search(np.array([query_embedding]).astype('float32'), k)
            
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(self.episodes) and idx >= 0:
                    results.append({
                        "request": self.episodes[idx]["request"],
                        "solution": self.episodes[idx]["solution"],
                        "distance": float(dist)
                    })
            return results
        except Exception as e:
            print(f"⚠ Warning: Memory search failed: {str(e)}")
            return []
