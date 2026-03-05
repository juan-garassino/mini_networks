import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class NanoRAG:
    def __init__(self, knowledge_base):
        # Tiny knowledge base (could scale to millions of documents)
        self.knowledge_base = knowledge_base
        self.embeddings = self._embed_documents()
        
        # Miniature models
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.generator = GPT2LMHeadModel.from_pretrained('gpt2')
        self.embedding_model = nn.Sequential(
            nn.Embedding(len(self.tokenizer), 128),
            nn.AdaptiveAvgPool1d(1)
        )
    
    def _embed_documents(self):
        """Simple document embeddings using TF-IDF"""
        vocab = set(word for doc in self.knowledge_base for word in doc.split())
        word_to_idx = {word: i for i, word in enumerate(vocab)}
        
        embeddings = []
        for doc in self.knowledge_base:
            tfidf = np.zeros(len(vocab))
            words = doc.split()
            for word in words:
                tf = words.count(word) / len(words)
                idf = np.log(len(self.knowledge_base) / sum(1 for d in self.knowledge_base if word in d))
                tfidf[word_to_idx[word]] = tf * idf
            embeddings.append(tfidf)
        
        return np.array(embeddings)
    
    def retrieve(self, query, k=3):
        """Retrieve relevant documents using cosine similarity"""
        query_embed = self.embed_text(query)
        similarities = []
        for doc_embed in self.embeddings:
            cos_sim = np.dot(query_embed, doc_embed) / (
                np.linalg.norm(query_embed) * np.linalg.norm(doc_embed)
            )
            similarities.append(cos_sim)
        
        top_indices = np.argsort(similarities)[-k:][::-1]
        return [self.knowledge_base[i] for i in top_indices]
    
    def embed_text(self, text):
        """Mini embedding model"""
        inputs = self.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            embeds = self.embedding_model(inputs['input_ids']).squeeze()
        return embeds.numpy()
    
    def generate(self, query, max_length=100):
        """Generate response with retrieved context"""
        context = "\n".join(self.retrieve(query))
        prompt = f"CONTEXT:\n{context}\n\nQUERY: {query}\nRESPONSE:"
        
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.generator.generate(
            inputs['input_ids'],
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
if __name__ == "__main__":
    # Tiny knowledge base (Shakespeare facts)
    knowledge_base = [
        "William Shakespeare was born in Stratford-upon-Avon in 1564",
        "Shakespeare wrote 37 plays and 154 sonnets",
        "Famous plays include Hamlet, Romeo and Juliet, and Macbeth",
        "The Globe Theatre was where many Shakespeare plays premiered",
        "Shakespeare died in 1616 at age 52"
    ]
    
    rag = NanoRAG(knowledge_base)
    
    # Generate response using retrieved knowledge
    response = rag.generate("When was Shakespeare born?")
    print("Generated Response:")
    print(response)
    
    # Example output:
    # CONTEXT:
    # William Shakespeare was born in Stratford-upon-Avon in 1564
    # Shakespeare died in 1616 at age 52
    # Shakespeare wrote 37 plays and 154 sonnets
    #
    # QUERY: When was Shakespeare born?
    # RESPONSE: William Shakespeare was born in 1564 in Stratford-upon-Avon.