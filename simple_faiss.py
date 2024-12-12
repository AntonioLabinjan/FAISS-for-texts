# Install required libraries
!pip install faiss-cpu sentence-transformers

# Import necessary libraries
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Step 1: Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Use a pre-trained sentence embedding model

# Step 2: Convert the texts to embeddings (vectors)
text_1 = "Danas je tužan dan"
text_2 = "Danas je ružan dan"

# Convert texts to embeddings
embedding_1 = model.encode([text_1])
embedding_2 = model.encode([text_2])

# Step 3: Convert the embeddings to numpy arrays
embedding_1 = np.array(embedding_1).astype('float32')
embedding_2 = np.array(embedding_2).astype('float32')

# Step 4: Create a Faiss index
dim = embedding_1.shape[1]  # The dimensionality of the embeddings
index = faiss.IndexFlatL2(dim)  # L2 distance (Euclidean distance) index

# Step 5: Add the first text's embedding to the Faiss index
index.add(embedding_1)

# Step 6: Search for the second text's embedding in the index
D, I = index.search(embedding_2, 1)  # '1' because we only want the nearest neighbor

# Step 7: Calculate the similarity percentage
similarity_score = 1 / (1 + D[0][0])  # Inverse of distance, lower distance means higher similarity

# Convert similarity score to a percentage
similarity_percentage = similarity_score * 100

# Print the similarity percentage and the distance
print(f"Similarity: {similarity_percentage:.2f}%")
print(f"Distance: {D[0][0]:.4f}")
