import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import namedtuple


# Define a namedtuple for the return values
DataLoadResult = namedtuple('DataLoadResult', [
    'train_ratings', 'val_ratings', 'test_ratings', 'num_users',
    'num_movies', 'num_genres', 'unique_movie_ids',
    'train_user_item_set', 'test_user_item_map', 'movie_genre_indices',
    'num_occupations', 'num_zipcodes', 'num_genders', 'users'
])

# --- 1. Data Loading and Preprocessing ---
def load_data(data_dir="ml-1m-sample"):
    try:
        # Load datasets
        ratings = pd.read_csv(f'{data_dir}/ratings.dat', sep=':', engine='python', names=['userId', 'movieId', 'rating', 'timestamp'])
        users = pd.read_csv(f'{data_dir}/users.dat', sep=':', engine='python', names=['userId', 'gender', 'age', 'occupation', 'zipcode'])
        movies = pd.read_csv(f'{data_dir}/movies.dat', sep=':', engine='python', names=['movieId', 'title', 'genres'], encoding='latin-1')
    except FileNotFoundError:
        print(f"Error: Data files not found in directory '{data_dir}'.")
        return None

    # Encode user and movie IDs
    user_encoder = LabelEncoder()
    ratings['user_id_encoded'] = user_encoder.fit_transform(ratings['userId'])
    movie_encoder = LabelEncoder()
    movies['movie_id_encoded'] = movie_encoder.fit_transform(movies['movieId'])
    ratings['movie_id_encoded'] = movie_encoder.transform(ratings['movieId'])

    # Encode additional user features
    users['gender_encoded'] = users['gender'].map({'M': 0, 'F': 1})
    users['zipcode_encoded'] = users['zipcode'].astype('category').cat.codes
    occupation_encoder = LabelEncoder()
    users['occupation_encoded'] = occupation_encoder.fit_transform(users['occupation'])

    # Filter users with at least 2 ratings
    user_counts = ratings['userId'].value_counts()
    valid_users = user_counts[user_counts >= 2].index
    ratings = ratings[ratings['userId'].isin(valid_users)]

    # Split data into train, validation, and test sets
    train_validation_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)
    train_ratings, validation_ratings = train_test_split(train_validation_ratings, test_size=0.1, random_state=42)

    # Create user-item interaction sets
    train_user_item_set = set(zip(train_ratings['user_id_encoded'], train_ratings['movie_id_encoded']))
    test_user_item_map = test_ratings.groupby('user_id_encoded')['movie_id_encoded'].apply(set).to_dict()

    # Extract and map genres to indices for movie-genre relationships
    genres_list = list(set(g for genre_list in movies['genres'].str.split('|') for g in genre_list))
    genre_map = {genre: i for i, genre in enumerate(genres_list)}
    movie_genre_indices = {
        row['movie_id_encoded']: [genre_map[g] for g in row['genres'].split('|')]
        for _, row in movies.iterrows()
    }

    return DataLoadResult(
        train_ratings=train_ratings,
        val_ratings=validation_ratings,
        test_ratings=test_ratings,
        num_users=len(user_encoder.classes_),
        num_movies=len(movie_encoder.classes_),
        num_genres=len(genre_map),
        unique_movie_ids=list(ratings['movie_id_encoded'].unique()),
        train_user_item_set=train_user_item_set,
        test_user_item_map=test_user_item_map,
        movie_genre_indices=movie_genre_indices,
        num_occupations=len(occupation_encoder.classes_),
        num_zipcodes=len(users['zipcode_encoded'].unique()),
        num_genders=2,
        users=users
    )

# --- 2. Dataset Class ---
class MovieLensDataset(Dataset):
    def __init__(self, ratings_df, users_df, all_movie_ids, train_user_item_set, movie_genre_indices, is_training=True):
        self.users = torch.tensor(ratings_df['user_id_encoded'].values, dtype=torch.long)
        self.genders = torch.tensor(users_df.set_index('userId').loc[ratings_df['userId']]['gender_encoded'].values, dtype=torch.long)
        self.ages = torch.tensor(users_df.set_index('userId').loc[ratings_df['userId']]['age'].values, dtype=torch.float)
        self.occupations = torch.tensor(users_df.set_index('userId').loc[ratings_df['userId']]['occupation_encoded'].values, dtype=torch.long)
        self.zipcodes = torch.tensor(users_df.set_index('userId').loc[ratings_df['userId']]['zipcode_encoded'].values, dtype=torch.long)
        self.pos_items = torch.tensor(ratings_df['movie_id_encoded'].values, dtype=torch.long)
        self.all_movie_ids = all_movie_ids
        self.num_all_movies = len(self.all_movie_ids)
        self.train_user_item_set = train_user_item_set
        self.movie_genre_indices = movie_genre_indices
        self.is_training = is_training

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        gender = self.genders[idx]
        age = self.ages[idx]
        occupation = self.occupations[idx]
        zipcode = self.zipcodes[idx]
        pos_item = self.pos_items[idx]

        if self.is_training:
            neg_item = pos_item
            while True:
                neg_item_candidate = self.all_movie_ids[np.random.randint(0, self.num_all_movies)]
                if (user.item(), neg_item_candidate) not in self.train_user_item_set:
                    neg_item = torch.tensor(neg_item_candidate, dtype=torch.long)
                    break
        else:
            neg_item = pos_item

        return user, gender, age, occupation, zipcode, pos_item, neg_item

# --- 3. Model Definition ---
class UserTower(nn.Module):
    def __init__(self, num_users, num_genders, num_occupations, num_zipcodes, embedding_dim):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.gender_embedding = nn.Embedding(num_genders, embedding_dim)
        self.occupation_embedding = nn.Embedding(num_occupations, embedding_dim)
        self.zipcode_embedding = nn.Embedding(num_zipcodes, embedding_dim)
        self.age_linear = nn.Linear(1, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 5, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, user_ids, genders, ages, occupations, zipcodes):
        user_emb = self.user_embedding(user_ids)
        gender_emb = self.gender_embedding(genders)
        occupation_emb = self.occupation_embedding(occupations)
        zipcode_emb = self.zipcode_embedding(zipcodes)
        age_emb = self.age_linear(ages.unsqueeze(1))
        combined = torch.cat([user_emb, gender_emb, occupation_emb, zipcode_emb, age_emb], dim=1)
        return self.mlp(combined)

class ItemTower(nn.Module):
    def __init__(self, num_movies, embedding_dim):
        super().__init__()
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, movie_ids):
        movie_emb = self.movie_embedding(movie_ids)
        return self.mlp(movie_emb)

class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_movies, num_genders, num_occupations, num_zipcodes, embedding_dim):
        super().__init__()
        self.user_tower = UserTower(num_users, num_genders, num_occupations, num_zipcodes, embedding_dim)
        self.item_tower = ItemTower(num_movies, embedding_dim)

    def forward(self, user_ids, genders, ages, occupations, zipcodes, movie_ids):
        user_embedding = self.user_tower(user_ids, genders, ages, occupations, zipcodes)
        item_embedding = self.item_tower(movie_ids)
        return user_embedding, item_embedding

# --- 4. Training Loop ---
def train_loop(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        user, gender, age, occupation, zipcode, pos_item, neg_item = [b.to(device) for b in batch]
        optimizer.zero_grad()

        user_emb, pos_item_emb = model(user, gender, age, occupation, zipcode, pos_item)
        _, neg_item_emb = model(user, gender, age, occupation, zipcode, neg_item)

        pos_logits = torch.sum(user_emb * pos_item_emb, dim=1)
        neg_logits = torch.sum(user_emb * neg_item_emb, dim=1)

        pos_labels = torch.ones_like(pos_logits)
        neg_labels = torch.zeros_like(neg_logits)
        logits = torch.cat([pos_logits, neg_logits])
        labels = torch.cat([pos_labels, neg_labels])

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# --- Evaluation Function ---
def evaluate(model, test_user_item_map, unique_movie_ids, item_embeddings, top_n, device):
    """Evaluate the model using HR@N and NDCG@N."""
    model.eval()
    hr, ndcg = 0, 0
    with torch.no_grad():
        for user, pos_items in test_user_item_map.items():
            # Compute user embedding
            user_tensor = torch.tensor([user], dtype=torch.long).to(device)
            user_embedding = model.user_tower(user_tensor, None, None, None, None)

            # Compute scores for all items
            scores = torch.matmul(item_embeddings, user_embedding.T).squeeze(1)

            # Rank items
            top_indices = torch.topk(scores, top_n).indices.cpu().numpy()
            recommended_items = [unique_movie_ids[i] for i in top_indices]

            # Evaluate metrics
            for pos_item in pos_items:
                if pos_item in recommended_items:
                    hr += 1
                    rank = recommended_items.index(pos_item) + 1
                    ndcg += 1 / np.log2(rank + 1)
                    break

    num_users = len(test_user_item_map)
    return hr / num_users, ndcg / num_users

# --- 5. Main Execution ---
if __name__ == "__main__":
    data_dir = "ml-1m-sample"
    data_result = load_data(data_dir)

    if data_result is None:
        exit()

    train_dataset = MovieLensDataset(
        data_result.train_ratings, data_result.users, data_result.unique_movie_ids,
        data_result.train_user_item_set, data_result.movie_genre_indices, is_training=True
    )
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoTowerModel(
        data_result.num_users, data_result.num_movies, data_result.num_genders,
        data_result.num_occupations, data_result.num_zipcodes, embedding_dim=32
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(5):
        train_loss = train_loop(model, train_loader, optimizer, criterion, device)
        print(f"Epoch [{epoch + 1}/5], Train Loss: {train_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "trained_two_tower_model.pth")
    print("Model saved as 'trained_two_tower_model.pth'.")
