import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time # Added for timestamp generation if needed

# --- 1. 데이터 준비 및 로딩 ---
def load_data(data_dir="ml-1m-sample"):
    try:
        ratings = pd.read_csv(f'{data_dir}/ratings.dat', sep=':', engine='python', names=['userId', 'movieId', 'rating', 'timestamp'])
        users = pd.read_csv(f'{data_dir}/users.dat', sep=':', engine='python', names=['userId', 'gender', 'age', 'occupation', 'zipcode'])
        movies = pd.read_csv(f'{data_dir}/movies.dat', sep=':', engine='python', names=['movieId', 'title', 'genres'], encoding='latin-1') # Specify encoding
    except FileNotFoundError:
        print(f"Error: Data files not found in directory '{data_dir}'.")
        print("Please make sure 'users.dat', 'movies.dat', and 'ratings.dat' are in that folder.")
        return None, None, None, None, None # Return None to indicate failure

    # 사용자 ID, 영화 ID 인코딩 (0부터 시작하도록)
    user_encoder = LabelEncoder()
    ratings['user_id_encoded'] = user_encoder.fit_transform(ratings['userId'])
    # Combine movie IDs from both datasets before fitting the encoder
    # Movie encoder for movies dataset
    movie_encoder = LabelEncoder()
    movies['movie_id_encoded'] = movie_encoder.fit_transform(movies['movieId'])

    # Ratings encoder for ratings dataset
    ratings_encoder = LabelEncoder()
    ratings['movie_id_encoded'] = ratings_encoder.fit_transform(ratings['movieId'])

    # Validate the number of unique movies in both datasets
    num_movies_in_movies = len(movie_encoder.classes_)
    num_movies_in_ratings = len(ratings_encoder.classes_)
    num_users = len(user_encoder.classes_)

    print(f"Number of unique movies in movies dataset: {num_movies_in_movies}")
    print(f"Number of unique movies in ratings dataset: {num_movies_in_ratings}")

    # 학습/검증/테스트 데이터 분할 (간단한 랜덤 분할, 실제로는 시간 기반 또는 사용자 기반 권장)
    print(ratings['userId'])
    # Filter out users with fewer than 2 ratings
    user_counts = ratings['userId'].value_counts()
    valid_users = user_counts[user_counts >= 2].index
    ratings = ratings[ratings['userId'].isin(valid_users)]

    train_val_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)
    train_ratings, val_ratings = train_test_split(train_val_ratings, test_size=0.1, random_state=42)
    print(f"Train size: {len(train_ratings)}")
    print(f"Validation size: {len(val_ratings)}")
    print(f"Test size: {len(test_ratings)}")

    # 학습 데이터셋의 사용자-아이템 상호작용 집합 (네거티브 샘플링 시 사용)
    train_user_item_set = set(zip(train_ratings['user_id_encoded'], train_ratings['movie_id_encoded']))

    # 테스트 데이터셋의 사용자별 실제 상호작용 아이템 맵 (평가 시 사용)
    test_user_item_map = test_ratings.groupby('user_id_encoded')['movie_id_encoded'].apply(set).to_dict()


    # --- 2. 피처 엔지니어링 (여기서는 간단히 ID만 사용) ---
    # TODO: users, movies 데이터프레임에서 추가 피처 추출 및 전처리 필요
    # 예: 영화 장르 처리 (이 코드는 예시이며 실제 모델에 맞게 수정 필요)
    movies['movie_id_encoded'] = movie_encoder.transform(movies['movieId']) # ratings와 동일한 인코더 사용 중요
    genres_list = list(set(g for genre_list in movies['genres'].str.split('|') for g in genre_list))
    genre_map = {genre: i for i, genre in enumerate(genres_list)}
    num_genres = len(genre_map)
    print(f"Number of unique genres: {num_genres}")

    movie_genre_indices = {}
    for _, row in movies.iterrows():
      indices = [genre_map[g] for g in row['genres'].split('|')]
      movie_genre_indices[row['movie_id_encoded']] = indices


    return train_ratings, val_ratings, test_ratings, num_users, num_movies_in_movies, num_genres, \
           list(ratings['movie_id_encoded'].unique()), train_user_item_set, test_user_item_map, \
           movie_genre_indices # Pass necessary info

# --- 3. PyTorch Dataset 및 DataLoader 생성 ---
class MovieLensDataset(Dataset):
    def __init__(self, ratings_df, all_movie_ids, train_user_item_set, movie_genre_indices, is_training=True):
        self.users = torch.tensor(ratings_df['user_id_encoded'].values, dtype=torch.long)
        self.pos_items = torch.tensor(ratings_df['movie_id_encoded'].values, dtype=torch.long)
        # self.ratings = torch.tensor(ratings_df['rating'].values, dtype=torch.float) # 평점 사용시
        self.all_movie_ids = all_movie_ids
        self.num_all_movies = len(self.all_movie_ids)
        self.train_user_item_set = train_user_item_set # 학습 데이터의 상호작용 정보
        self.movie_genre_indices = movie_genre_indices
        self.is_training = is_training

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        pos_item = self.pos_items[idx]

        # 학습 시에만 네거티브 샘플링 수행
        if self.is_training:
            neg_item = pos_item # Initialize with positive item
            while True:
                neg_item_candidate_idx = np.random.randint(0, self.num_all_movies)
                neg_item_candidate = self.all_movie_ids[neg_item_candidate_idx]
                if (user.item(), neg_item_candidate) not in self.train_user_item_set:
                    neg_item = torch.tensor(neg_item_candidate, dtype=torch.long)
                    break
        else:
            # 평가 시에는 네거티브 아이템이 필요 없거나 다른 방식으로 처리
            # 여기서는 임시로 포지티브 아이템과 동일하게 설정 (실제 평가 로직에서 사용 안 함)
             neg_item = pos_item

        # TODO: 사용자 및 아이템의 다른 피처들도 가져오기
        # 예: 성별, 나이, 직업 / 장르, 제목 임베딩 등
        # 현재는 장르 인덱스만 가져오는 예시 (실제로는 패딩 등 처리 필요)
        pos_item_genres = self.movie_genre_indices.get(pos_item.item(), [])
        neg_item_genres = self.movie_genre_indices.get(neg_item.item(), []) if self.is_training else []

        # Return user_id, pos_item_id, neg_item_id, and potentially their features
        # Features need careful handling (padding for sequences like genres)
        # Returning raw genre list here - model/collate_fn needs to handle padding/pooling
        return user, pos_item, neg_item #, pos_item_genres, neg_item_genres


# --- 4. 투-타워 모델 정의 ---
class UserTower(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super().__init__()
        # nn.Embedding(num_embeddings, embedding_dimension)
        self.user_embedding = nn.Embedding(num_users, embedding_dim, sparse=False) # sparse=True can save memory for large embeddings
        # TODO: 다른 사용자 피처 임베딩 및 MLP 추가
        # 예시: 간단한 MLP 추가
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1), # Added dropout
            nn.Linear(128, embedding_dim) # 최종 출력 차원
        )

    def forward(self, user_ids):
        user_emb = self.user_embedding(user_ids)
        # TODO: 다른 피처 임베딩과 결합
        output = self.mlp(user_emb) # Apply MLP
        return output

class ItemTower(nn.Module):
    def __init__(self, num_movies, num_genres, embedding_dim):
        super().__init__()
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim, sparse=False)
        # Use EmbeddingBag for variable length genre lists
        # self.genre_embedding_bag = nn.EmbeddingBag(num_genres, embedding_dim, mode='mean', sparse=False)

        if self.movie_embedding.num_embeddings != num_movies:
            raise ValueError("Mismatch between num_movies and embedding layer size.")

        # TODO: 다른 영화 피처(제목 등) 임베딩 및 MLP 추가
        # MLP 예시 (여기서는 영화 ID 임베딩만 사용한다고 가정 - 장르 부분 비활성화)
        self.mlp = nn.Sequential(
             nn.Linear(embedding_dim, 128), # Input dim depends on combined features
             nn.ReLU(),
             nn.Dropout(0.1),
             nn.Linear(128, embedding_dim) # 최종 출력 차원
        )

    # forward 함수는 입력 피처에 따라 수정 필요
    # def forward(self, movie_ids, genre_indices_list=None, offsets=None): # EmbeddingBag 사용시 offsets 필요
    def forward(self, movie_ids):
        movie_emb = self.movie_embedding(movie_ids)

        # --- EmbeddingBag 사용 예시 (주석 처리) ---
        # if genre_indices_list is not None and offsets is not None:
        #     # genre_indices_list: 모든 배치의 장르 인덱스를 하나로 합친 1D 텐서
        #     # offsets: 각 아이템의 시작 인덱스 (0 포함)
        #     genre_emb = self.genre_embedding_bag(genre_indices_list, offsets)
        #     # 영화 ID 임베딩과 장르 임베딩 결합 (예: Concatenate)
        #     combined_emb = torch.cat([movie_emb, genre_emb], dim=1)
        # else: # 장르 정보 없을 경우 영화 ID 임베딩만 사용
        #     # If only using movie_emb, adjust MLP input dimension
        #     # combined_emb = movie_emb
        #     # Need padding if concatenating with genre_emb when some items have no genres
        #     pass
        # -----------------------------------------

        # 현재는 영화 임베딩만 MLP 통과 가정
        output = self.mlp(movie_emb)
        return output

class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_movies, num_genres, embedding_dim):
        super().__init__()
        self.user_tower = UserTower(num_users, embedding_dim)
        self.item_tower = ItemTower(num_movies, num_genres, embedding_dim)

    # forward 함수는 입력 피처에 따라 수정 필요
    def forward(self, user_ids, movie_ids):
        user_embedding = self.user_tower(user_ids)
        item_embedding = self.item_tower(movie_ids)
        return user_embedding, item_embedding

    # 추론 시 사용자 임베딩만 계산하는 함수 (옵션)
    def get_user_embedding(self, user_ids):
         return self.user_tower(user_ids)

    # 추론 시 아이템 임베딩만 계산하는 함수 (옵션)
    def get_item_embedding(self, movie_ids):
         return self.item_tower(movie_ids)


# --- 5. 손실 함수 정의 ---
# BCE Loss 사용 (내적 + Sigmoid와 함께 사용)
criterion = nn.BCEWithLogitsLoss()

# --- 6. 학습 루프 구현 ---
def train_loop(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        user, pos_item, neg_item = [b.to(device) for b in batch]
        # TODO: 배치에서 해당 user/item의 피처들도 가져와서 모델에 전달해야 함
        # 현재 모델은 user_id, movie_id 만 받음

        optimizer.zero_grad()

        # 모델 forward 호출
        user_emb, pos_item_emb = model(user, pos_item)
        # 네거티브 아이템 임베딩 계산 (사용자 임베딩 재사용 안 함 - 모델 구조에 따라 가능)
        # item_tower만 호출하여 효율화 가능: neg_item_emb = model.item_tower(neg_item)
        _, neg_item_emb = model(user, neg_item) # 현재 구조에서는 user도 다시 계산됨

        # 유사도 계산 (내적)
        pos_logits = torch.sum(user_emb * pos_item_emb, dim=1)
        neg_logits = torch.sum(user_emb * neg_item_emb, dim=1)

        # 손실 계산 (Positive=1, Negative=0)
        pos_labels = torch.ones_like(pos_logits)
        neg_labels = torch.zeros_like(neg_logits)
        logits = torch.cat([pos_logits, neg_logits])
        labels = torch.cat([pos_labels, neg_labels])

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# --- 7. 평가 함수 ---
# (이전 답변에서 제공된 evaluate_model 함수 - 필요시 복사하여 사용)
# 주의: evaluate_model 함수는 현재 코드의 모델(ID만 사용)에 맞게 수정 필요
# 특히 모든 아이템 임베딩 계산 및 사용자-아이템 점수 계산 부분
def evaluate_model(model, test_user_item_map, all_movie_ids_list, train_user_item_set, device, k=10):
    model.eval()
    all_users = list(test_user_item_map.keys())
    user_batch_size = 128 # Adjust as needed
    precisions, recalls, ndcgs = [], [], []

    # 모든 아이템 임베딩 미리 계산
    all_item_ids_tensor = torch.tensor(all_movie_ids_list, dtype=torch.long).to(device)
    all_item_embs = None
    with torch.no_grad():
         # 아이템 타워만 사용하여 계산
         all_item_embs = model.get_item_embedding(all_item_ids_tensor) # Assumes get_item_embedding method exists

    for i in range(0, len(all_users), user_batch_size):
        batch_user_ids = all_users[i : i + user_batch_size]
        batch_user_ids_tensor = torch.tensor(batch_user_ids, dtype=torch.long).to(device)

        with torch.no_grad():
            # 사용자 임베딩 계산
            batch_user_embs = model.get_user_embedding(batch_user_ids_tensor) # Assumes get_user_embedding method exists

            # 모든 아이템과의 유사도 계산 (내적)
            # (batch_users, embedding_dim) x (embedding_dim, num_all_items) -> (batch_users, num_all_items)
            scores = torch.matmul(batch_user_embs, all_item_embs.t())

        # 학습 데이터에서 본 아이템 점수 낮추기 (추천에서 제외)
        for j, user_id in enumerate(batch_user_ids):
            seen_items_indices = [idx for idx, item_id in enumerate(all_movie_ids_list) if (user_id, item_id) in train_user_item_set]
            if seen_items_indices:
                 scores[j, seen_items_indices] = -np.inf # Set score to negative infinity

        # 상위 K개 아이템 추출 및 평가
        _, top_k_indices = torch.topk(scores, k=k, dim=1) # (batch_users, k)
        top_k_item_ids = torch.tensor(all_movie_ids_list)[top_k_indices.cpu()] # (batch_users, k)

        # 배치 내 각 사용자별로 평가 지표 계산
        for j, user_id in enumerate(batch_user_ids):
             pred_items = top_k_item_ids[j].numpy()
             true_items = test_user_item_map.get(user_id, set())
             if not true_items:
                 continue

             hits = len(set(pred_items) & true_items)
             # Precision@K
             precisions.append(hits / k)
             # Recall@K
             recalls.append(hits / len(true_items))
             # NDCG@K
             pred_list = pred_items.tolist()
             true_list = list(true_items)
             # Calculate gain for each predicted item
             gain = [1.0 if item in true_items else 0.0 for item in pred_list]
             # Calculate DCG
             dcg = sum([g / np.log2(rank + 2) for rank, g in enumerate(gain)])
             # Calculate Ideal DCG
             ideal_gain = [1.0] * min(len(true_list), k)
             ideal_dcg = sum([g / np.log2(rank + 2) for rank, g in enumerate(ideal_gain)])
             ndcgs.append(dcg / ideal_dcg if ideal_dcg > 0 else 0.0)


    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0
    avg_ndcg = np.mean(ndcgs) if ndcgs else 0

    return avg_precision, avg_recall, avg_ndcg


# --- Main Execution ---
if __name__ == "__main__":
    # 데이터 로드
    data_dir = "ml-1m-sample"
    train_ratings, val_ratings, test_ratings, num_users, num_movies, num_genres, \
    all_movie_ids_list, train_user_item_set, test_user_item_map, movie_genre_indices = load_data(data_dir)

    if train_ratings is None: # Check if data loading failed
        print("Exiting due to data loading error.")
        exit()


    # 데이터셋 및 데이터로더 생성
    # Note: Validation dataset should ideally not use negative sampling or handle it differently
    train_dataset = MovieLensDataset(train_ratings, all_movie_ids_list, train_user_item_set, movie_genre_indices, is_training=True)
    # For validation/testing, often we evaluate ranking metrics, not loss on sampled negatives
    # val_dataset can be simplified if only used for ranking evaluation
    val_dataset = MovieLensDataset(val_ratings, all_movie_ids_list, train_user_item_set, movie_genre_indices, is_training=False) # is_training=False

    batch_size = 512 # Reduced batch size for sample data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) # Set num_workers=0 for simplicity
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0) # Not typically used for evaluation like this

    # 모델 및 옵티마이저 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    embedding_dim = 32 # Reduced embedding dimension for sample data
    model = TwoTowerModel(num_users, num_movies, num_genres, embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005) # Adjusted learning rate
    num_epochs = 5 # Reduced epochs for sample data

    # 학습 및 검증 루프
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = train_loop(model, train_loader, optimizer, criterion, device)
        end_time = time.time()

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Time: {end_time - start_time:.2f}s")

        # --- 검증 (랭킹 지표 사용) ---
        # Validation loss calculation (as in train_loop) can be misleading if using neg sampling
        # Instead, calculate ranking metrics on the validation set (similar to test set)
        # For simplicity, using the test evaluation logic here on validation data
        # Create val_user_item_map similar to test_user_item_map
        val_user_item_map = val_ratings.groupby('user_id_encoded')['movie_id_encoded'].apply(set).to_dict()
        if val_user_item_map: # Only evaluate if validation set is not empty
             val_precision, val_recall, val_ndcg = evaluate_model(model, val_user_item_map, all_movie_ids_list, train_user_item_set, device, k=10)
             print(f"Epoch [{epoch+1}/{num_epochs}], Validation Metrics @10: Precision={val_precision:.4f}, Recall={val_recall:.4f}, NDCG={val_ndcg:.4f}")
        else:
             print(f"Epoch [{epoch+1}/{num_epochs}], Validation set empty, skipping metrics.")


    # --- 최종 테스트 평가 ---
    print("\n--- Final Test Evaluation ---")
    if test_user_item_map:
        test_precision, test_recall, test_ndcg = evaluate_model(model, test_user_item_map, all_movie_ids_list, train_user_item_set, device, k=10)
        print(f"Test Metrics @10: Precision={test_precision:.4f}, Recall={test_recall:.4f}, NDCG={test_ndcg:.4f}")
    else:
        print("Test set empty, skipping evaluation.")

    # 모델 저장 (옵션)
    # torch.save(model.state_dict(), 'two_tower_model.pth')
    print("\nTraining and evaluation finished.")
