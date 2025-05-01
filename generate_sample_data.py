import pandas as pd
import numpy as np
import os
import time
import random

# --- 생성할 데이터 크기 설정 ---
NUM_USERS = 120
NUM_MOVIES = 150
NUM_RATINGS = 600 # 사용자/영화 수보다 훨씬 많게 설정
DATA_DIR = "ml-1m-sample"

# --- 데이터 저장 폴더 생성 ---
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# --- 1. 사용자 데이터 생성 (users.dat) ---
print(f"Generating {NUM_USERS} users...")
user_data = {
    'userId': np.arange(1, NUM_USERS + 1),
    'gender': np.random.choice(['M', 'F'], size=NUM_USERS),
    'age': np.random.choice([1, 18, 25, 35, 45, 50, 56], size=NUM_USERS, p=[0.05, 0.2, 0.3, 0.2, 0.1, 0.1, 0.05]),
    'occupation': np.random.randint(0, 21, size=NUM_USERS), # 0 to 20
    'zipcode': [f"{np.random.randint(10000, 99999)}" for _ in range(NUM_USERS)] # Simple zipcode generation
}
users_df = pd.DataFrame(user_data)

# 파일 저장
users_file_path = os.path.join(DATA_DIR, 'users.dat')
users_df.to_csv(users_file_path, sep=':', header=False, index=False)
print(f"'users.dat' created with {len(users_df)} entries.")

# --- 2. 영화 데이터 생성 (movies.dat) ---
print(f"Generating {NUM_MOVIES} movies...")
movie_ids = np.arange(1, 1 + NUM_MOVIES) # 영화 ID는 101부터 시작 가정

# 샘플 장르 목록
genres_options = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

movie_data = []
base_titles = ["Epic Journey", "Silent Witness", "Cosmic Shift", "Urban Legends", "Forgotten Path",
               "Digital Dream", "Crimson Tide", "Midnight Bloom", "Steel Heart", "Velvet Fog"]

for i, movie_id in enumerate(movie_ids):
    # 제목 생성 (간단한 방식)
    title = f"{random.choice(base_titles)} Part {i % 5 + 1} ({np.random.randint(1980, 2025)})"

    # 장르 생성 (1~3개 랜덤 선택)
    num_genres_for_movie = np.random.randint(1, 4)
    movie_genres = "|".join(np.random.choice(genres_options, size=num_genres_for_movie, replace=False))
    movie_data.append({'movieId': movie_id, 'title': title, 'genres': movie_genres})

movies_df = pd.DataFrame(movie_data)

# 파일 저장
movies_file_path = os.path.join(DATA_DIR, 'movies.dat')
movies_df.to_csv(movies_file_path, sep=':', header=False, index=False, encoding='utf-8') # Encoding utf-8
print(f"'movies.dat' created with {len(movies_df)} entries.")

# --- 3. 평점 데이터 생성 (ratings.dat) ---
print(f"Generating {NUM_RATINGS} ratings...")
# 사용자 ID와 영화 ID 리스트 생성
user_ids_list = users_df['userId'].tolist()
movie_ids_list = movies_df['movieId'].tolist()

rating_data = []
# NUM_RATINGS 만큼 랜덤하게 사용자-영화 쌍 생성 (중복 가능)
random_user_ids = np.random.choice(user_ids_list, size=NUM_RATINGS)
random_movie_ids = np.random.choice(movie_ids_list, size=NUM_RATINGS)

# 평점 생성 (3, 4, 5점에 높은 확률 부여)
ratings_values = np.random.choice([1, 2, 3, 4, 5], size=NUM_RATINGS, p=[0.05, 0.1, 0.3, 0.4, 0.15])

# 타임스탬프 생성 (현재 시간 근처에서 랜덤하게)
current_timestamp = int(time.time())
timestamps = np.random.randint(current_timestamp - 30*24*60*60, current_timestamp, size=NUM_RATINGS) # 최근 한달간

for i in range(NUM_RATINGS):
    rating_data.append({
        'userId': random_user_ids[i],
        'movieId': random_movie_ids[i],
        'rating': ratings_values[i],
        'timestamp': timestamps[i]
    })

ratings_df = pd.DataFrame(rating_data)
# 중복 제거 옵션 (사용자-영화 쌍 기준, 첫번째 것만 남김) - 필요시 주석 해제
# ratings_df = ratings_df.drop_duplicates(subset=['userId', 'movieId'], keep='first')
# print(f"Generated {len(ratings_df)} unique user-movie ratings after potential deduplication.")

# 파일 저장
ratings_file_path = os.path.join(DATA_DIR, 'ratings.dat')
ratings_df.to_csv(ratings_file_path, sep=':', header=False, index=False)
print(f"'ratings.dat' created with {len(ratings_df)} entries.")

print("\nSample data generation complete!")
print(f"Data saved in folder: {DATA_DIR}")
