import torch
from two_tower_recommendation import TwoTowerModel  # Import the model definition


def load_model(model_path, num_users, num_movies, num_genders, num_occupations, num_zipcodes, embedding_dim, device):
    # Initialize the model
    model = TwoTowerModel(
        num_users, num_movies, num_genders, num_occupations, num_zipcodes, embedding_dim
    ).to(device)

    # Load the saved state dictionary
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set the model to evaluation mode
    return model

def recommend_items(model, user_id, unique_movie_ids, top_n, device):
    # Retrieve user attributes
    user_info = user_data.get(user_id, {"gender": 0, "age": 0, "occupation": 0, "zipcode": 0})
    genders = torch.tensor([user_info["gender"]], dtype=torch.long).to(device)
    ages = torch.tensor([user_info["age"]], dtype=torch.float).to(device)  # Convert to float
    occupations = torch.tensor([user_info["occupation"]], dtype=torch.long).to(device)
    zipcodes = torch.tensor([user_info["zipcode"]], dtype=torch.long).to(device)

    # Convert user ID to tensor
    user_tensor = torch.tensor([user_id], dtype=torch.long).to(device)

    # Compute user embedding
    user_embedding = model.user_tower(user_tensor, genders, ages, occupations, zipcodes)

    # Compute item embeddings
    item_ids_tensor = torch.tensor(unique_movie_ids, dtype=torch.long).to(device)
    item_embeddings = model.item_tower(item_ids_tensor)

    # Compute scores for all items
    scores = torch.matmul(item_embeddings, user_embedding.T).squeeze(1)

    # Get top N recommended items
    top_indices = torch.topk(scores, top_n).indices.cpu().numpy()
    return [unique_movie_ids[i] for i in top_indices]

user_data = {
    1: {"gender": 0, "age": 25, "occupation": 3, "zipcode": 101},
    2: {"gender": 1, "age": 30, "occupation": 5, "zipcode": 102},
}

if __name__ == "__main__":
    # Example configuration
    model_path = "trained_two_tower_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Example data (replace with actual data)
    num_users = 118
    num_movies = 150
    num_genders = 2
    num_occupations = 21
    num_zipcodes = 120
    embedding_dim = 32
    unique_movie_ids = list(range(num_movies))  # Example movie IDs

    # Load the model
    model = load_model(
        model_path=model_path,
        num_users=num_users,
        num_movies=num_movies,
        num_genders=num_genders,
        num_occupations=num_occupations,
        num_zipcodes=num_zipcodes,
        embedding_dim=embedding_dim,
        device=device
    )

    # Recommend items for a specific user
    user_id = 1  # Example user ID
    top_n = 10  # Number of recommendations
    recommended_items = recommend_items(model, user_id, unique_movie_ids, top_n, device)
    print(f"Recommended items for user {user_id}: {recommended_items}")
