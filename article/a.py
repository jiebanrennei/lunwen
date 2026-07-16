import kagglehub

# Download latest version
path = kagglehub.dataset_download("carolzhangdc/imdb-5000-movie-dataset")

print("Path to dataset files:", path)