import librosa
import numpy as np
import pathlib as pl
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize


def main():
    paths, labels, vectors = [], [], []

    # label 1 = liked, label 0 = random
    for label, folder in [(1, "data/liked"), (0, "data/random")]:
        for song_file in get_all_files(folder):
            if embedding := get_embeddings(song_file):
                paths.append(song_file)
                labels.append(label)
                vectors.append(embedding)

    X = np.array(vectors)        # shape: (num_songs, 12)
    print(vectors)
    labels = np.array(labels)

    # Scale each of the 12 columns, then make each row length 1
    X = StandardScaler().fit_transform(X)

    # Fit once, use many times
    nn = NearestNeighbors(n_neighbors=6, metric="cosine").fit(X)

    # 1) Query one song
    query_index = 7
    print(f"Query: {paths[query_index]}\n")
    for path, similarity in find_similar(nn, X, paths, query_index, k=5):
        print(f"{similarity:.3f}  {path}")

    # 2) Test the whole system
    score, baseline = precision_at_k(nn, X, labels, k=5)
    print(f"\nprecision@5: {score:.2%}")
    print(f"random baseline: {baseline:.2%}")


def find_similar(nn, X, paths, query_index, k=5):
    # Ask for k+1 neighbors because the closest one is the song itself
    distances, indices = nn.kneighbors(X[query_index].reshape(1, -1), n_neighbors=k + 1)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == query_index:
            continue  # skip the query song
        results.append((paths[idx], 1 - dist))  # cosine similarity = 1 - distance
    return results[:k]


def precision_at_k(nn, X, labels, k=5):
    scores = []
    for i in np.where(labels == 1)[0]:  # only use liked songs as queries
        _, indices = nn.kneighbors(X[i].reshape(1, -1), n_neighbors=k + 1)
        neighbors = [j for j in indices[0] if j != i][:k]
        scores.append(labels[neighbors].mean())  # fraction of neighbors that are liked

    num_liked = labels.sum()
    baseline = (num_liked - 1) / (len(labels) - 1)  # expected score from random guessing
    return np.mean(scores), baseline


def get_all_files(folder: str) -> list[str]:
    return [str(f) for f in pl.Path(folder).rglob("*") if f.is_file()]


def get_embeddings(file_path: str) -> list[float]:
    try:
        y, sr = librosa.load(file_path)

        if y.size == 0:
            raise ValueError("Audio file contains no samples.")

        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        embeddings = np.mean(chroma, axis=1).tolist()
        return embeddings

    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
        return []
    except Exception as e:
        print(f"Error processing audio '{file_path}': {e}")
        return []


if __name__ == "__main__":
    main()