import librosa
import numpy as np
import pathlib as pl


def main():
    liked_song_files = get_all_files("data/liked")
    random_song_files = get_all_files("data/random")
    song_embeddings = {"liked": [], "random": []}

    for song_file in liked_song_files:
        if embedding := get_embeddings(song_file):
            song_embeddings["liked"].append(embedding)

    for song_file in random_song_files:
        if embedding := get_embeddings(song_file):
            song_embeddings["random"].append(embedding)
    print(song_embeddings)




def get_all_files(folder: str) -> list[str]: 
    return [str(f) for f in pl.Path(folder).rglob("*") if f.is_file()]


def get_embeddings(file_path: str) -> list[float]:
    try:
        y, sr = librosa.load(file_path)
        
        # Audio is empty
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


main()
