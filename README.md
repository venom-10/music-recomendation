# Music Rec

A small, local experiment in finding similar songs from audio features. It extracts a 12-dimensional chroma vector from each track, scales the vectors, and uses cosine-distance nearest neighbors to rank similar tracks. It also reports precision@5 for liked songs against a random-selection baseline.

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Put your own audio files in these folders (create them if needed):

```text
data/
├── liked/   # songs you like
└── random/  # comparison songs
```

The `data/` directory is ignored by Git; audio files are not included in this repository. The current script queries the song at index 7 and requests five other neighbors, so supply **at least eight tracks total**, including liked tracks, before running it.

## Run

From the repository root:

```bash
uv run python main.py
```

The script prints the extracted vectors, the query track and its five closest matches with cosine similarity scores, then precision@5 and the random baseline. Files that cannot be decoded are skipped with an error message. The query index and neighbor count are currently hard-coded in `main.py`.

> Note: The `music-rec` console script configured in `pyproject.toml` still points to the package's placeholder greeting; use `main.py` to run the recommender.
