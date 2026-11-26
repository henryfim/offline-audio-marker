"""Reproducible balanced sample of the official FMA Small archive."""

import argparse
import bz2
import csv
import hashlib
import io
import json
import random
import struct
import time
import urllib.request
import zipfile
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = "https://os.unil.cloud.switch.ch/fma"
AUDIO_URL = f"{SOURCE}/fma_small.zip"
METADATA_URL = f"{SOURCE}/fma_metadata.zip"
AUDIO_SIZE = 7679594875
METADATA_SIZE = 358412441
ARCHIVE_SIZES = {AUDIO_URL: AUDIO_SIZE, METADATA_URL: METADATA_SIZE}
SPLITS = ("train", "validation", "test")
FIELDS = ("path", "label", "split", "track_id", "artist_id", "album_id", "license", "artist", "title", "crc32")


class ZipTail(io.RawIOBase):
    def __init__(self, data, total_size):
        self.data = data
        self.size = total_size
        self.start = total_size - len(data)
        self.offset = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.offset

    def seek(self, offset, whence=0):
        base = (0, self.offset, self.size)[whence]
        self.offset = base + offset
        if self.offset < 0:
            raise OSError("seek before start of archive")
        return self.offset

    def read(self, count=-1):
        if self.offset < self.start:
            raise OSError("requested ZIP directory bytes outside the downloaded tail")
        begin = self.offset - self.start
        end = None if count < 0 else begin + count
        result = self.data[begin:end]
        self.offset += len(result)
        return result


def directory(path, total_size):
    with zipfile.ZipFile(ZipTail(path.read_bytes(), total_size)) as archive:
        return {info.filename: info for info in archive.infolist()}


def download_range(url, first, last):
    expected_length = last - first + 1
    request = urllib.request.Request(url, headers={"Range": f"bytes={first}-{last}"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=24) as response:
                content_range = response.headers.get("Content-Range", "")
                if response.status != 206 or content_range != f"bytes {first}-{last}/{ARCHIVE_SIZES[url]}":
                    raise OSError(f"Unexpected HTTP status/range {response.status} {content_range!r}")
                payload = response.read(expected_length + 1)
                if len(payload) != expected_length:
                    raise OSError(f"Expected {expected_length} bytes, received {len(payload)}")
                return payload
        except OSError:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def compressed_ranges(member, chunk_size):
    end = member.header_offset + 30 + len(member.filename) + 128 + member.compress_size
    return [
        (position, min(end, position + chunk_size) - 1)
        for position in range(member.header_offset, end, chunk_size)
    ]


def decompress(payload, member):
    signature, _, flags, method, _, _, _, _, _, name_len, extra_len = struct.unpack_from(
        "<4s5H3L2H", payload
    )
    if signature != b"PK\x03\x04" or flags & 1 or method != member.compress_type:
        raise ValueError(f"Invalid local ZIP header for {member.filename}")
    if payload[30:30 + name_len].decode("ascii") != member.filename:
        raise ValueError(f"ZIP local filename mismatch for {member.filename}")
    start = 30 + name_len + extra_len
    compressed = payload[start:start + member.compress_size]
    if method == zipfile.ZIP_BZIP2:
        data = bz2.decompress(compressed)
    elif method == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(compressed, wbits=-15)
    elif method == zipfile.ZIP_STORED:
        data = compressed
    else:
        raise ValueError(f"Unsupported ZIP method {method}")
    if len(data) != member.file_size or zlib.crc32(data) != member.CRC:
        raise ValueError(f"ZIP size or CRC mismatch for {member.filename}")
    return data


def write_atomic(destination, payload):
    temp_path = destination.with_suffix(destination.suffix + ".part")
    temp_path.write_bytes(payload)
    temp_path.replace(destination)


def get_directory(data_dir, filename, url, tail_size):
    path = data_dir / filename
    size = ARCHIVE_SIZES[url]
    if not path.exists():
        write_atomic(path, download_range(url, size - tail_size, size - 1))
    return directory(path, size)


def get_tracks_csv(data_dir, workers):
    destination = data_dir / "tracks.csv"
    if destination.exists():
        return destination
    archive = get_directory(data_dir, "metadata_tail.bin", METADATA_URL, 64 * 1024)
    member = archive["fma_metadata/tracks.csv"]
    ranges = compressed_ranges(member, 384 * 1024)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download_range, METADATA_URL, first, last) for first, last in ranges]
        payload = b"".join(future.result() for future in futures)
    write_atomic(destination, decompress(payload, member))
    return destination


def official_rows(tracks_csv, archive):
    with tracks_csv.open(encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        fields = {name: index for index, name in enumerate(zip(next(reader), next(reader)))}
        next(reader)
        for row in reader:
            if row[fields["set", "subset"]] != "small":
                continue
            track_id = f"{int(row[0]):06d}"
            path = f"fma_small/{track_id[:3]}/{track_id}.mp3"
            info = archive[path]
            yield {
                "path": path,
                "label": row[fields["track", "genre_top"]],
                "split": row[fields["set", "split"]].replace("training", "train"),
                "track_id": track_id,
                "artist_id": row[fields["artist", "id"]],
                "album_id": row[fields["album", "id"]],
                "license": row[fields["track", "license"]],
                "artist": row[fields["artist", "name"]],
                "title": row[fields["track", "title"]],
                "crc32": f"{info.CRC:08x}",
            }


def rank_tracks(rows, seed):
    by_group = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_group[row["split"], row["label"]][row["artist_id"]].append(row)
    randomizer = random.Random(seed)
    tracks_by_group = {}
    for group, artist_rows in sorted(by_group.items()):
        artists = sorted(artist_rows)
        randomizer.shuffle(artists)
        for choices in artist_rows.values():
            choices.sort(key=lambda row: row["track_id"])
            randomizer.shuffle(choices)
        candidates = []
        while artists:
            for artist in artists:
                candidates.append(artist_rows[artist].pop())
            artists = [artist for artist in artists if artist_rows[artist]]
        tracks_by_group[group] = candidates
    if len(tracks_by_group) != 24:
        raise ValueError(f"Expected 24 split/genre combinations, found {len(tracks_by_group)}")
    return tracks_by_group


def choose_tracks(tracks_by_group, counts):
    for group, candidates in tracks_by_group.items():
        if len(candidates) < counts[group[0]]:
            raise ValueError(f"{group} only has {len(candidates)} candidates")
    return [
        tracks_by_group[group][rank]
        for rank in range(max(counts.values()))
        for group in sorted(tracks_by_group)
        if rank < counts[group[0]]
    ]


def remove_album_overlap(selected, ranked):
    album_splits = {}
    replacements = {}
    selected_ids = {row["track_id"] for row in selected}
    split_order = ("test", "validation", "train")
    for split in split_order:
        conflicts = []
        for row in selected:
            if row["split"] != split or not row["album_id"]:
                continue
            album_id = row["album_id"]
            if album_splits.setdefault(album_id, split) != split:
                conflicts.append(row)
        for row in conflicts:
            for candidate in ranked[split, row["label"]]:
                album = candidate["album_id"]
                if candidate["track_id"] in selected_ids or (album and album_splits.get(album, split) != split):
                    continue
                selected_ids.add(candidate["track_id"])
                if album:
                    album_splits[album] = split
                replacements[row["track_id"]] = candidate
                break
            else:
                raise ValueError(f"No same-label replacement for conflicting album of {row['track_id']}")
    return [replacements.get(row["track_id"], row) for row in selected], replacements


def verify_split_groups(rows):
    for field in ("track_id", "artist_id", "album_id"):
        assigned_splits = {}
        for row in rows:
            group = row[field]
            split = row["split"]
            if group and assigned_splits.setdefault(group, split) != split:
                raise ValueError(f"{field} {group} appears in more than one split")


def write_manifest(destination, rows):
    temporary = destination.with_suffix(".part")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)


def fetch_track(data_dir, row, member):
    destination = data_dir / row["path"]
    if destination.is_file() and destination.stat().st_size == member.file_size:
        if zlib.crc32(destination.read_bytes()) == member.CRC:
            return
    destination.parent.mkdir(parents=True, exist_ok=True)
    first = member.header_offset
    last = first + 30 + len(member.filename) + 128 + member.compress_size - 1
    payload = download_range(AUDIO_URL, first, last)
    write_atomic(destination, decompress(payload, member))


def obtain_tracks(data_dir, rows, archive, workers):
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(fetch_track, data_dir, row, archive[row["path"]]): row for row in rows}
        completed = 0
        for job in as_completed(jobs):
            row = jobs[job]
            try:
                job.result()
            except Exception as error:
                failures.append(row["track_id"])
                print(f"Failed {row['track_id']}: {error}", flush=True)
                continue
            completed += 1
            if completed % 12 == 0 or completed == len(rows):
                print(f"Verified {completed}/{len(rows)}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} audio entries incomplete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT, help="Directory for audio, metadata, and manifests.")
    parser.add_argument("--train-per-genre", type=int, default=24)
    parser.add_argument("--validation-per-genre", type=int, default=8)
    parser.add_argument("--test-per-genre", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--workers", type=int, default=36)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--only-replacements", action="store_true",
                        help="Fetch only the track substitutions for a previous sample with overlapping albums.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    data_dir = args.output_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    tracks_csv = get_tracks_csv(data_dir, args.workers)
    if args.metadata_only:
        print(f"Verified {tracks_csv.name}", flush=True)
        return
    archive = get_directory(data_dir, "audio_tail.bin", AUDIO_URL, 640 * 1024)
    counts = dict(zip(SPLITS, (args.train_per_genre, args.validation_per_genre, args.test_per_genre)))
    if min(counts.values()) < 1:
        parser.error("each split needs at least one track per genre")
    ranked = rank_tracks(official_rows(tracks_csv, archive), args.seed)
    rows, replacements = remove_album_overlap(choose_tracks(ranked, counts), ranked)
    verify_split_groups(rows)
    write_manifest(data_dir / "planned_manifest_album_disjoint.csv", rows)
    print(f"Planned {len(rows)} FMA Small tracks", flush=True)
    if args.only_replacements:
        print(f"Downloading {len(replacements)} replacements from disjoint albums", flush=True)
        obtain_tracks(data_dir, list(replacements.values()), archive, args.workers)
        return
    obtain_tracks(data_dir, rows, archive, args.workers)
    write_manifest(data_dir / "manifest.csv", rows)
    provenance = {
        "source": "FMA Small",
        "repository": "https://github.com/mdeff/fma",
        "audio_archive": AUDIO_URL,
        "metadata_archive": METADATA_URL,
        "published_archive_sha1": {
            "fma_small.zip": "ade154f733639d52e35e32f5593efe5be76c6d70",
            "fma_metadata.zip": "f0df49ffe5f2a6008d7dc83c6915b31835dfe733",
        },
        "tracks_csv_sha256": hashlib.sha256(tracks_csv.read_bytes()).hexdigest(),
        "sampling_seed": args.seed,
        "sampling_method": (
            "Within each official split and genre, shuffle artists and their tracks deterministically, then visit "
            "artists in rounds. Resolve cross-split albums with same-split and same-genre substitutions, prioritizing "
            "test and then validation tracks."
        ),
        "cross_split_groups": {"track_ids": 0, "artist_ids": 0, "album_ids": 0},
        "album_overlap_replacements": {
            previous: replacement["track_id"] for previous, replacement in replacements.items()
        },
        "tracks_by_split": dict(Counter(row["split"] for row in rows)),
        "tracks_by_label": dict(Counter(row["label"] for row in rows)),
        "audio_verification": (
            "Every selected audio file matches the size and CRC in the official ZIP directory. "
            "Source archive SHA-1 values are provided for reference because downloads used byte ranges."
        ),
        "nominal_clip_seconds": 30,
        "metadata_license": "CC BY 4.0",
        "audio_license": "Individual track license and attribution are listed in manifest.csv.",
    }
    write_atomic(data_dir / "provenance.json", (json.dumps(provenance, indent=2) + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
