import os
import multiprocessing
import regex as re
from collections.abc import Iterable, Iterator

from cs336_basics.pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _count_pretokens(text: str, special_tokens: list[str]) -> dict[tuple[int, ...], int]:
    """Pre-tokenize `text` and count pretokens, splitting on special tokens first
    so that BPE merges never cross a special-token (e.g. document) boundary.
    Module-level so it can be pickled and run in a worker process.
    """
    if special_tokens:
        sorted_specials = sorted(special_tokens, key=len, reverse=True)
        special_pat = '|'.join(re.escape(st) for st in sorted_specials)
        segments = re.split(special_pat, text)
    else:
        segments = [text]

    counts = {}
    for segment in segments:
        for match in re.finditer(PAT, segment):
            word_tuple = tuple(match.group().encode("utf-8"))
            counts[word_tuple] = counts.get(word_tuple, 0) + 1
    return counts


def _pretokenize_file_chunk(args):
    """Worker entry point: read this process's byte range out of the file and count pretokens."""
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    # Match Python's text-mode universal-newline translation, since we read bytes directly.
    chunk = chunk.replace("\r\n", "\n").replace("\r", "\n")
    return _count_pretokens(chunk, special_tokens)

class BPETokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab.copy()
        # merges: list of (bytes, bytes) pairs, ordered by creation
        # rebuild int-pair -> new_id mapping using the vocab for lookup
        self.merges = {}
        # build reverse vocab: bytes -> id (for the base 256 tokens it's just the byte value)
        bytes_to_id = {v: k for k, v in self.vocab.items()}
        for idx, (b1, b2) in enumerate(merges, start=256):
            id1 = bytes_to_id[b1]
            id2 = bytes_to_id[b2]
            merged = b1 + b2
            self.merges[(id1, id2)] = idx
            self.vocab[idx] = merged
            bytes_to_id[merged] = idx

        # special tokens
        self.special_tokens = {}  # str -> id
        if special_tokens:
            curr_id = max(self.vocab.keys()) + 1 if self.vocab else 0
            for st in special_tokens:
                if st not in self.special_tokens:
                    self.special_tokens[st] = curr_id
                    self.vocab[curr_id] = st.encode("utf-8")
                    curr_id += 1

    @classmethod
    def train(cls, text, vocab_size):
        word_counts = _count_pretokens(text, special_tokens=[])
        return cls._train_from_word_counts(word_counts, vocab_size)

    @classmethod
    def train_from_file(cls, input_path, vocab_size, special_tokens=None, num_processes=None):
        """Train BPE on a (possibly large) file, parallelizing pre-tokenization across
        processes. The file is split into byte-range chunks whose boundaries always
        fall on a special-token occurrence (via `find_chunk_boundaries`), so no chunk
        -- and no pretoken -- ever straddles a document boundary.
        """
        special_tokens = special_tokens or []
        if num_processes is None:
            num_processes = os.cpu_count() or 1

        # Boundaries must land on a special token so we never split a document/chunk
        # in a way that would let BPE merge across a document boundary.
        split_token = special_tokens[0].encode("utf-8") if special_tokens else b"\n"

        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_processes, split_token)

        chunk_args = [
            (input_path, start, end, special_tokens)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]

        word_counts = {}
        if len(chunk_args) > 1:
            with multiprocessing.Pool(len(chunk_args)) as pool:
                chunk_results = pool.map(_pretokenize_file_chunk, chunk_args)
        else:
            chunk_results = [_pretokenize_file_chunk(args) for args in chunk_args]

        for local_counts in chunk_results:
            for word, c in local_counts.items():
                word_counts[word] = word_counts.get(word, 0) + c

        return cls._train_from_word_counts(word_counts, vocab_size)

    @classmethod
    def _train_from_word_counts(cls, word_counts, vocab_size):
        """Run the merge loop given pretoken counts, indexing pair counts and
        incrementally updating them after each merge (only pairs that overlap the
        merged pair ever change) instead of rescanning every pretoken every merge.
        """
        vocab = {i: bytes([i]) for i in range(256)}
        merges_bytes = []

        words = [list(w) for w in word_counts]
        freqs = list(word_counts.values())

        pair_counts = {}
        pair_to_words = {}  # pair -> set of word indices that (once) contained it
        for idx, w in enumerate(words):
            f = freqs[idx]
            for i in range(len(w) - 1):
                pair = (w[i], w[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + f
                pair_to_words.setdefault(pair, set()).add(idx)

        num_merges = vocab_size - 256
        for i in range(num_merges):
            # only pairs with positive count are real candidates
            candidates = {p: c for p, c in pair_counts.items() if c > 0}
            if not candidates:
                break

            # deterministic tie-breaking
            best_pair = max(candidates.keys(), key=lambda p: (candidates[p], p))
            a, b = best_pair

            new_id = 256 + i
            vocab[new_id] = vocab[a] + vocab[b]
            merges_bytes.append((vocab[a], vocab[b]))

            for idx in pair_to_words.get(best_pair, ()):
                w = words[idx]
                f = freqs[idx]
                if len(w) < 2:
                    continue

                new_w = []
                j = 0
                changed = False
                while j < len(w):
                    if j < len(w) - 1 and w[j] == a and w[j + 1] == b:
                        changed = True
                        # the pairs that bordered this occurrence in the *original*
                        # word no longer exist once (a, b) collapses into new_id
                        if j > 0:
                            left = (w[j - 1], w[j])
                            pair_counts[left] = pair_counts.get(left, 0) - f
                        if j + 2 < len(w):
                            right = (w[j + 1], w[j + 2])
                            pair_counts[right] = pair_counts.get(right, 0) - f
                        new_w.append(new_id)
                        j += 2
                    else:
                        new_w.append(w[j])
                        j += 1

                if not changed:
                    continue

                words[idx] = new_w
                for k in range(len(new_w) - 1):
                    if new_w[k] == new_id or new_w[k + 1] == new_id:
                        p = (new_w[k], new_w[k + 1])
                        pair_counts[p] = pair_counts.get(p, 0) + f
                        pair_to_words.setdefault(p, set()).add(idx)

            pair_counts.pop(best_pair, None)
            pair_to_words.pop(best_pair, None)

        return vocab, merges_bytes

    def _get_stats(self, ids):
        stats = {}
        for pair in zip(ids, ids[1:]):
            stats[pair] = stats.get(pair, 0) + 1
        return stats

    def _merge(self, ids, pair, new_id):
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def _bpe_encode_chunk(self, text: str) -> list[int]:
        """Apply BPE to a plain text chunk (no special tokens)."""
        tokens = []
        for match in re.finditer(PAT, text):
            ids = list(match.group().encode("utf-8"))
            while len(ids) >= 2:
                stats = self._get_stats(ids)
                pair = min(stats.keys(), key=lambda p: self.merges.get(p, float('inf')))
                if pair not in self.merges:
                    break
                ids = self._merge(ids, pair, self.merges[pair])
            tokens.extend(ids)
        return tokens

    def encode(self, text: str) -> list[int]:
        if not self.special_tokens:
            return self._bpe_encode_chunk(text)

        # Split on special tokens — longest first so overlapping tokens match correctly
        sorted_specials = sorted(self.special_tokens.keys(), key=len, reverse=True)
        special_pat = '|'.join(re.escape(st) for st in sorted_specials)
        parts = re.split(f'({special_pat})', text)

        tokens = []
        for part in parts:
            if part in self.special_tokens:
                tokens.append(self.special_tokens[part])
            elif part:
                tokens.extend(self._bpe_encode_chunk(part))
        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
        return b"".join(part_bytes).decode("utf-8", errors="replace")
