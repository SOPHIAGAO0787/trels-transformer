import regex as re
import collections

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
class BPETokenizer:
    def __init__(self):
        self.vocab = {}
        self.merges = {}
        self.initialize_vocab()
      
        
    def initialize_vocab(self):
        for i in range(256):
            self.vocab[i] = bytes([i])
            
    def train(self, text, vocab_size):
        counts = {}
        
        #Pre-tokenization
        #findall used to scan through the text
        for match in re.finditer(PAT, text):
            #match.group to get the vocab inside
            word_string = match.group()
            word_in_bytes = word_string.encode("utf-8")
            #tuple: cannot be changed
            word_tuple = tuple(word_in_bytes)
            
            #count
            if word_tuple in counts:
                #if counted, then add a count to this
                counts[word_tuple] = counts[word_tuple] + 1
            else:
                counts[word_tuple] = 1
        
        #Merging
        num_merges = vocab_size-256
        for i in range(num_merges):
            pair_counts = self.get_status(counts)
            
            if not pair_counts:
                break
            #Deterministic Tie-breaking
            best_pair = max(pair_counts.keys(), key=lambda p: (pair_counts[p], p))
            
            new_id = 256 + i
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            counts = self.merge_counts(counts, best_pair, new_id)
                    
    def get_status(self, counts):
        #count the number of pairs
        stats = {}
        for word_tuple, freq in counts.items():
            for i in range(len(word_tuple) - 1):
                pair = (word_tuple[i], word_tuple[i+1])
                if pair in stats:
                   stats[pair] = stats[pair] + freq
                else:
                   stats[pair] = freq
        return stats
            
    def merge_counts(self, counts, pair, new_id):
        new_counts = {}
        for word_tuple, freq in counts.items():
            new_word = []
            i = 0
            while i < len(word_tuple):
                if i < len(word_tuple) - 1 and word_tuple[i] == pair[0] and word_tuple[i+1] == pair[1]:
                    new_word.append(new_id)
                    i = i+2
                else:
                    new_word.append(word_tuple[i])
                    i = i+1
            new_counts[tuple(new_word)] = freq
        return new_counts
    
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab.copy()
        self.merges = {pair: idx for idx, pair in enumerate(merges, start=256)}
        
        #special tokens
        self.special_tokens = {} #str->id
        if special_tokens:
            curr_id = max(self.vocab.keys()) + 1 if self.vocab else 0
            for st in special_tokens:
                if st not in self.special_tokens:
                    self.special_tokens[st] = curr_id
                    self.vocab[curr_id] = st.encode("utf-8")
                    curr_id += 1
    
    def encode(self, text: str) -> list[int]:
        tokens = []
        for match in re.finditer(PAT, text):
            word_bytes = match.group().encode("utf-8")
            ids = list(word_bytes)

            while len(ids) >= 2:
                stats = self._get_stats(ids)
                pair = min(stats.keys(), key=lambda p: self.merges.get(p,float('inf')))

                if pair not in self.merges:
                    break
                ids = self._merge(ids,pair,self.merges[pair])
            tokens.extend(ids)
        return tokens
                
            
    def encode_iterable(self,iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)
             
    def decode(self, ids:list[int]) -> str:
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            else:
                pass
        return b"".join(part_bytes).decode("utf-8", errors="replace")
                
    
    