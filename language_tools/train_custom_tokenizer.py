from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers import normalizers
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer

bert_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
bert_tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
bert_tokenizer.pre_tokenizer = Whitespace()

bert_tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", 1),
        ("[SEP]", 2),
    ],
)


trainer = WordPieceTrainer(
    vocab_size=12_000, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)
files = ['data/train_raw.txt']
bert_tokenizer.train(files, trainer)

bert_tokenizer.save("data/custom_tokenizer/tokenizer.json")

tokenizer = Tokenizer.from_file("data/custom_tokenizer/tokenizer.json")

vocab = ['+++']*tokenizer.get_vocab_size()
for token in tokenizer.get_vocab():
    index = tokenizer.get_vocab()[token]
    vocab[index] = token

with open('data/custom_tokenizer/vocab.txt','w') as file:
    file.write("\n".join(vocab))


#
# from pathlib import Path
#
# from tokenizers import ByteLevelBPETokenizer
#
# # paths = [str(x) for x in Path("./data/").glob("**/*.txt")]
# paths = ['data/train_raw.txt']
# # Initialize a tokenizer
# tokenizer = ByteLevelBPETokenizer()
#
# # Customize training
# tokenizer.train(files=paths, vocab_size=15_000, min_frequency=2, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
#
# # Save files to disk
# tokenizer.save_model(".", "rnn_custom")
