import torch
import torch.nn as nn
import torchaudio


class AudioFeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=80,
            power=2.0
        )
        
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80) 
        
 
    def forward(self,audio):
        
        mel = self.mel(audio)
        
        mel = self.db_transform(mel)
        
        return mel



extractor = AudioFeatureExtractor()

audio = torch.randn(1, 16000 * 5)

features = extractor(audio)

print("Input:", audio.shape)
print("Features:", features.shape)




class CharacterTokenizer:

    def __init__(self):

        self.tokens = [
            "<blank>"
        ] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ' ")

        self.char_to_id = {
            char: idx
            for idx, char in enumerate(self.tokens)
        }

        self.id_to_char = {
            idx: char
            for idx, char in enumerate(self.tokens)
        }

        self.blank_id = 0

    @property
    def vocab_size(self):
        return len(self.tokens)

    def normalize(self, text):

        text = text.upper()

        text = re.sub(r"[^A-Z' ]", "", text)

        text = re.sub(r"\s+", " ", text).strip()

        return text

    def encode(self, text):

        text = self.normalize(text)

        return torch.tensor(
            [self.char_to_id[c] for c in text],
            dtype=torch.long
        )

    def decode(self, ids):

        chars = []

        for idx in ids:

            idx = int(idx)

            if idx == self.blank_id:
                continue

            chars.append(self.id_to_char[idx])

        return "".join(chars)


tokenizer = CharacterTokenizer()

text = "HELLO WORLD"

encoded = tokenizer.encode(text)

print("Text:", text)
print("Encoded:", encoded)
print("Decoded:", tokenizer.decode(encoded))
print("Vocabulary:", tokenizer.vocab_size)



from torch.utils.data import Dataset


class SpeechDataset(Dataset):
    
    def __init__(self, dataset):
        
        self.dataset = dataset

        self.extractor = AudioFeatureExtractor()

        self.tokenizer = CharacterTokenizer()

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, index):

        item = self.dataset[index]

        audio_data = item["audio"]

        waveform = torch.tensor(
            audio_data["array"],
            dtype=torch.float32
        )

        sample_rate = audio_data["sampling_rate"]

        if waveform.ndim == 2:
            waveform = waveform.mean(dim=-1)

        if sample_rate != 16000:

            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=16000
            )


        features = self.extractor(waveform)

        features = features.transpose(0, 1)

        targets = self.tokenizer.encode(item["text"])

        return {
            "features": features,
            "targets": targets
        }



sd = SpeechDataset(dataset)

sample = sd[0]

print(sample["features"].shape)
print(sample["targets"].shape)


import torch
from torch.nn.utils.rnn import pad_sequence



def collate_fn(batch):
    
    features = [
        item["features"] for item in batch
    ]
    
    targets = [
        item['targets'] for item in batch
    ] 

    feature_lengths = torch.tensor(
    [x.size(0) for x in features],
    dtype=torch.long
    )
    
    target_lengths = torch.tensor(
        [x.size(0) for x in targets]
    )

    padded_features = pad_sequence(
        features,
        batch_first=True,
        padding_value=0.0
    )
    
    concatenated_targets = torch.cat(targets)

    return {
        "features": padded_features,
        "targets": concatenated_targets,
        "feature_lengths": feature_lengths,
        "target_lengths": target_lengths
    }



from datasets import load_dataset
from torch.utils.data import DataLoader


raw_dataset = load_dataset(
    "openslr/librispeech_asr",
    "clean",
    split="train.100"
)

print("Dataset size:", len(raw_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=10,
    shuffle=True,
    collate_fn=collate_fn,
)



class CNNEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(
                1, 
                32,
            kernel_size=3,
            padding=1
            )
            
            nn.ReLU(),

            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU()
    
    )

    def forward(self, x):

        return self.cnn(x)



class GRUEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.gru = nn.GRU(
            input_size=80 * 64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):

        output, hidden = self.gru(x)

        return output    