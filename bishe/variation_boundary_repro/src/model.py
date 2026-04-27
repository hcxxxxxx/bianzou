from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


class MelEmbedding(nn.Module):
    def __init__(self, embed_dim: int = 24, dropout: float = 0.2) -> None:
        super().__init__()
        first_filters = max(1, embed_dim // 2)
        self.net = nn.Sequential(
            nn.Conv2d(1, first_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 0)),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Conv2d(first_filters, embed_dim, kernel_size=(1, 12), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 0)),
            nn.ELU(),
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, mel]
        x = self.net(x.unsqueeze(1))
        x = x.mean(dim=-1)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return self.dropout(x)


class SACNFolk(nn.Module):
    def __init__(
        self,
        n_mels: int = 128,
        embed_dim: int = 24,
        hidden_size: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.2,
        sr: int = 44100,
        hop_length: int = 512,
        fold_seconds: float = 1.0,
        init_confidence: float = 0.001,
        model_variant: str = "cnn_lstm",
    ) -> None:
        super().__init__()
        self.sr = sr
        self.hop_length = hop_length
        self.fold_seconds = fold_seconds
        self.model_variant = model_variant
        self.fold_size = max(1, int(round(fold_seconds * sr / hop_length)))
        if model_variant == "cnn_lstm":
            self.embedding = MelEmbedding(embed_dim=embed_dim, dropout=dropout)
            lstm_input_dim = embed_dim * self.fold_size
        elif model_variant == "mel_lstm":
            self.embedding = None
            lstm_input_dim = n_mels * self.fold_size
        else:
            raise ValueError(f"unknown model variant: {model_variant}")
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size * 2, 1)
        nn.init.constant_(self.classifier.bias, logit(init_confidence))

    def forward(self, inputs: torch.Tensor, frame_lengths: list[int] | torch.Tensor | None = None) -> torch.Tensor:
        if self.embedding is None:
            embeddings = inputs
        else:
            embeddings = self.embedding(inputs)
        batch, frames, channels = embeddings.shape
        n_folds = frames // self.fold_size
        if n_folds <= 0:
            raise ValueError("input is shorter than one aggregation window")
        embeddings = embeddings[:, : n_folds * self.fold_size, :]
        folded = embeddings.reshape(batch, n_folds, self.fold_size * channels)

        if frame_lengths is not None:
            if isinstance(frame_lengths, torch.Tensor):
                frame_lengths = frame_lengths.detach().cpu().tolist()
            fold_lengths = [max(1, min(n_folds, int(length) // self.fold_size)) for length in frame_lengths]
            packed = pack_padded_sequence(
                folded,
                lengths=fold_lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            packed_outputs, _ = self.lstm(packed)
            outputs, _ = pad_packed_sequence(
                packed_outputs,
                batch_first=True,
                total_length=n_folds,
            )
        else:
            outputs, _ = self.lstm(folded)

        logits = self.classifier(outputs).squeeze(-1)
        return logits
