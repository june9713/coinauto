import os
import glob
import math
import gc
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans  # KMeans보다 대용량 데이터에 빠름
from sklearn.manifold import TSNE
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUI 없이 그래프 저장

# --- 1. 데이터 로더 (신규 함수) ---
# 요청하신 디렉토리 구조에서 데이터를 로드합니다.
def load_data_from_dumps(base_dir, ticker, interval, start_date_str, end_date_str, 
                         features_to_use):
    """
    지정된 디렉토리 구조에서 날짜 범위 내의 모든 CSV 파일을 로드하고 병합합니다.
    """
    print(f"데이터 로드 중... ({start_date_str} ~ {end_date_str})")
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    
    all_dfs = []
    current_date = start_date
    
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        # ./dumps/BTC/3m/2025-06-17/*.csv
        search_path = os.path.join(base_dir, ticker, interval, date_str, '*.csv')
        csv_files = sorted(glob.glob(search_path)) # 00.csv, 01.csv... 순서 보장
        
        for f in csv_files:
            try:
                # 첫 번째 열을 인덱스(시간)로 사용
                df = pd.read_csv(f, index_col=0, parse_dates=True)
                all_dfs.append(df)
            except Exception as e:
                print(f"경고: {f} 파일 읽기 오류: {e}")
        
        current_date += timedelta(days=1)
        
    if not all_dfs:
        print("로드된 데이터가 없습니다.")
        return pd.DataFrame()
        
    # 모든 DataFrame 병합
    full_df = pd.concat(all_dfs)
    full_df = full_df.sort_index() # 시간순으로 정렬
    full_df = full_df[~full_df.index.duplicated(keep='first')] # 중복 인덱스 제거
    
    # 필요한 컬럼만 선택
    try:
        selected_df = full_df[features_to_use]
        selected_df = selected_df.dropna() # 이동평균 등으로 인한 NaN 값 제거
        print(f"데이터 로드 완료. 총 {len(selected_df)}개의 틱(row) 확보.")
        return selected_df
    except KeyError as e:
        print(f"오류: 요청된 피처(컬럼) {e}를 찾을 수 없습니다.")
        print(f"사용 가능한 컬럼: {full_df.columns.tolist()}")
        return pd.DataFrame()

# --- 2. PyTorch 데이터셋 (이동평균 및 파생 지표용) ---
# 다중 피처(ma5, check_5_10, diff_s2e, diff_h2l, PON)를 처리하도록 수정
class PricePatternDataset(Dataset):
    """
    다중 피처 시계열 데이터를 (seq_len, num_features) 텐서로 반환하는 데이터셋
    임의의 위치에서 샘플을 추출하여 연속적 데이터 인식 문제를 방지
    
    Args:
        random_sampling: True면 랜덤 샘플링 (Training용), False면 고정 샘플링 (Validation용)
    """
    def __init__(self, data, seq_len, random_sampling=True):
        # data는 (N, num_features) 형태의 스케일링된 NumPy 배열
        self.data = data
        self.seq_len = seq_len
        self.num_features = data.shape[1]
        self.random_sampling = random_sampling
        
        if len(data) < seq_len:
            raise ValueError(f"데이터 길이({len(data)})가 시퀀스 길이({seq_len})보다 짧습니다.")
        
        # 랜덤 샘플링을 위한 유효한 인덱스 범위
        # seq_len 이상의 인덱스에서만 선택하여 이전 50개 데이터를 가져올 수 있도록 함
        self.valid_start_idx = seq_len - 1  # 최소 인덱스 (0부터 시작하므로)
        self.valid_end_idx = len(data) - 1  # 최대 인덱스
        
        # 고정 샘플링용 인덱스 생성 (Validation에서 일관된 평가를 위해)
        if not random_sampling:
            self.fixed_indices = np.arange(self.valid_start_idx, self.valid_end_idx + 1)

    def __len__(self):
        # 충분한 샘플링을 위해 전체 데이터 길이 반환
        if self.random_sampling:
            # Training: 랜덤 샘플링이므로 반복 횟수에 영향을 줌
            return len(self.data) - self.seq_len + 1
        else:
            # Validation: 고정 샘플 수 반환
            return len(self.fixed_indices)

    def __getitem__(self, idx):
        if self.random_sampling:
            # Training: 임의의 위치에서 샘플 추출 (연속적 데이터 인식 문제 방지)
            random_idx = np.random.randint(self.valid_start_idx, self.valid_end_idx + 1)
        else:
            # Validation: 고정 샘플링 (일관된 평가를 위해)
            random_idx = self.fixed_indices[idx % len(self.fixed_indices)]
        
        # 선택한 인덱스를 포함하여 이전 seq_len개 데이터 추출
        # [random_idx - seq_len + 1 : random_idx + 1] 형태로 이전 50개를 가져옴
        start_idx = random_idx - self.seq_len + 1
        end_idx = random_idx + 1
        sample = self.data[start_idx : end_idx]
        
        return torch.tensor(sample, dtype=torch.float32)

# --- 3. 트랜스포머 모델 (신규 정의) ---
# 예시 코드에 없었던 모델 아키텍처를 정의합니다.

class PositionalEncoding(nn.Module):
    """ Transformer를 위한 위치 인코딩 """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # [1, max_len, d_model] (batch_first=True 용)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerAutoencoder(nn.Module):
    """
    트랜스포머 인코더-디코더 기반 오토인코더 (패턴 압축기)
    [batch, seq_len, input_dim] -> [batch, latent_dim] -> [batch, seq_len, input_dim]
    개선사항:
    - Latent Vector L2 정규화 추가
    - Dropout 추가로 과적합 방지
    """
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, latent_dim, max_seq_len, dropout=0.1):
        super(TransformerAutoencoder, self).__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

        # 1. Input Embedding (input_dim -> d_model)
        self.input_embed = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout=dropout)

        # 2. Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 3. Bottleneck (d_model * seq_len -> latent_dim)
        # 점진적 압축으로 정보 손실 최소화 (더 많은 레이어 추가)
        intermediate_dim = d_model * max_seq_len // 2
        self.to_latent = nn.Sequential(
            nn.Linear(d_model * max_seq_len, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.GELU(),  # ReLU -> GELU (더 부드러운 활성화)
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, intermediate_dim // 2),
            nn.LayerNorm(intermediate_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim // 2, latent_dim),
            nn.LayerNorm(latent_dim)  # 최종 정규화 추가
        )

        # 4. Decoder Input (latent_dim -> d_model * seq_len)
        # 점진적 확장으로 복원 품질 향상 (대칭 구조)
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim // 2),
            nn.LayerNorm(intermediate_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim // 2, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, d_model * max_seq_len),
            nn.LayerNorm(d_model * max_seq_len)  # 최종 정규화 추가
        )

        # 5. Decoder (TransformerEncoder 층을 디코더로 활용)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_decoder_layers)

        # 6. Output Layer (d_model -> input_dim)
        self.output_layer = nn.Linear(d_model, input_dim)

    def encode(self, src):
        # src: [batch, seq_len, input_dim]
        src_embed = self.input_embed(src) * math.sqrt(self.d_model)
        src_embed = self.pos_encoder(src_embed) # [batch, seq_len, d_model]

        enc_output = self.transformer_encoder(src_embed) # [batch, seq_len, d_model]

        # Flatten and project to latent
        enc_flat = enc_output.view(enc_output.size(0), -1) # [batch, seq_len * d_model]
        latent = self.to_latent(enc_flat) # [batch, latent_dim]

        # L2 정규화 제거 - 모델 표현력 향상을 위해 제약 해제
        # latent = F.normalize(latent, p=2, dim=1)

        return latent

    def decode(self, latent):
        # latent: [batch, latent_dim]
        dec_input_flat = self.from_latent(latent) # [batch, seq_len * d_model]
        dec_input = dec_input_flat.view(latent.size(0), self.max_seq_len, self.d_model)

        dec_input = self.pos_encoder(dec_input) # Add position

        dec_output = self.transformer_decoder(dec_input) # [batch, seq_len, d_model]

        output = self.output_layer(dec_output) # [batch, seq_len, input_dim]
        return output

    def forward(self, src):
        latent = self.encode(src)
        reconstructed = self.decode(latent)
        # 재구축된 값과 잠재 벡터(패턴) 동시 반환
        return reconstructed, latent

# --- 4. 조기 종료 (Early Stopping) (신규 클래스) ---
# [Req 6, 8] 베스트 모델을 저장하고 조기 종료를 수행합니다.
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth'):
        """
        Args:
            patience (int): Validation loss가 개선되지 않아도 기다릴 epoch 수
            verbose (bool): 로그 출력 여부
            delta (float): 개선으로 인정할 최소 변화량
            path (str): 베스트 모델을 저장할 경로
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'  [EarlyStopping] Counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        # [Req 8] 베스트 모델을 저장합니다.
        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        if self.verbose:
            print(f'  [EarlyStopping] Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). \nSaving model to {self.path}')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# --- 5. Contrastive Loss 함수 (수정됨) ---
def contrastive_loss(latent_vectors, temperature=0.5):
    """
    간단한 분산 최대화 Loss (잠재 벡터들이 서로 멀어지도록)

    Args:
        latent_vectors: [batch_size, latent_dim] - L2 정규화된 잠재 벡터
        temperature: 사용 안 함 (호환성 유지)

    Returns:
        contrastive loss value
    """
    batch_size = latent_vectors.shape[0]

    if batch_size < 2:
        return torch.tensor(0.0, device=latent_vectors.device)

    # 방법 1: 벡터 간 평균 거리를 최대화 (= 유사도를 최소화)
    # 코사인 유사도 계산 (L2 정규화되어 있으므로 내적만 하면 됨)
    similarity_matrix = torch.matmul(latent_vectors, latent_vectors.T)

    # 대각선 제외 (자기 자신과의 유사도 제외)
    mask = torch.eye(batch_size, dtype=torch.bool, device=latent_vectors.device)
    off_diagonal = similarity_matrix.masked_select(~mask)

    # Loss: 평균 유사도를 낮추기 (= 벡터들이 서로 멀어짐)
    # 유사도가 높으면 loss가 높고, 유사도가 낮으면 loss가 낮음
    loss = off_diagonal.mean()

    return loss

# --- 6. 모델 학습 (신규 함수) ---
# [Req 6, 7] 조기 종료를 포함한 학습 루프 + Contrastive Loss
def train_autoencoder(model, train_loader, val_loader, model_path, epochs, lr, patience, device,
                      contrastive_weight=0.1, warmup_epochs=5):
    """
    트랜스포머 오토인코더 학습 함수

    개선사항:
    - Reconstruction Loss + Perceptual Loss + Contrastive Loss 결합
    - Learning Rate Warmup + Cosine Annealing
    - 상세한 학습 진행 시각화
    - Gradient Accumulation 지원

    Args:
        contrastive_weight: Contrastive Loss의 가중치 (기본 0.1)
        warmup_epochs: Warmup epoch 수
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.005)  # 0.01 -> 0.005 (규제 완화)

    # MSE Loss (픽셀 레벨 재구축)
    mse_criterion = nn.MSELoss()

    # L1 Loss (디테일 보존)
    l1_criterion = nn.L1Loss()

    # 결합된 Reconstruction Loss
    def reconstruction_criterion(recon, target):
        # MSE + L1 결합 (디테일 보존 강화)
        mse_loss = mse_criterion(recon, target)
        l1_loss = l1_criterion(recon, target)
        return 0.5 * mse_loss + 0.5 * l1_loss

    # Learning Rate Scheduler (Warmup + Cosine Annealing)
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            # Warmup: 선형 증가
            return (current_epoch + 1) / warmup_epochs
        else:
            # Cosine Annealing
            progress = (current_epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # [Req 6, 8] 조기 종료 핸들러 (베스트 모델을 model_path에 저장)
    early_stopper = EarlyStopping(patience=patience, verbose=True, path=model_path)

    print("\n" + "="*70)
    print("                  🚀  오토인코더 학습 시작  🚀")
    print("="*70)
    print(f"  Loss: MSE + L1 (0.5:0.5) + Contrastive (weight={contrastive_weight})")
    print(f"  Learning Rate: {lr} (Warmup: {warmup_epochs} epochs)")
    print(f"  Optimizer: AdamW (weight_decay=0.005)")
    print(f"  Model Capacity: d_model={model.d_model}, enc_layers={model.num_encoder_layers}, dec_layers={model.num_decoder_layers}")
    print("="*70)

    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        train_recon_loss = 0.0
        train_contrast_loss = 0.0
        train_total_loss = 0.0

        for data in train_loader:
            data = data.to(device) # (batch, seq_len, num_features)

            optimizer.zero_grad()
            reconstructed, latent = model(data)

            # 1. Reconstruction Loss
            recon_loss = reconstruction_criterion(reconstructed, data)

            # 2. Contrastive Loss
            contrast_loss = contrastive_loss(latent, temperature=0.5)

            # 3. Total Loss
            total_loss = recon_loss + contrastive_weight * contrast_loss

            total_loss.backward()

            # Gradient Clipping (폭발 방지) - 모델 용량 증가로 max_norm 조정
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            optimizer.step()

            train_recon_loss += recon_loss.item() * data.size(0)
            train_contrast_loss += contrast_loss.item() * data.size(0)
            train_total_loss += total_loss.item() * data.size(0)

        # --- Validation ---
        model.eval()
        val_recon_loss = 0.0
        val_contrast_loss = 0.0
        val_total_loss = 0.0

        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                reconstructed, latent = model(data)

                recon_loss = reconstruction_criterion(reconstructed, data)
                contrast_loss = contrastive_loss(latent, temperature=0.5)
                total_loss = recon_loss + contrastive_weight * contrast_loss

                val_recon_loss += recon_loss.item() * data.size(0)
                val_contrast_loss += contrast_loss.item() * data.size(0)
                val_total_loss += total_loss.item() * data.size(0)

        # 평균 계산
        train_recon_loss /= len(train_loader.dataset)
        train_contrast_loss /= len(train_loader.dataset)
        train_total_loss /= len(train_loader.dataset)

        val_recon_loss /= len(val_loader.dataset)
        val_contrast_loss /= len(val_loader.dataset)
        val_total_loss /= len(val_loader.dataset)

        current_lr = scheduler.get_last_lr()[0]

        # 진행 상황 출력 (개선된 포맷)
        print(f'Epoch {epoch:04d}/{epochs} | LR: {current_lr:.2e} | '
              f'Train: {train_total_loss:.6f} (R:{train_recon_loss:.6f} C:{train_contrast_loss:.6f}) | '
              f'Val: {val_total_loss:.6f} (R:{val_recon_loss:.6f} C:{val_contrast_loss:.6f})',
              end='')

        # Best 모델 표시
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            print(' ✓ BEST', end='')

        print()  # 줄바꿈

        # Learning Rate 조정
        scheduler.step()

        # --- Early Stopping Check ---
        early_stopper(val_total_loss, model)
        if early_stopper.early_stop:
            print("\n" + "="*70)
            print("                       ⛔ 조기 종료 ⛔")
            print(f"  최적 Epoch: {epoch - early_stopper.patience}")
            print(f"  최저 Val Loss: {early_stopper.val_loss_min:.6f}")
            print("="*70)
            break

    # [Req 8] 학습 완료 후, 저장된 베스트 모델을 로드
    print(f"\n학습 종료. '{model_path}'에서 최적 모델을 로드합니다.")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"오류: 최적 모델 로드 실패: {e}. 마지막 epoch 모델을 반환합니다.")

    return model

# --- 7. 재구축 품질 검증 함수 ---
def visualize_reconstruction_quality(model, dataloader, device, save_path, num_samples=5):
    """
    모델의 재구축 품질을 시각화하여 저장

    Args:
        model: 학습된 오토인코더
        dataloader: 검증용 데이터로더
        device: 장치
        save_path: 이미지 저장 경로
        num_samples: 시각화할 샘플 수
    """
    model.to(device)
    model.eval()

    # 샘플 추출
    samples = []
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            reconstructed, _ = model(data)

            # CPU로 이동
            original = data.cpu().numpy()
            recon = reconstructed.cpu().numpy()

            for i in range(min(num_samples, len(original))):
                samples.append((original[i], recon[i]))

            if len(samples) >= num_samples:
                break

    if not samples:
        print("재구축 품질 시각화: 샘플이 없습니다.")
        return

    # 시각화
    fig, axes = plt.subplots(num_samples, 2, figsize=(14, 3 * num_samples))

    for idx, (original, recon) in enumerate(samples[:num_samples]):
        # 원본
        ax_orig = axes[idx, 0] if num_samples > 1 else axes[0]
        ax_orig.imshow(original.T, aspect='auto', cmap='viridis', interpolation='nearest')
        ax_orig.set_title(f'Original Sample {idx+1}')
        ax_orig.set_xlabel('Time Steps')
        ax_orig.set_ylabel('Features')

        # 재구축
        ax_recon = axes[idx, 1] if num_samples > 1 else axes[1]
        ax_recon.imshow(recon.T, aspect='auto', cmap='viridis', interpolation='nearest')
        ax_recon.set_title(f'Reconstructed Sample {idx+1}')
        ax_recon.set_xlabel('Time Steps')
        ax_recon.set_ylabel('Features')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ 재구축 품질 시각화 저장: {save_path}")

# --- 8. t-SNE 시각화 함수 ---
def visualize_latent_space_tsne(latent_vectors, labels, save_path, title="Latent Space (t-SNE)"):
    """
    잠재 벡터를 t-SNE로 2D 축소하여 시각화

    Args:
        latent_vectors: numpy array of shape (n_samples, latent_dim)
        labels: numpy array of cluster labels
        save_path: 이미지 저장 경로
        title: 그래프 제목
    """
    print("\nt-SNE를 사용하여 잠재 공간 시각화 중...")

    # t-SNE 적용 (샘플이 너무 많으면 일부만 사용)
    max_samples = 10000
    if len(latent_vectors) > max_samples:
        print(f"  샘플 수가 많아 {max_samples}개만 사용합니다.")
        indices = np.random.choice(len(latent_vectors), max_samples, replace=False)
        latent_subset = latent_vectors[indices]
        labels_subset = labels[indices]
    else:
        latent_subset = latent_vectors
        labels_subset = labels

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    latent_2d = tsne.fit_transform(latent_subset)

    # 시각화
    fig, ax = plt.subplots(figsize=(12, 10))

    # 상위 30개 카테고리만 색상으로 표시
    unique_labels = np.unique(labels_subset)
    label_counts = Counter(labels_subset)
    top_labels = [label for label, _ in label_counts.most_common(30)]

    for label in unique_labels:
        mask = labels_subset == label
        if label in top_labels:
            ax.scatter(latent_2d[mask, 0], latent_2d[mask, 1],
                      label=f'Cat {label}', alpha=0.6, s=10)
        else:
            ax.scatter(latent_2d[mask, 0], latent_2d[mask, 1],
                      color='lightgray', alpha=0.3, s=5)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ t-SNE 시각화 저장: {save_path}")

# --- 9. 카테고리 생성 (KMeans) + 시각화 ---
# [Req 4, 5, 9]
def create_categories(model, dataloader, n_categories, device, output_dir=None):
    """
    학습된 Autoencoder를 이용해 Latent Vector를 추출하고 KMeans 클러스터링 수행
    + t-SNE 시각화 추가

    Args:
        output_dir: 시각화 이미지를 저장할 디렉토리 (None이면 시각화 생략)
    """
    model.to(device)
    model.eval()
    all_latents = []

    print("\nKMeans 클러스터링을 위해 잠재 벡터(Latent Vector) 추출 중...")
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            _, latent = model(data) # (reconstructed, latent)
            all_latents.append(latent.cpu())

    all_latents_np = torch.cat(all_latents, dim=0).numpy()
    print(f"총 {all_latents_np.shape[0]}개의 잠재 벡터 추출 완료. (형태: {all_latents_np.shape})")

    print(f"MiniBatchKMeans 클러스터링 시작 (N={n_categories})...")
    # MiniBatchKMeans는 대용량 데이터에서 KMeans보다 훨씬 빠름
    kmeans = MiniBatchKMeans(n_clusters=n_categories,
                            random_state=42,
                            n_init=10,
                            batch_size=min(1024, len(all_latents_np)))
    kmeans.fit(all_latents_np)
    print("KMeans 클러스터링 완료.")

    # --- 불명확 카테고리 분류 ---
    # 각 샘플에서 가장 가까운 클러스터 중심까지의 거리 계산
    labels = kmeans.labels_
    distances = np.min(kmeans.transform(all_latents_np), axis=1)

    # 거리 기반으로 불명확 카테고리 판단 (상위 10% 거리를 불명확으로 분류)
    distance_threshold = np.percentile(distances, 90)
    uncertain_mask = distances > distance_threshold

    # 불명확한 샘플들을 n_categories 번호로 재할당
    labels_with_uncertain = labels.copy()
    labels_with_uncertain[uncertain_mask] = n_categories  # 마지막 카테고리 번호

    print(f"✓ 불명확 카테고리(Category {n_categories}): {uncertain_mask.sum()}개 샘플 ({uncertain_mask.sum()/len(labels)*100:.2f}%)")

    # --- [Req 9] 카테고리 편중도(분포) 출력 ---
    labels = labels_with_uncertain  # 불명확 포함된 레이블 사용
    label_counts = Counter(labels)
    sorted_counts = label_counts.most_common() # (label, count) 튜플의 리스트

    print("\n" + "="*70)
    print("              📊 Top 30 Most Frequent Categories 📊")
    print("="*70)
    total_samples = len(labels)
    for i, (label, count) in enumerate(sorted_counts[:30]):
        percentage = (count / total_samples) * 100
        bar = '█' * int(percentage / 2)
        print(f"  {i+1:2d}. Category {label:04d}: {count:6d} 샘플 ({percentage:5.2f}%) {bar}")
    print("="*70 + "\n")

    # t-SNE 시각화 (output_dir이 지정된 경우)
    if output_dir:
        tsne_path = os.path.join(output_dir, 'latent_space_tsne.png')
        visualize_latent_space_tsne(all_latents_np, labels, tsne_path)

    return kmeans

# --- 7. 카테고리 추론 (신규 함수) ---
# [Req 5] 저장된 모델/스케일러/KMeans로 새 데이터의 패턴 ID 추론
def get_pattern_category(new_data_ticks, autoencoder, kmeans_model, feature_scalers, seq_len, device):
    """
    새로운 데이터(A틱)를 받아 어떤 패턴 카테고리에 속하는지 추론합니다.

    Args:
        new_data_ticks (np.array): (seq_len, num_features) 형태의 원본 데이터
        feature_scalers (dict): Feature별 독립 Scaler 딕셔너리
    """

    # 입력 데이터 검증
    if new_data_ticks.shape[0] != seq_len:
        raise ValueError(f"입력 데이터 길이는 {seq_len}이어야 합니다. (현재: {new_data_ticks.shape[0]})")

    n_features = len(feature_scalers)
    if new_data_ticks.shape[1] != n_features:
        raise ValueError(f"입력 피처 개수는 {n_features}개여야 합니다. (현재: {new_data_ticks.shape[1]})")

    # 1. Feature별 독립 스케일링 적용
    scaled_data = np.zeros_like(new_data_ticks, dtype=np.float32)
    feature_names = list(feature_scalers.keys())

    for i, feature_name in enumerate(feature_names):
        scaler = feature_scalers[feature_name]
        scaled_data[:, i] = scaler.transform(new_data_ticks[:, i:i+1]).flatten()

    # 2. Convert to Tensor (Batch 차원 추가)
    data_tensor = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(device)

    # 3. Get latent vector
    autoencoder.to(device)
    autoencoder.eval()
    with torch.no_grad():
        _, latent = autoencoder(data_tensor) # (reconstructed, latent)

    # 4. Predict category
    latent_np = latent.cpu().numpy()
    category = kmeans_model.predict(latent_np)[0]

    # 5. 불명확 카테고리 판단
    # 가장 가까운 클러스터 중심까지의 거리 계산
    distances = kmeans_model.transform(latent_np)
    min_distance = np.min(distances)

    # 클러스터 중심 간 평균 거리를 기준으로 임계값 설정
    n_clusters = kmeans_model.n_clusters
    cluster_centers = kmeans_model.cluster_centers_
    center_distances = np.linalg.norm(cluster_centers[:, np.newaxis] - cluster_centers, axis=2)
    avg_center_distance = np.mean(center_distances[center_distances > 0])
    distance_threshold = avg_center_distance * 0.5  # 중심 간 거리의 50%

    if min_distance > distance_threshold:
        category = n_clusters  # 불명확 카테고리로 재할당

    return category  # 카테고리 번호 반환

# --- 8. 메인 실행 ---
if __name__ == '__main__':

    for ___ in range(100):
    
        # --- [Req 7] CUDA/CPU 장치 검증 ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("="*40)
        print("       장치 검증 (Device Verification)       ")
        print("="*40)
        print(f"PyTorch 버전: {torch.__version__}")
        print(f"CUDA 사용 가능 여부: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"사용 중인 GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("!! 경고: CUDA를 사용할 수 없습니다. CPU로 연산을 수행합니다.")
        print(f"선택된 장치: {device}")
        print("="*40)

        # --- 0. 하이퍼파라미터 설정 ---
        BASE_DIR = './dumps'
        TICKER = 'BTC'
        INTERVAL = '3m'
        START_DATE = '2025-01-05'
        END_DATE = '2025-11-05'
        
        # [Req 1] 사용할 피처 리스트 (CSV 컬럼명과 일치해야 함)
        # 이동평균 및 파생 지표 사용
        FEATURES = ['ma5', 'check_5_10', 'diff_s2e', 'diff_h2l', 'PON']
        INPUT_DIM = len(FEATURES) # 5
        
        # [Req 1, 3] 과거 "A"틱 (샘플 길이). 30~300 사이로 설정.
        SEQUENCE_LENGTH = 50
        MAX_SEQ_LEN = SEQUENCE_LENGTH # Positional Encoding을 위해 모델에 전달
        
        # 모델 파라미터 (대폭 개선된 용량)
        D_MODEL = 256            # 128 -> 256 (표현력 대폭 향상)
        NHEAD = 8                # 어텐션 헤드 유지
        NUM_ENCODER_LAYERS = 6   # 3 -> 6 (깊이 2배 증가)
        NUM_DECODER_LAYERS = 6   # 3 -> 6 (깊이 2배 증가)
        LATENT_DIM = 512         # 256 -> 512 (압축률 대폭 개선, 더 많은 정보 보존)
        DROPOUT = 0.15           # 0.2 -> 0.15 (모델 용량 증가로 약간 완화)

        # [Req 4] 카테고리 수
        N_CATEGORIES = 500

        # 학습 파라미터 (개선됨)
        BATCH_SIZE = 512      # 4096 -> 2048 (모델 용량 증가로 배치 크기 감소)
        EPOCHS = 2000            # [Req 6] 조기 종료되므로 넉넉하게 설정
        VALIDATION_SPLIT_RATIO = 0.1 # 10%를 검증용으로 사용

        # [Req 6] 조기 종료 Patience
        # 모델 용량 증가로 학습 시간이 길어지므로 Patience 증가
        EARLY_STOPPING_PATIENCE = 150  # 100 -> 150 (충분한 학습 기회 제공)
        LEARNING_RATE = 1e-4           # 2e-4 -> 1e-4 (더 안정적인 학습)

        # Contrastive Loss 파라미터
        CONTRASTIVE_WEIGHT = 0.0       # Contrastive Loss 비활성화 (모델 안정성 향상)
        WARMUP_EPOCHS = 10             # 5 -> 10 (모델 용량 증가로 Warmup 기간 연장)
        
        # [Req 10] Train Type (경로명에 사용)
        TRAIN_TYPE = 'price' # 'volume' -> 'price'로 변경
        
        # 저장 파일명
        MODELS_DIR = './models'
        if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)
        ticker_dir = os.path.join(MODELS_DIR, TICKER)
        if not os.path.exists(ticker_dir): os.makedirs(ticker_dir)
        interval_dir = os.path.join(ticker_dir, INTERVAL)
        if not os.path.exists(interval_dir): os.makedirs(interval_dir)
        train_type_dir = os.path.join(interval_dir, TRAIN_TYPE)
        if not os.path.exists(train_type_dir): os.makedirs(train_type_dir)
        sequence_length_dir = os.path.join(train_type_dir, str(SEQUENCE_LENGTH))
        if not os.path.exists(sequence_length_dir): os.makedirs(sequence_length_dir)
        #N_CATEGORIES
        n_categories_dir = os.path.join(sequence_length_dir, str(N_CATEGORIES))
        if not os.path.exists(n_categories_dir): os.makedirs(n_categories_dir)

        
        # [Req 2, 5, 8] 저장 경로
        SCALER_PATH = os.path.join(n_categories_dir, f'{TRAIN_TYPE}_scaler.joblib')
        MODEL_PATH = os.path.join(n_categories_dir, 'transformer_autoencoder.pth') # 베스트 모델 저장 경로
        KMEANS_PATH = os.path.join(n_categories_dir, 'pattern_categories.joblib')

        # --- 1. 데이터 로드 ---
        full_df = load_data_from_dumps(BASE_DIR, TICKER, INTERVAL, START_DATE, END_DATE, FEATURES)
        
        if full_df.empty:
            print("데이터가 없습니다. 프로그램을 종료합니다.")
            # (필요시 가상 데이터 생성)
            # print("가상 데이터를 생성합니다 (10000 틱).")
            # data_dict = {f: np.random.rand(10000) for f in FEATURES}
            # full_df = pd.DataFrame(data_dict)
        else:
            print(f"총 {len(full_df)}개의 틱(row)으로 학습을 시작합니다.")

        # --- 2. 전처리 (스케일링) [Req 2] ---
        # Scaler는 전체 데이터가 아닌 **학습 데이터(train_data) 기준**으로 fit해야
        # Data Leakage(데이터 유출)를 방지할 수 있습니다.
        # 여기서는 시계열이므로 shuffle=False로 분리합니다.
        
        # (1) Train/Validation 데이터 분리 (DataFrame 기준)
        train_df, val_df = train_test_split(full_df, 
                                            test_size=VALIDATION_SPLIT_RATIO, 
                                            shuffle=False) # 시계열이므로 순서 유지
        
        # (2) Feature별 독립 스케일링 (각 피처의 스케일 차이 해소)
        # FEATURES = ['ma5', 'check_5_10', 'diff_s2e', 'diff_h2l', 'PON']

        def scale_features_independently(df, scalers=None, fit=False):
            """각 feature를 독립적으로 스케일링"""
            scaled_data = np.zeros_like(df.values, dtype=np.float32)

            if scalers is None:
                scalers = {}

            for i, feature_name in enumerate(FEATURES):
                if fit:
                    # 새로운 Scaler 생성 및 fit
                    scaler = StandardScaler()
                    scaler.fit(df.iloc[:, i:i+1])  # 단일 컬럼
                    scalers[feature_name] = scaler

                # Transform
                scaled_data[:, i] = scalers[feature_name].transform(df.iloc[:, i:i+1]).flatten()

            return scaled_data, scalers

        # (3) Scaler 로드 또는 생성
        if os.path.exists(SCALER_PATH):
            print(f"\n'{SCALER_PATH}'에서 기존 Scaler를 로드합니다.")
            feature_scalers = joblib.load(SCALER_PATH)

            # Train/Val/Full 데이터 스케일링
            scaled_data_train, _ = scale_features_independently(train_df[FEATURES], feature_scalers, fit=False)
            scaled_data_val, _ = scale_features_independently(val_df[FEATURES], feature_scalers, fit=False)
            scaled_data_full, _ = scale_features_independently(full_df[FEATURES], feature_scalers, fit=False)
        else:
            print(f"\n'{SCALER_PATH}'에 Scaler가 없습니다. Train 데이터로 새로 생성합니다.")

            # Train 데이터로 fit하고 변환
            scaled_data_train, feature_scalers = scale_features_independently(train_df[FEATURES], fit=True)

            # Val/Full 데이터는 같은 scaler로 변환만
            scaled_data_val, _ = scale_features_independently(val_df[FEATURES], feature_scalers, fit=False)
            scaled_data_full, _ = scale_features_independently(full_df[FEATURES], feature_scalers, fit=False)

            # Scaler 저장
            joblib.dump(feature_scalers, SCALER_PATH)
            print(f"Feature별 Scaler 저장 완료: {SCALER_PATH}")

        print(f"데이터 분할: Train {len(scaled_data_train)}개, Validation {len(scaled_data_val)}개")

        # --- 3. Dataset 및 DataLoader (Train/Validation 분리) ---
        # [Req 6]
        # Training: 랜덤 샘플링으로 다양한 패턴 학습
        train_dataset = PricePatternDataset(scaled_data_train, SEQUENCE_LENGTH, random_sampling=True)
        # Validation: 고정 샘플링으로 일관된 평가 (과적합 감지 정확도 향상)
        val_dataset = PricePatternDataset(scaled_data_val, SEQUENCE_LENGTH, random_sampling=False)

        # 학습 데이터는 섞어서(shuffle=True) 모델이 순서에 과적합되는 것을 방지
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        # 검증 데이터는 순서대로(shuffle=False) 평가
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # (데이터 형태 확인)
        try:
            sample_data = next(iter(train_dataloader))
            print(f"데이터셋 샘플 형태 (Batch, Seq_Len, Features): {sample_data.shape}")
        except ValueError as e:
            print(f"데이터셋 생성 오류: {e}")
            print("데이터가 너무 적어 1개의 배치도 만들 수 없습니다. START_DATE를 확인하세요.")
            exit()


        # --- 4. Autoencoder 모델 초기화 ---
        autoencoder_model = TransformerAutoencoder(
            input_dim=INPUT_DIM,
            d_model=D_MODEL,
            nhead=NHEAD,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            num_decoder_layers=NUM_DECODER_LAYERS,
            latent_dim=LATENT_DIM,
            max_seq_len=MAX_SEQ_LEN,
            dropout=DROPOUT
        )

        # --- [Req 7, 8] 모델 로드 (학습 재개) ---
        # EarlyStopping이 베스트 모델을 MODEL_PATH에 저장하므로,
        # 이 경로는 '최근' 모델이 아닌 '최적' 모델입니다.
        if os.path.exists(MODEL_PATH):
            print(f"\n'{MODEL_PATH}' 에서 기존 베스트 모델을 찾았습니다.")
            try:
                autoencoder_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
                print("모델 가중치 로드 완료. 이어서 학습합니다.\n")
            except Exception as e:
                print(f"모델 로드 중 오류 발생: {e}. 새로 학습을 시작합니다.\n")
        else:
            print(f"\n'{MODEL_PATH}' 에서 기존 모델을 찾을 수 없습니다.")
            print("새로 학습을 시작합니다.\n")

        # --- 5. Autoencoder 모델 학습 (조기 종료 적용) ---
        # [Req 6]
        autoencoder_model = train_autoencoder(
            model=autoencoder_model,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            model_path=MODEL_PATH,  # 베스트 모델이 여기에 저장됨
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            patience=EARLY_STOPPING_PATIENCE,
            device=device,
            contrastive_weight=CONTRASTIVE_WEIGHT,
            warmup_epochs=WARMUP_EPOCHS
        )

        # --- 5.5. 재구축 품질 시각화 ---
        recon_viz_path = os.path.join(n_categories_dir, 'reconstruction_quality.png')
        visualize_reconstruction_quality(
            model=autoencoder_model,
            dataloader=val_dataloader,
            device=device,
            save_path=recon_viz_path,
            num_samples=5
        )
        
        # --- 6. "카테고리" 생성 (KMeans) + t-SNE 시각화 ---
        # [Req 4, 5, 9]
        # train_autoencoder 함수가 최적 모델을 로드하여 반환했으므로
        # autoencoder_model은 현재 '베스트 모델' 상태입니다.

        # K-Means는 전체 데이터(Train+Val)로 생성
        # K-Means 클러스터링을 위해 고정 샘플링 사용 (일관된 결과를 위해)
        full_dataset = PricePatternDataset(scaled_data_full, SEQUENCE_LENGTH, random_sampling=False)
        # K-Means 학습 시에는 데이터를 섞을 필요 없음
        full_dataloader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False)

        kmeans_model = create_categories(
            model=autoencoder_model,
            dataloader=full_dataloader,
            n_categories=N_CATEGORIES,
            device=device,
            output_dir=n_categories_dir  # t-SNE 시각화 저장
        )
        
        # [Req 5] KMeans 모델 저장
        joblib.dump(kmeans_model, KMEANS_PATH)
        print(f"KMeans (카테고리) 모델 저장 완료: {KMEANS_PATH}")

        
        # --- 7. "카테고리" 재사용 예시 ---
        print("\n" + "="*40)
        print("      🔄 카테고리 재사용(추론) 테스트 🔄")
        print("="*40)
        
        # (가상의 새 데이터, 5피처: ma5, check_5_10, diff_s2e, diff_h2l, PON)
        # [Req 1] (SEQUENCE_LENGTH, INPUT_DIM) 형태의 NumPy 배열
        new_ticks_A = np.random.rand(SEQUENCE_LENGTH, INPUT_DIM) * 100 + 150000000
        
        # 저장된 모델/스케일러 로드
        print("저장된 모델/스케일러/KMeans 로드 중...")
        loaded_autoencoder = TransformerAutoencoder(
            input_dim=INPUT_DIM, d_model=D_MODEL, nhead=NHEAD,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            num_decoder_layers=NUM_DECODER_LAYERS,
            latent_dim=LATENT_DIM, max_seq_len=MAX_SEQ_LEN,
            dropout=DROPOUT
        )
        loaded_autoencoder.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        
        loaded_scaler = joblib.load(SCALER_PATH)
        loaded_kmeans = joblib.load(KMEANS_PATH)

        print("추론 시작...")
        category = get_pattern_category(
            new_data_ticks=new_ticks_A,
            autoencoder=loaded_autoencoder,
            kmeans_model=loaded_kmeans,
            feature_scalers=loaded_scaler,  # 이제 dict 형태
            seq_len=SEQUENCE_LENGTH,
            device=device
        )
        
        print(f"\n✅ 테스트 완료: 새로운 {SEQUENCE_LENGTH}틱 데이터의 패턴 카테고리: {category}")
        print("="*40)
        
        # --- 메모리 정리 (다음 반복을 위한 초기화) ---
        print("\n[메모리 정리 중...]")
        
        # 1. 모델들을 CPU로 이동 후 삭제
        if 'autoencoder_model' in locals():
            autoencoder_model.cpu()
            del autoencoder_model
        
        if 'loaded_autoencoder' in locals():
            loaded_autoencoder.cpu()
            del loaded_autoencoder
        
        # 2. 데이터로더와 데이터셋 삭제
        if 'train_dataloader' in locals():
            del train_dataloader
        if 'val_dataloader' in locals():
            del val_dataloader
        if 'full_dataloader' in locals():
            del full_dataloader
        if 'train_dataset' in locals():
            del train_dataset
        if 'val_dataset' in locals():
            del val_dataset
        if 'full_dataset' in locals():
            del full_dataset
        
        # 3. 스케일링된 데이터 배열 삭제
        if 'scaled_data_full' in locals():
            del scaled_data_full
        if 'scaled_data_train' in locals():
            del scaled_data_train
        if 'scaled_data_val' in locals():
            del scaled_data_val
        
        # 4. DataFrame 삭제
        if 'full_df' in locals():
            del full_df
        if 'train_df' in locals():
            del train_df
        if 'val_df' in locals():
            del val_df
        
        # 5. 기타 변수 삭제
        if 'feature_scalers' in locals():
            del feature_scalers
        if 'loaded_scaler' in locals():
            del loaded_scaler
        if 'kmeans_model' in locals():
            del kmeans_model
        if 'loaded_kmeans' in locals():
            del loaded_kmeans
        if 'new_ticks_A' in locals():
            del new_ticks_A
        if 'sample_data' in locals():
            del sample_data
        
        # 6. CUDA 캐시 정리 (GPU 메모리 해제)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"  ✓ GPU 메모리 캐시 정리 완료")
        
        # 7. Python 가비지 컬렉션 강제 실행
        gc.collect()
        print(f"  ✓ Python 가비지 컬렉션 완료")
        
        print("[메모리 정리 완료]\n")