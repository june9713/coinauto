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
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans  # KMeans보다 대용량 데이터에 빠름
from collections import Counter

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

# --- 2. PyTorch 데이터셋 (Price용) ---
# 다중 피처(N, 8)를 처리하도록 수정
class PricePatternDataset(Dataset):
    """
    다중 피처 시계열 데이터를 (seq_len, num_features) 텐서로 반환하는 데이터셋
    """
    def __init__(self, data, seq_len):
        # data는 (N, num_features) 형태의 스케일링된 NumPy 배열
        self.data = data
        self.seq_len = seq_len
        self.num_features = data.shape[1]
        
        if len(data) < seq_len:
            raise ValueError(f"데이터 길이({len(data)})가 시퀀스 길이({seq_len})보다 짧습니다.")

    def __len__(self):
        # 슬라이딩 윈도우 방식
        return len(self.data) - self.seq_len + 1

    def __getitem__(self, idx):
        # (seq_len, num_features) 형태의 샘플 추출
        sample = self.data[idx : idx + self.seq_len]
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
    """
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, latent_dim, max_seq_len):
        super(TransformerAutoencoder, self).__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        # 1. Input Embedding (input_dim -> d_model)
        self.input_embed = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        # 2. Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # 3. Bottleneck (d_model * seq_len -> latent_dim)
        # 시퀀스 전체를 flatten하여 압축
        self.to_latent = nn.Sequential(
            nn.Linear(d_model * max_seq_len, d_model),
            nn.ReLU(),
            nn.Linear(d_model, latent_dim)
        )
        
        # 4. Decoder Input (latent_dim -> d_model * seq_len)
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model * max_seq_len)
        )
        
        # 5. Decoder (TransformerEncoder 층을 디코더로 활용)
        # num_decoder_layers 파라미터 사용
        decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True)
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
        if self.verbose:
            print(f'  [EarlyStopping] Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model to {self.path}')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# --- 5. 모델 학습 (신규 함수) ---
# [Req 6, 7] 조기 종료를 포함한 학습 루프
def train_autoencoder(model, train_loader, val_loader, model_path, epochs, lr, patience, device):
    """
    트랜스포머 오토인코더 학습 함수
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss() # 재구축 오차(Reconstruction Loss)
    
    # [Req 6, 8] 조기 종료 핸들러 (베스트 모델을 model_path에 저장)
    early_stopper = EarlyStopping(patience=patience, verbose=True, path=model_path)
    
    print("\n" + "="*40)
    print("      🚀  오토인코더 학습 시작  🚀")
    print("="*40)
    
    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for data in train_loader:
            data = data.to(device) # (batch, seq_len, num_features)
            
            optimizer.zero_grad()
            reconstructed, _ = model(data)
            loss = criterion(reconstructed, data) # 원본(data)과 재구축(reconstructed) 비교
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
        
        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                reconstructed, _ = model(data)
                loss = criterion(reconstructed, data)
                val_loss += loss.item() * data.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        
        print(f'Epoch: {epoch:04d} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {val_loss:.6f}')
        
        # --- Early Stopping Check ---
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print("="*40)
            print("           ⛔ 조기 종료 ⛔")
            print(f"최적 Epoch: {epoch - early_stopper.patience}")
            print(f"최저 Val Loss: {early_stopper.val_loss_min:.6f}")
            print("="*40)
            break
    
    # [Req 8] 학습 완료 후, 저장된 베스트 모델을 로드
    print(f"\n학습 종료. '{model_path}'에서 최적 모델을 로드합니다.")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"오류: 최적 모델 로드 실패: {e}. 마지막 epoch 모델을 반환합니다.")
        
    return model

# --- 6. 카테고리 생성 (KMeans) (신규 함수) ---
# [Req 4, 5, 9]
def create_categories(model, dataloader, n_categories, device):
    """
    학습된 Autoencoder를 이용해 Latent Vector를 추출하고 KMeans 클러스터링 수행
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
    
    # --- [Req 9] 카테고리 편중도(분포) 출력 ---
    labels = kmeans.labels_
    label_counts = Counter(labels)
    sorted_counts = label_counts.most_common() # (label, count) 튜플의 리스트
    
    print("\n" + "="*40)
    print("    📊 Top 5 Most Frequent Categories 📊")
    print("="*40)
    total_samples = len(labels)
    for i, (label, count) in enumerate(sorted_counts[:30]):
        percentage = (count / total_samples) * 100
        print(f"  {i+1}. Category {label:03d}: {count}개 샘플 ({percentage:.2f}%)")
    print("="*40 + "\n")
    
    return kmeans

# --- 7. 카테고리 추론 (신규 함수) ---
# [Req 5] 저장된 모델/스케일러/KMeans로 새 데이터의 패턴 ID 추론
def get_pattern_category(new_data_ticks, autoencoder, kmeans_model, scaler, seq_len, device):
    """
    새로운 데이터(A틱)를 받아 어떤 패턴 카테고리에 속하는지 추론합니다.
    
    Args:
        new_data_ticks (np.array): (seq_len, num_features) 형태의 원본 데이터
    """
    
    # 입력 데이터 검증
    if new_data_ticks.shape[0] != seq_len:
        raise ValueError(f"입력 데이터 길이는 {seq_len}이어야 합니다. (현재: {new_data_ticks.shape[0]})")
    if new_data_ticks.shape[1] != scaler.n_features_in_:
        raise ValueError(f"입력 피처 개수는 {scaler.n_features_in_}개여야 합니다. (현재: {new_data_ticks.shape[1]})")

    # 1. Scale
    scaled_data = scaler.transform(new_data_ticks)
    
    # 2. Convert to Tensor (Batch 차원 추가)
    data_tensor = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 3. Get latent vector
    autoencoder.to(device)
    autoencoder.eval()
    with torch.no_grad():
        _, latent = autoencoder(data_tensor) # (reconstructed, latent)
    
    # 4. Predict category
    latent_np = latent.cpu().numpy()
    category = kmeans_model.predict(latent_np)
    
    return category[0] # [batch_size=1]이므로 첫 번째 결과 반환

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
        START_DATE = '2025-06-17'
        END_DATE = '2025-11-05'
        
        # [Req 1] 사용할 피처 리스트 (CSV 컬럼명과 일치해야 함)
        FEATURES = ['open', 'high', 'low', 'close', 'volume', 'ma5', 'ma7', 'ma10']
        INPUT_DIM = len(FEATURES) # 8
        
        # [Req 1, 3] 과거 "A"틱 (샘플 길이). 30~300 사이로 설정.
        SEQUENCE_LENGTH = 50
        MAX_SEQ_LEN = SEQUENCE_LENGTH # Positional Encoding을 위해 모델에 전달
        
        # 모델 파라미터
        D_MODEL = 64
        NHEAD = 4
        NUM_ENCODER_LAYERS = 3
        NUM_DECODER_LAYERS = 3
        LATENT_DIM = 32          # 32차원으로 패턴 압축
        
        # [Req 4] 카테고리 수
        N_CATEGORIES = 500
        
        # 학습 파라미터
        BATCH_SIZE = 4096*2
        EPOCHS = 500            # [Req 6] 조기 종료되므로 넉넉하게 설정
        VALIDATION_SPLIT_RATIO = 0.1 # 10%를 검증용으로 사용
        
        # [Req 6] 조기 종료 Patience
        EARLY_STOPPING_PATIENCE = 50
        LEARNING_RATE = 1e-4
        
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
        
        # [Req 2, 5, 8] 저장 경로
        SCALER_PATH = os.path.join(sequence_length_dir, f'{TRAIN_TYPE}_scaler.joblib')
        MODEL_PATH = os.path.join(sequence_length_dir, 'transformer_autoencoder.pth') # 베스트 모델 저장 경로
        KMEANS_PATH = os.path.join(sequence_length_dir, 'pattern_categories.joblib')

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
        
        # (2) Scaler 로드 또는 생성
        if os.path.exists(SCALER_PATH):
            print(f"\n'{SCALER_PATH}'에서 기존 Scaler를 로드합니다.")
            scaler = joblib.load(SCALER_PATH)
        else:
            print(f"\n'{SCALER_PATH}'에 Scaler가 없습니다. Train 데이터로 새로 생성합니다.")
            scaler = StandardScaler()
            # Train 데이터 기준으로만 fit
            scaler.fit(train_df[FEATURES])
            joblib.dump(scaler, SCALER_PATH)
            print(f"Scaler 저장 완료: {SCALER_PATH}")

        # (3.1) 전체 데이터를 스케일링 (KMeans 학습용)
        scaled_data_full = scaler.transform(full_df[FEATURES])
        # (3.2) Train/Val 데이터를 스케일링 (모델 학습용)
        scaled_data_train = scaler.transform(train_df[FEATURES])
        scaled_data_val = scaler.transform(val_df[FEATURES])

        print(f"데이터 분할: Train {len(scaled_data_train)}개, Validation {len(scaled_data_val)}개")

        # --- 3. Dataset 및 DataLoader (Train/Validation 분리) ---
        # [Req 6]
        train_dataset = PricePatternDataset(scaled_data_train, SEQUENCE_LENGTH)
        val_dataset = PricePatternDataset(scaled_data_val, SEQUENCE_LENGTH)

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
            max_seq_len=MAX_SEQ_LEN
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
            device=device
        )
        
        # --- 6. "카테고리" 생성 (KMeans) ---
        # [Req 4, 5, 9]
        # train_autoencoder 함수가 최적 모델을 로드하여 반환했으므로
        # autoencoder_model은 현재 '베스트 모델' 상태입니다.
        
        # K-Means는 전체 데이터(Train+Val)로 생성
        full_dataset = PricePatternDataset(scaled_data_full, SEQUENCE_LENGTH)
        # K-Means 학습 시에는 데이터를 섞을 필요 없음
        full_dataloader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        kmeans_model = create_categories(
            model=autoencoder_model, 
            dataloader=full_dataloader, 
            n_categories=N_CATEGORIES,
            device=device
        )
        
        # [Req 5] KMeans 모델 저장
        joblib.dump(kmeans_model, KMEANS_PATH)
        print(f"KMeans (카테고리) 모델 저장 완료: {KMEANS_PATH}")

        
        # --- 7. "카테고리" 재사용 예시 ---
        print("\n" + "="*40)
        print("      🔄 카테고리 재사용(추론) 테스트 🔄")
        print("="*40)
        
        # (가상의 새 데이터 100틱, 8피처)
        # [Req 1] (SEQUENCE_LENGTH, INPUT_DIM) 형태의 NumPy 배열
        new_ticks_A = np.random.rand(SEQUENCE_LENGTH, INPUT_DIM) * 100 + 150000000
        
        # 저장된 모델/스케일러 로드
        print("저장된 모델/스케일러/KMeans 로드 중...")
        loaded_autoencoder = TransformerAutoencoder(
            input_dim=INPUT_DIM, d_model=D_MODEL, nhead=NHEAD,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            num_decoder_layers=NUM_DECODER_LAYERS,
            latent_dim=LATENT_DIM, max_seq_len=MAX_SEQ_LEN
        )
        loaded_autoencoder.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        
        loaded_scaler = joblib.load(SCALER_PATH)
        loaded_kmeans = joblib.load(KMEANS_PATH)

        print("추론 시작...")
        category = get_pattern_category(
            new_data_ticks=new_ticks_A, 
            autoencoder=loaded_autoencoder, 
            kmeans_model=loaded_kmeans, 
            scaler=loaded_scaler, 
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
        if 'scaler' in locals():
            del scaler
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