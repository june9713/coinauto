import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split # <--- 이 줄 추가
import joblib
import math
import os
import warnings

# 경고 메시지 무시 (KMeans 초기화 관련 경고 등)
warnings.filterwarnings('ignore')

# --- 1. 데이터 로딩 ---
# (사용자 환경에 맞게 경로 등 수정 필요)
def load_data_from_dumps(base_dir, ticker, interval, start_date, end_date):
    """
    지정된 경로와 조건에 맞는 CSV 파일들을 읽어 하나의 DataFrame으로 합칩니다.
    
    :param base_dir: './dumps'
    :param ticker: 'BTC'
    :param interval: '3m'
    :param start_date: '2025-06-29' (pd.Timestamp 호환)
    :param end_date: '2025-06-30' (pd.Timestamp 호환)
    :return: 합쳐진 DataFrame
    """
    base_path = Path(base_dir) / ticker / interval
    all_files = []
    
    # 날짜 범위 생성
    date_range = pd.date_range(start=start_date, end=end_date)
    
    for date_obj in date_range:
        date_str = date_obj.strftime('%Y-%m-%d')
        date_path = base_path / date_str
        if date_path.exists():
            all_files.extend(date_path.glob('*.csv'))

    if not all_files:
        print(f"경로에서 파일을 찾을 수 없습니다: {base_path}")
        return pd.DataFrame()

    df_list = []
    for f in sorted(all_files): # 파일명을 정렬하여 시간 순서 보장
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            df_list.append(df)
        except Exception as e:
            print(f"파일 읽기 오류 {f}: {e}")
            
    if not df_list:
        return pd.DataFrame()

    full_df = pd.concat(df_list)
    full_df = full_df.sort_index() # 시간순으로 최종 정렬
    full_df = full_df[~full_df.index.duplicated(keep='first')] # 중복 인덱스 제거
    
    print(f"데이터 로드 완료: {len(full_df)} 행, {full_df.index.min()} 부터 {full_df.index.max()} 까지")
    return full_df


# --- 2. 데이터 정규화 (핵심) ---
def preprocess_volume(volume_series):
    """
    거래량을 정규화하고 Scaler를 반환합니다.
    """
    # 1. 로그 변환: 값의 분포를 정규분포에 가깝게 만듦 (매우 큰 값의 영향을 줄임)
    log_volume = np.log1p(volume_series.values.reshape(-1, 1))
    
    # 2. StandardScaler: 평균 0, 분산 1로 스케일링
    scaler = StandardScaler()
    scaled_volume = scaler.fit_transform(log_volume)
    
    return scaled_volume.flatten(), scaler


# --- 3. PyTorch Dataset 생성 ---
class VolumePatternDataset(Dataset):
    """
    슬라이딩 윈도우 및 제로 패딩을 적용하는 Dataset
    """
    def __init__(self, data, sequence_length_A, max_length=100):
        """
        :param data: 정규화된 1D numpy 배열
        :param sequence_length_A: 사용자가 지정한 과거 "A" 틱
        :param max_length: 모델 입력 최대 길이 (100)
        """
        self.data = data
        self.sequence_length_A = sequence_length_A
        self.max_length = max_length
        
        if sequence_length_A > max_length:
            raise ValueError("A (sequence_length_A)는 max_length(100)보다 클 수 없습니다.")

    def __len__(self):
        # 전체 데이터에서 (A틱 - 1) 만큼을 제외한 개수가 샘플 개수
        return len(self.data) - self.sequence_length_A + 1

    def __getitem__(self, idx):
        # 1. idx부터 idx + A 만큼의 시퀀스 추출
        raw_sequence = self.data[idx : idx + self.sequence_length_A]
        
        # 2. 제로 패딩
        # (max_length, 1) 크기의 0으로 채워진 텐서 생성 (Transformer 입력은 (Seq, Feature) 형태)
        padded_sequence = np.zeros((self.max_length, 1))
        
        # 뒤쪽에 패딩: padded_sequence[0 : self.sequence_length_A, 0] = raw_sequence
        # 앞쪽에 패딩 (최신 데이터가 뒤로 가도록):
        padding_size = self.max_length - self.sequence_length_A
        padded_sequence[padding_size:, 0] = raw_sequence
        
        # 3. 텐서로 변환
        return torch.FloatTensor(padded_sequence)


# --- 4. 모델 설계 (Transformer Autoencoder) ---

class PositionalEncoding(nn.Module):
    """ Transformer의 Positional Encoding """
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=256, latent_dim=32, dropout=0.1, max_seq_len=100):
        super(TransformerAutoencoder, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # --- Encoder ---
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # (Batch, Seq, d_model) -> (Batch, latent_dim)
        # 시퀀스 전체의 평균을 내어 하나의 벡터로 압축
        self.encoder_output_linear = nn.Linear(d_model, latent_dim)

        # --- Decoder ---
        self.latent_to_decoder_input = nn.Linear(latent_dim, max_seq_len * d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.output_linear = nn.Linear(d_model, input_dim)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src):
        # src: (Batch, Seq_len=100, 1)
        
        # 1. (Batch, 100, 1) -> (Batch, 100, d_model)
        src_emb = self.input_linear(src) * math.sqrt(self.d_model)
        
        # 2. Positional Encoding
        # (Batch, 100, d_model) -> (100, Batch, d_model) (PyTorch Transformer 기본)
        # batch_first=True로 설정했으므로 (Batch, 100, d_model) 유지
        src_pos = self.pos_encoder(src_emb.transpose(0, 1)).transpose(0, 1)
        
        # 3. Transformer Encoder
        # (Batch, 100, d_model)
        encoder_output = self.transformer_encoder(src_pos)
        
        # 4. Global Average Pooling: (Batch, 100, d_model) -> (Batch, d_model)
        # 시퀀스 전체의 문맥을 하나의 벡터로 요약
        pooled_output = encoder_output.mean(dim=1) 
        
        # 5. (Batch, d_model) -> (Batch, latent_dim)
        latent_vector = self.encoder_output_linear(pooled_output)
        return latent_vector

    def decode(self, latent_vector):
        # latent_vector: (Batch, latent_dim)
        
        # 1. (Batch, latent_dim) -> (Batch, 100 * d_model) -> (Batch, 100, d_model)
        decoder_input = self.latent_to_decoder_input(latent_vector)
        decoder_input = decoder_input.view(-1, self.max_seq_len, self.d_model)

        # Decoder는 'memory' (Encoder의 출력)가 필요하지만,
        # 여기서는 Autoencoder이므로 latent_vector에서 복원된 값을 memory로 사용
        # 혹은 더 간단하게 MLP Decoder를 사용할 수도 있습니다.
        # 여기서는 TransformerDecoder를 사용하되, latent에서 복원된 값을 'memory'이자 'tgt'로 사용합니다.
        
        # (Batch, 100, d_model) -> (100, Batch, d_model)
        decoder_input_pos = self.pos_encoder(decoder_input.transpose(0, 1)).transpose(0, 1)

        # Autoencoder에서는 memory와 tgt가 동일할 수 있음 (latent로부터 복원된 정보)
        decoder_output = self.transformer_decoder(tgt=decoder_input_pos, memory=decoder_input_pos)
        
        # (Batch, 100, d_model) -> (Batch, 100, 1)
        output = self.output_linear(decoder_output)
        return output

    def forward(self, src):
        latent_vector = self.encode(src)
        reconstructed_output = self.decode(latent_vector)
        return reconstructed_output


# --- 5. 1단계: Autoencoder 학습 ---
def train_autoencoder(model, train_loader, val_loader, model_path, epochs=100, lr=1e-4, patience=10):
    """
    Autoencoder 모델을 학습하고, 조기 종료 및 최적 모델 저장을 수행합니다.
    
    :param model: TransformerAutoencoder 객체
    :param train_loader: 학습용 DataLoader
    :param val_loader: 검증용 DataLoader
    :param model_path: 최적 모델을 저장할 경로
    :param epochs: 최대 학습 Epochs
    :param lr: 학습률
    :param patience: 조기 종료를 위한 대기 Epochs 수
    :return: 학습이 완료된 모델 객체 (가장 마지막 Epoch 기준)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Autoencoder 학습 시작 (Device: {device})")
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    last_val_loss = float('inf') # 수렴 척도(Delta) 계산용

    for epoch in range(epochs):
        # --- 1. 학습(Training) 단계 ---
        model.train()
        total_train_loss = 0
        for data in train_loader:
            data = data.to(device) # (Batch, 100, 1)
            
            # 순전파
            reconstructed = model(data)
            loss = criterion(reconstructed, data)
            
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)

        # --- 2. 검증(Validation) 단계 ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                reconstructed = model(data)
                loss = criterion(reconstructed, data)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        # --- 3. 수렴 척도 계산 및 출력 ---
        loss_delta = last_val_loss - avg_val_loss
        last_val_loss = avg_val_loss
        
        status_sign = " " # 개선 여부 표시
        
        # --- 4. 조기 종료 및 최적 모델 저장 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # 최적 모델 저장
            torch.save(model.state_dict(), model_path)
            status_sign = "★" # 개선됨
        else:
            epochs_no_improve += 1
            status_sign = " "

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f} | "
              f"Delta: {loss_delta:+.6f} | " # <--- 수렴 척도
              f"Patience: {epochs_no_improve}/{patience} {status_sign}")

        if epochs_no_improve >= patience:
            print(f"\n{patience} Epochs 동안 Val Loss가 개선되지 않았습니다. 조기 종료합니다.")
            print(f"최적 모델이 '{model_path}'에 저장되었습니다 (Val Loss: {best_val_loss:.6f}).")
            break

    print("Autoencoder 학습 완료.")
    # (참고: 반환되는 모델은 마지막 Epoch 모델이지만, 
    #  파일에는 최적 모델(best_val_loss)이 저장되어 있음)
    return model


# --- 6. 2단계: 패턴 "카테고리화" (Clustering) ---
def create_categories(model, dataloader, n_categories=50):
    """
    Autoencoder 모델을 사용하여 잠재 벡터를 추출하고 KMeans 클러스터링을 수행합니다.
    :param model: 학습된 TransformerAutoencoder 모델 객체
    :param dataloader: 전체 데이터셋의 DataLoader (shuffle=False 권장)
    :param n_categories: 분류할 카테고리(클러스터)의 수
    :return: 학습된 KMeans 모델
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("패턴 카테고리화(Clustering) 시작...")
    
    model.to(device)
    model.eval() # 모델을 평가 모드로 전환 (Dropout 등 비활성화)
    
    all_latent_vectors = []
    with torch.no_grad(): # 추론 시에는 그래디언트 계산 비활성화
        for data in dataloader:
            data = data.to(device)
            # model.encode() 메서드를 직접 호출
            latent_vectors = model.encode(data) # (Batch, latent_dim)
            all_latent_vectors.append(latent_vectors.cpu().numpy())
            
    all_latent_vectors = np.concatenate(all_latent_vectors, axis=0)
    print(f"총 {all_latent_vectors.shape[0]}개의 잠재 벡터 추출 완료.")

    # MiniBatchKMeans: 대용량 데이터에 적합
    kmeans = MiniBatchKMeans(n_clusters=n_categories, 
                             random_state=42, 
                             batch_size=256, 
                             n_init=10)
    kmeans.fit(all_latent_vectors)
    
    print("KMeans 클러스터링 완료.")
    
    # 모델을 다시 학습 모드로 돌려놓을 필요가 있다면 model.train()을 호출
    # (여기서는 이 함수 이후 모델을 사용하지 않으므로 생략)
    return kmeans


# --- 7. "카테고리" 재사용 (추론) ---
# --- 7. "카테고리" 재사용 (추론) ---
def get_pattern_category(new_data_point, model, kmeans, scaler, A, max_length=100):
    """
    새로운 데이터 조각(A틱)의 카테고리를 반환합니다.
    
    :param new_data_point: (A,) 크기의 1D numpy 배열 (정규화되지 않은 원본 거래량)
    :param model: 학습된 Autoencoder 모델 객체
    :param kmeans: 학습된 KMeans 모델
    :param scaler: 학습에 사용된 StandardScaler
    :param A: 시퀀스 길이
    :return: 카테고리 ID (int)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() # 모델을 평가 모드로 전환

    # 1. 정규화 (학습 때 사용한 Scaler 사용)
    log_data = np.log1p(new_data_point.reshape(-1, 1))
    scaled_data = scaler.transform(log_data).flatten()
    
    # 2. 패딩
    padded_sequence = np.zeros((max_length, 1))
    padding_size = max_length - A
    padded_sequence[padding_size:, 0] = scaled_data
    
    # 3. 텐서 변환
    input_tensor = torch.FloatTensor(padded_sequence).unsqueeze(0).to(device) # (1, 100, 1)
    
    # 4. 잠재 벡터 추출
    with torch.no_grad():
        latent_vector = model.encode(input_tensor) # (1, latent_dim)
    
    # 5. 카테고리 예측
    category_id = kmeans.predict(latent_vector.cpu().numpy())
    
    return category_id[0]


# --- 전체 실행 예시 ---
if __name__ == '__main__':
    
    # --- CUDA/CPU 장치 검증 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*40)
    print("      장치 검증 (Device Verification)      ")
    print("="*40)
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 사용 가능 여부: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"사용 중인 GPU: {torch.cuda.get_device_name(0)}")
        print(f"선택된 장치: {device}")
    else:
        print("!! 경고: CUDA를 사용할 수 없습니다.")
        print("!! CPU로 연산을 수행합니다.")
        print(f"선택된 장치: {device}")
    print("="*40)
    # --- (검증 코드 끝) ---

    
    # --- 0. 하이퍼파라미터 설정 ---
    BASE_DIR = './dumps'
    TICKER = 'BTC'
    INTERVAL = '3m'
    START_DATE = '2024-09-17'
    END_DATE = '2025-11-05' # 예시 (실제로는 더 긴 기간 필요)
    
    SEQUENCE_LENGTH_A = 100  # 과거 "A"틱 (100 이하)
    MAX_SEQ_LEN = 100
    
    # 모델 파라미터
    D_MODEL = 64
    NHEAD = 4
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    LATENT_DIM = 32         # 32차원으로 패턴 압축
    N_CATEGORIES = 50       # 50개의 패턴 카테고리로 분류
    
    # 학습 파라미터
    BATCH_SIZE = 64
    EPOCHS = 9999 # (조기 종료되므로 넉넉하게 설정)
    VALIDATION_SPLIT_RATIO = 0.1 # 10%를 검증용으로 사용
    EARLY_STOPPING_PATIENCE = 300 # 10 Epochs 개선 없으면 중지
    LEARNING_RATE = 1e-4
    
    # 저장 파일명
    MODELS_DIR = './models'
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    ticker_dir = os.path.join(MODELS_DIR, TICKER)
    if not os.path.exists(ticker_dir):
        os.makedirs(ticker_dir)
    interval_dir = os.path.join(ticker_dir, INTERVAL)
    if not os.path.exists(interval_dir):
        os.makedirs(interval_dir)
    SCALER_PATH = os.path.join(interval_dir, 'volume_scaler.joblib')
    MODEL_PATH = os.path.join(interval_dir, 'transformer_autoencoder.pth')
    KMEANS_PATH = os.path.join(interval_dir, 'pattern_categories.joblib')

    # --- 1. & 2. 데이터 로드 및 전처리 ---
    full_df = load_data_from_dumps(BASE_DIR, TICKER, INTERVAL, START_DATE, END_DATE)
    if full_df.empty:
        print("데이터가 없습니다. 가상 데이터로 대체합니다.")
        print("가상 데이터를 생성합니다 (5000 틱).")
        volume_data_raw = pd.Series(np.abs(np.random.randn(5000) * 100 + np.sin(np.arange(5000) / 50) * 50 + 50))
    else:
        volume_data_raw = full_df['volume']
        
    # Scaler는 전체 데이터 기준으로 fit (일관된 스케일링)
    scaled_volume, scaler = preprocess_volume(volume_data_raw)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler 저장 완료: {SCALER_PATH}")

    # --- 3. Dataset 및 DataLoader (Train/Validation 분리) ---
    
    # 시계열 데이터이므로 shuffle=False로 순서 유지하며 분리
    train_data, val_data = train_test_split(scaled_volume, 
                                            test_size=VALIDATION_SPLIT_RATIO, 
                                            shuffle=False)
    
    print(f"데이터 분할: Train {len(train_data)}개, Validation {len(val_data)}개")

    train_dataset = VolumePatternDataset(train_data, SEQUENCE_LENGTH_A, MAX_SEQ_LEN)
    val_dataset = VolumePatternDataset(val_data, SEQUENCE_LENGTH_A, MAX_SEQ_LEN)

    # 학습 데이터는 섞어서 사용
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # 검증 데이터는 순서대로 사용
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # (데이터 형태 확인)
    sample_data = next(iter(train_dataloader))
    print(f"데이터셋 샘플 형태: {sample_data.shape}")

    # --- 4. Autoencoder 모델 초기화 ---
    autoencoder_model = TransformerAutoencoder(
        input_dim=1, d_model=D_MODEL, nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        latent_dim=LATENT_DIM, max_seq_len=MAX_SEQ_LEN
    )

    # --- 모델 로드 (학습 재개) ---
    if os.path.exists(MODEL_PATH):
        print(f"\n'{MODEL_PATH}' 에서 기존 모델을 찾았습니다.")
        try:
            autoencoder_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print("모델 가중치 로드 완료. 학습을 재개합니다.\n")
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {e}")
            print("새로 학습을 시작합니다.\n")
    else:
        print(f"\n'{MODEL_PATH}' 에서 기존 모델을 찾을 수 없습니다.")
        print("새로 학습을 시작합니다.\n")


    # --- 5. Autoencoder 모델 학습 (조기 종료 적용) ---
    autoencoder_model = train_autoencoder(
        model=autoencoder_model, 
        train_loader=train_dataloader, 
        val_loader=val_dataloader, 
        model_path=MODEL_PATH, 
        epochs=EPOCHS, 
        lr=LEARNING_RATE, 
        patience=EARLY_STOPPING_PATIENCE
    )
    
    # --- 6. "카테고리" 생성 (KMeans) ---
    # 학습이 완료되었으므로, 파일에 저장된 "최적" 모델을 다시 로드
    print(f"\n학습 완료. '{MODEL_PATH}'에서 최적 모델을 로드하여 클러스터링을 진행합니다.")
    try:
        autoencoder_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except Exception as e:
        print(f"최적 모델 로드 실패: {e}. 마지막 Epoch 모델로 클러스터링합니다.")
        # (이 경우 autoencoder_model은 train_autoencoder가 반환한 마지막 모델)

    # K-Means는 전체 데이터(Train+Val)로 생성
    full_dataset = VolumePatternDataset(scaled_volume, SEQUENCE_LENGTH_A, MAX_SEQ_LEN)
    full_dataloader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    kmeans_model = create_categories(autoencoder_model, full_dataloader, n_categories=N_CATEGORIES)
    
    joblib.dump(kmeans_model, KMEANS_PATH)
    print(f"KMeans (카테고리) 모델 저장 완료: {KMEANS_PATH}")

    
    # --- 7. "카테고리" 재사용 예시 ---
    print("\n--- 카테고리 재사용 테스트 ---")
    
    # (가상의 새 데이터 60틱)
    new_ticks_A = np.abs(np.random.randn(SEQUENCE_LENGTH_A) * 150 + 30)
    
    # 저장된 모델/스케일러 로드
    loaded_autoencoder = TransformerAutoencoder(
        input_dim=1, d_model=D_MODEL, nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        latent_dim=LATENT_DIM, max_seq_len=MAX_SEQ_LEN
    )
    
    loaded_autoencoder.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    loaded_scaler = joblib.load(SCALER_PATH)
    loaded_kmeans = joblib.load(KMEANS_PATH)

    category = get_pattern_category(
        new_ticks_A, 
        loaded_autoencoder, 
        loaded_kmeans, 
        loaded_scaler, 
        SEQUENCE_LENGTH_A
    )
    
    print(f"새로운 {SEQUENCE_LENGTH_A}틱 데이터의 패턴 카테고리: {category}")