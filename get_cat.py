import os
import glob
import math
import sys
import traceback
import warnings
from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

# Sklearn 경고 무시
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# --- 모델 클래스 정의 (volume_opt.py와 동일) ---
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
    - 점진적 압축/확장으로 정보 손실 최소화
    - Dropout 추가로 과적합 방지
    """
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, latent_dim, max_seq_len, dropout=0.1):
        super(TransformerAutoencoder, self).__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model

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
        # 점진적 압축으로 정보 손실 최소화
        self.to_latent = nn.Sequential(
            nn.Linear(d_model * max_seq_len, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, latent_dim)
        )

        # 4. Decoder Input (latent_dim -> d_model * seq_len)
        # 점진적 확장으로 복원 품질 향상
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, d_model * max_seq_len)
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

# --- 카테고리 추론 함수 (volume_opt.py와 동일) ---
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

# --- 전체 CSV 파일들을 연속된 데이터로 처리하는 함수 ---
def process_all_csv_files(csv_files, base_dir, output_base_dir, autoencoder, kmeans_model, feature_scalers, features, seq_len, device, start_date=None):
    """
    모든 CSV 파일들을 시간순으로 연결하여 하나의 연속된 데이터로 처리합니다.
    
    Args:
        csv_files (list): 모든 CSV 파일 경로 리스트
        base_dir (str): 입력 기본 디렉토리
        output_base_dir (str): 출력 기본 디렉토리
        autoencoder: 학습된 오토인코더 모델
        kmeans_model: 학습된 KMeans 모델
        feature_scalers: Feature별 독립 스케일러 딕셔너리
        features (list): 사용할 피처 컬럼명 리스트
        seq_len (int): 시퀀스 길이
        device: PyTorch 장치
        start_date (str or None): 시작 날짜 (YYYY-MM-DD 형식). 이 날짜 이후의 데이터만 처리. None이면 모든 데이터 처리.
    
    Returns:
        tuple: (성공한 파일 수, 실패한 파일 수)
    """
    if not csv_files:
        return 0, 0
    
    success_count = 0
    fail_count = 0
    
    try:
        # 1. 모든 CSV 파일을 시간순으로 읽어서 병합
        all_dfs = []
        file_to_df_map = {}  # 각 파일 경로와 해당 DataFrame을 매핑
        
        # START_DATE 파싱
        start_date_dt = None
        if start_date:
            try:
                start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
                print(f"시작 날짜 필터: {start_date} 이후의 데이터만 처리합니다.")
            except ValueError:
                print(f"경고: 시작 날짜 형식이 잘못되었습니다: {start_date}. 모든 데이터를 처리합니다.")
        
        print(f"전체 {len(csv_files)}개 CSV 파일 읽기 중...")
        for csv_path in sorted(csv_files):
            try:
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                
                # START_DATE 이후의 데이터만 필터링
                if start_date_dt is not None:
                    df = df[df.index >= start_date_dt]
                    if len(df) == 0:
                        # 이 파일에는 START_DATE 이후의 데이터가 없음
                        continue
                
                # 필요한 피처 컬럼 확인
                missing_features = [f for f in features if f not in df.columns]
                if missing_features:
                    print(f"  경고: {os.path.basename(csv_path)}에 다음 피처 컬럼이 없습니다: {missing_features}. 건너뜁니다.")
                    fail_count += 1
                    continue
                
                # cat01 컬럼 초기화
                df['cat01'] = np.nan
                
                all_dfs.append(df)
                file_to_df_map[csv_path] = df
                
            except Exception as e:
                print(f"  경고: {os.path.basename(csv_path)} 읽기 실패: {e}. 건너뜁니다.")
                fail_count += 1
                continue
        
        if not all_dfs:
            print("경고: START_DATE 이후의 데이터가 없습니다.")
            return 0, fail_count
        
        # 2. 모든 DataFrame을 시간순으로 병합 (전체 데이터를 하나의 연속 데이터로)
        
        merged_df = pd.concat(all_dfs)
        merged_df = merged_df.sort_index()  # 시간순 정렬
        merged_df = merged_df[~merged_df.index.duplicated(keep='first')]  # 중복 제거
        
        # START_DATE 이후의 데이터만 다시 확인 (병합 후에도 필터링)
        if start_date_dt is not None:
            merged_df = merged_df[merged_df.index >= start_date_dt]
        
        # 3. 피처 컬럼만 선택하여 유효한 데이터 추출
        merged_features = merged_df[features].copy()
        valid_mask = ~merged_features.isna().any(axis=1)
        merged_features_clean = merged_features[valid_mask]
        
        if len(merged_features_clean) < seq_len:
            print(f"경고: 전체 유효 데이터가 {seq_len}틱 미만입니다. (현재: {len(merged_features_clean)}틱)")
            return 0, fail_count + len(csv_files)
        
        # 4. 슬라이딩 윈도우로 카테고리 예측
        # 처음 (seq_len-1)개 틱은 예측 불가이므로 NaN으로 유지
        # seq_len번째 틱부터 예측 가능
        categories_dict = {}  # {인덱스: 카테고리} 딕셔너리
        total_ticks = len(merged_features_clean)
        clean_indices = merged_features_clean.index
        predictable_ticks = total_ticks - (seq_len - 1)  # 예측 가능한 틱 수
        
        print(f"전체 데이터 처리 중: {len(csv_files)}개 파일, 총 {total_ticks}틱 (처음 {seq_len-1}틱은 예측 불가)")
        print(f"예측 가능한 틱 수: {predictable_ticks}틱")
        print("-" * 60)
        
        # 진행률 표시 주기 (1% 또는 100틱마다)
        progress_interval = max(1, predictable_ticks // 100)  # 1%마다 표시
        
        for i in range(seq_len - 1, total_ticks):
            # 슬라이딩 윈도우: [i-seq_len+1 : i+1]
            window_data = merged_features_clean.iloc[i-seq_len+1:i+1].values
            
            # 현재 처리 중인 틱의 날짜 정보
            current_idx = clean_indices[i]
            if isinstance(current_idx, pd.Timestamp):
                current_date = current_idx.strftime('%Y-%m-%d')
                current_time = current_idx.strftime('%H:%M:%S')
            else:
                current_date = str(current_idx)[:10] if len(str(current_idx)) >= 10 else str(current_idx)
                current_time = ""
            
            # 진행률 계산
            progress_num = i - (seq_len - 1) + 1  # 예측 가능한 틱 중 현재 위치
            progress_percent = (progress_num / predictable_ticks) * 100
            
            # 진행률 표시 (주기적으로 또는 마지막 틱)
            if progress_num % progress_interval == 0 or progress_num == predictable_ticks:
                if current_time:
                    print(f"진행률: {progress_num:,}/{predictable_ticks:,} ({progress_percent:.1f}%) - 날짜: {current_date} {current_time}")
                else:
                    print(f"진행률: {progress_num:,}/{predictable_ticks:,} ({progress_percent:.1f}%) - 날짜: {current_date}")
            
            try:
                category = get_pattern_category(
                    new_data_ticks=window_data,
                    autoencoder=autoencoder,
                    kmeans_model=kmeans_model,
                    feature_scalers=feature_scalers,
                    seq_len=seq_len,
                    device=device
                )
                # 예측된 카테고리를 딕셔너리에 저장
                idx = clean_indices[i]
                categories_dict[idx] = category
            except Exception as e:
                print(f"  경고: {i}번째 틱 ({current_date}) 예측 실패: {e}")
                continue
        
        # 5. 각 원본 CSV 파일에 cat01 값 할당 및 저장
        print(f"\n각 CSV 파일에 cat01 값 할당 중...")
        for csv_path, df in file_to_df_map.items():
            # 해당 파일의 인덱스에 대해 cat01 값 할당
            for idx in df.index:
                if idx in categories_dict:
                    df.loc[idx, 'cat01'] = categories_dict[idx]
                # 처음 (seq_len-1)개 틱은 예측 불가이므로 이미 NaN으로 초기화되어 있음
            
            # 출력 경로 생성
            # csv_path: ./dumps/BTC/3m/2024-11-05/file.csv
            # output_base_dir: ./dumps2
            # 결과: ./dumps2/BTC/3m/2024-11-05/file.csv
            # base_dir 이후의 경로만 추출
            if csv_path.startswith(base_dir):
                rel_path = csv_path[len(base_dir):].lstrip(os.sep).lstrip('/')
                output_path = os.path.join(output_base_dir, rel_path)
            else:
                # 절대 경로인 경우 처리
                parts = csv_path.replace('\\', '/').split('/')
                try:
                    dumps_idx = parts.index('dumps')
                    if dumps_idx + 1 < len(parts):
                        rel_parts = parts[dumps_idx + 1:]
                        output_path = os.path.join(output_base_dir, *rel_parts)
                    else:
                        print(f"  경고: {csv_path} 경로 파싱 실패")
                        continue
                except ValueError:
                    print(f"  경고: {csv_path}에서 'dumps' 폴더를 찾을 수 없습니다")
                    continue
            
            # 출력 디렉토리 생성
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # CSV 파일 저장
            df.to_csv(output_path, index=True)
            
            predicted_in_file = df['cat01'].notna().sum()
            print(f"  ✓ {os.path.basename(csv_path)}: {len(df)}틱 중 {predicted_in_file}틱 예측 완료")
            success_count += 1
        
        return success_count, fail_count
        
    except Exception as e:
        err = traceback.format_exc()
        print(f"오류: 전체 데이터 처리 실패")
        print(f"err: {err}")
        return 0, len(csv_files)

# --- 메인 함수 ---
def main():
    # 하이퍼파라미터 설정 (volume_opt.py와 동일하게 맞춤)
    BASE_DIR = './dumps'
    OUTPUT_DIR = './dumps2'
    TICKER = 'BTC'
    INTERVAL = '3m'
    TRAIN_TYPE = 'price'
    SEQUENCE_LENGTH = 50
    N_CATEGORIES = 100  # volume_opt.py와 동일
    START_DATE = '2025-10-05'  # YYYY-MM-DD 형식으로 수정

    # volume_opt.py와 동일한 FEATURES
    FEATURES = ['open', 'high', 'low', 'close']
    INPUT_DIM = len(FEATURES)  # 4
    MAX_SEQ_LEN = SEQUENCE_LENGTH

    # 모델 파라미터 (volume_opt.py와 동일)
    D_MODEL = 128
    NHEAD = 8
    NUM_ENCODER_LAYERS = 4
    NUM_DECODER_LAYERS = 4
    LATENT_DIM = 256
    
    # 모델 파일 경로
    MODELS_DIR = './models'
    sequence_length_dir = os.path.join(MODELS_DIR, TICKER, INTERVAL, TRAIN_TYPE, str(SEQUENCE_LENGTH) ,str(N_CATEGORIES))
    
    SCALER_PATH = os.path.join(sequence_length_dir, f'{TRAIN_TYPE}_scaler.joblib')
    MODEL_PATH = os.path.join(sequence_length_dir, 'transformer_autoencoder.pth')
    KMEANS_PATH = os.path.join(sequence_length_dir, 'pattern_categories.joblib')
    
    # 장치 설정 (GPU 우선 사용)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("="*60)
    print("       패턴 카테고리 예측 및 저장 프로그램")
    print("="*60)
    print(f"입력 디렉토리: {BASE_DIR}")
    print(f"출력 디렉토리: {OUTPUT_DIR}")
    print(f"티커: {TICKER}")
    print(f"인터벌: {INTERVAL}")
    print(f"시퀀스 길이: {SEQUENCE_LENGTH}")
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"사용 중인 GPU: {torch.cuda.get_device_name(0)}")
    print(f"선택된 장치: {device}")
    print("="*60)
    
    # 1. 모델 파일 존재 확인
    if not os.path.exists(MODEL_PATH):
        print(f"오류: 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        sys.exit(1)
    if not os.path.exists(SCALER_PATH):
        print(f"오류: 스케일러 파일을 찾을 수 없습니다: {SCALER_PATH}")
        sys.exit(1)
    if not os.path.exists(KMEANS_PATH):
        print(f"오류: KMeans 파일을 찾을 수 없습니다: {KMEANS_PATH}")
        sys.exit(1)
    
    # 2. 모델, 스케일러, KMeans 로드
    print("\n모델 및 전처리 도구 로드 중...")
    try:
        # 모델 초기화
        autoencoder = TransformerAutoencoder(
            input_dim=INPUT_DIM,
            d_model=D_MODEL,
            nhead=NHEAD,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            num_decoder_layers=NUM_DECODER_LAYERS,
            latent_dim=LATENT_DIM,
            max_seq_len=MAX_SEQ_LEN
        )
        autoencoder.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"  ✓ 모델 로드 완료")
        
        # 스케일러 로드
        scaler = joblib.load(SCALER_PATH)
        print(f"  ✓ 스케일러 로드 완료")
        
        # KMeans 로드
        kmeans_model = joblib.load(KMEANS_PATH)
        print(f"  ✓ KMeans 모델 로드 완료")
        
    except Exception as e:
        err = traceback.format_exc()
        print("err", err)
        raise
    
    # 3. dumps 폴더에서 모든 CSV 파일 찾기
    print(f"\n{BASE_DIR}에서 CSV 파일 검색 중...")
    search_pattern = os.path.join(BASE_DIR, TICKER, INTERVAL, '**', '*.csv')
    csv_files = sorted(glob.glob(search_pattern, recursive=True))
    
    if not csv_files:
        print(f"경고: {BASE_DIR}/{TICKER}/{INTERVAL}/ 에서 CSV 파일을 찾을 수 없습니다.")
        sys.exit(1)
    
    print(f"총 {len(csv_files)}개의 CSV 파일을 찾았습니다.")
    
    # 4. 전체 CSV 파일을 하나의 연속 데이터로 처리
    print("\n전체 CSV 파일을 연속 데이터로 처리 시작...")
    if START_DATE:
        print(f"시작 날짜: {START_DATE} 이후의 데이터만 처리합니다.")
    total_success, total_fail = process_all_csv_files(
        csv_files=csv_files,
        base_dir=BASE_DIR,
        output_base_dir=OUTPUT_DIR,
        autoencoder=autoencoder,
        kmeans_model=kmeans_model,
        feature_scalers=scaler,  # feature_scalers로 전달 (dict 타입)
        features=FEATURES,
        seq_len=SEQUENCE_LENGTH,
        device=device,
        start_date=START_DATE
    )
    
    # 6. 결과 출력
    print("\n" + "="*60)
    print("                    처리 완료")
    print("="*60)
    print(f"총 파일 수: {len(csv_files)}개")
    print(f"성공: {total_success}개")
    print(f"실패: {total_fail}개")
    print(f"출력 디렉토리: {OUTPUT_DIR}")
    print("="*60)

if __name__ == '__main__':
    try:
        main()
        sys.exit(0)
    except Exception as e:
        err = traceback.format_exc()
        print("err", err)
        sys.exit(1)
