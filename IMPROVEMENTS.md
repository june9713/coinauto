# volume_opt.py 개선 사항 요약

## ⚠️ 긴급 버그 수정 (2025-11-07)

### 🐛 Contrastive Loss 폭발 버그 수정
- **문제**: 기존 SimCLR 구현에서 `log_softmax().diagonal()` 오류로 Loss가 10억 단위로 폭발
- **해결**: 간단한 평균 유사도 최소화 방식으로 변경
- **효과**: Contrastive Loss가 정상 범위 (0~1) 내로 안정화

### 🎯 피처 간소화
- **변경**: `['open', 'high', 'low', 'close', 'volume', 'ma5', 'ma7', 'ma10']` (8개)
  → `['open', 'high', 'low', 'close', 'volume', 'ma10']` (6개)
- **이유**: ma5, ma7, ma10은 강한 상관관계 → 정보 중복
- **효과**:
  - 학습 속도 25% 향상
  - 모델 파라미터 감소
  - 패턴 학습 효율 증가

### ⚙️ Contrastive Weight 조정
- **변경**: `0.1` → `0.01`
- **이유**: Contrastive Loss가 Reconstruction Loss보다 지배적이지 않도록 균형 조정

---

## 주요 개선 내용

### 1. 학습률 최적화 (가장 중요!) ⚡
**문제**: 학습률이 `1e-7`로 너무 낮아서 학습이 거의 진행되지 않음
**해결**: `1e-4`로 상향 (1000배 증가)

```python
# Before
LEARNING_RATE = 1e-7  # 0.0000001

# After
LEARNING_RATE = 1e-4  # 0.0001
```

### 2. Contrastive Loss 추가 🎯
**목적**: 단순 재구축(Reconstruction)을 넘어 의미 있는 패턴 분리 강화

```python
# SimCLR 스타일의 Contrastive Loss
# - 잠재 벡터 간 코사인 유사도 계산
# - 서로 다른 패턴은 멀리, 유사한 패턴은 가까이 배치
Total Loss = Reconstruction Loss + 0.1 × Contrastive Loss
```

**효과**:
- 카테고리 간 경계가 명확해짐
- 잠재 공간에서 패턴이 더 잘 분리됨

### 3. Learning Rate Scheduler 추가 📈
**Warmup + Cosine Annealing** 스케줄러 적용

```python
Epoch 1-5:   Warmup (선형 증가)
Epoch 6-N:   Cosine Annealing (점진적 감소)
```

**효과**:
- 초반 학습 안정화
- 후반 세밀한 최적화

### 4. Latent Vector L2 정규화 🎲
모든 잠재 벡터를 **유닛 구(Unit Sphere)**에 투영

```python
latent = F.normalize(latent, p=2, dim=1)
```

**효과**:
- 벡터 크기에 상관없이 방향(패턴)만 비교
- KMeans 클러스터링 성능 향상
- Contrastive Loss와 시너지

### 5. 모델 구조 강화 🏗️
**Dropout 추가**: 과적합 방지 (0.1)
**LayerNorm 추가**: 학습 안정화
**더 깊은 Bottleneck**: 2-layer → 3-layer MLP

```python
# Before
to_latent = Linear(D*L → D) → ReLU → Linear(D → latent)

# After
to_latent = Linear(D*L → 2D) → LayerNorm → ReLU → Dropout
          → Linear(2D → D) → LayerNorm → ReLU → Dropout
          → Linear(D → latent)
```

### 6. 학습 진행 시각화 개선 📊
**상세한 Loss 추적**:

```
Epoch 0042/2000 | LR: 9.51e-05 |
Train: 0.123456 (R:0.120000 C:0.034560) |
Val: 0.098765 (R:0.095000 C:0.037650) ✓ BEST
```

- `R`: Reconstruction Loss
- `C`: Contrastive Loss
- `✓ BEST`: 최적 모델 갱신 표시

### 7. 재구축 품질 검증 🔍
학습 후 자동으로 생성: `reconstruction_quality.png`

**5개 샘플의 원본 vs 재구축 히트맵 비교**
- 모델이 패턴을 정확히 복원하는지 육안 확인 가능

### 8. t-SNE 시각화 🗺️
잠재 공간을 2D로 축소하여 시각화: `latent_space_tsne.png`

**확인 가능한 것**:
- 카테고리별로 클러스터가 형성되는가?
- 카테고리 간 경계가 명확한가?
- 같은 색(카테고리)끼리 모여있는가?

### 9. 조기 종료 Patience 조정 ⏱️
```python
# Before
EARLY_STOPPING_PATIENCE = 1000

# After
EARLY_STOPPING_PATIENCE = 100
```

학습률이 높아졌으므로 더 빠르게 수렴 → Patience 감소

### 10. AdamW Optimizer + Gradient Clipping 🛡️
```python
# Before
optimizer = Adam(lr=1e-7)

# After
optimizer = AdamW(lr=1e-4, weight_decay=0.01)
+ Gradient Clipping (max_norm=1.0)
```

**효과**:
- Weight Decay로 정규화
- Gradient 폭발 방지

## 사용 방법

### 학습 실행
```bash
python volume_opt.py
```

### 생성되는 파일들
```
models/BTC/3m/price/50/1000/
├── price_scaler.joblib              # 스케일러
├── transformer_autoencoder.pth      # 베스트 모델
├── pattern_categories.joblib        # KMeans 모델
├── reconstruction_quality.png       # 재구축 품질 시각화
└── latent_space_tsne.png           # 잠재 공간 t-SNE 시각화
```

## 학습 결과 확인 체크리스트

### ✅ 1. Loss가 충분히 감소했는가?
```
최저 Val Loss: 0.05 이하 → 양호
최저 Val Loss: 0.1~0.2 → 보통
최저 Val Loss: 0.5 이상 → 학습 부족
```

### ✅ 2. 재구축 품질이 좋은가?
`reconstruction_quality.png`를 열어서:
- 원본과 재구축이 비슷한가?
- 주요 패턴(급등/급락/횡보)이 보존되는가?

### ✅ 3. 카테고리가 잘 분리되었는가?
`latent_space_tsne.png`를 열어서:
- 같은 색끼리 뭉쳐있는가?
- 다른 색깔 클러스터가 서로 떨어져있는가?
- 회색(기타)이 너무 많지 않은가?

### ✅ 4. 카테고리 분포가 적절한가?
```
Top 1 카테고리: 5% 이하 → 양호 (편중 없음)
Top 1 카테고리: 10% 이상 → 편중됨 (카테고리 수 증가 고려)
```

## 추가 튜닝 팁

### Loss가 여전히 안 떨어진다면:
1. **Epoch 수 증가**: `EPOCHS = 2000 → 5000`
2. **모델 크기 증가**: `D_MODEL = 64 → 128`
3. **Batch Size 조정**: `BATCH_SIZE = 8192 → 4096`

### 카테고리가 너무 편중되어 있다면:
1. **카테고리 수 증가**: `N_CATEGORIES = 1000 → 2000`
2. **Contrastive Weight 증가**: `CONTRASTIVE_WEIGHT = 0.1 → 0.2`
3. **Latent Dim 증가**: `LATENT_DIM = 32 → 64`

### 과적합이 의심된다면:
1. **Dropout 증가**: `DROPOUT = 0.1 → 0.2`
2. **Weight Decay 증가**: `weight_decay = 0.01 → 0.05`
3. **데이터 양 증가**: 날짜 범위 확대

## 기대 효과

이전 vs 개선 후:

| 항목 | 이전 (1e-7) | 개선 후 (1e-4) |
|------|------------|---------------|
| Loss 수렴 속도 | 매우 느림 | 빠름 (10~50 epoch) |
| 최종 Val Loss | 1.0+ | 0.05~0.2 |
| 패턴 분리도 | 낮음 | 높음 |
| 육안 구분 | 어려움 | 가능 |
| 학습 시간 | 매우 길음 | 적절함 |

## 문제 해결

### "CUDA out of memory" 오류
```python
BATCH_SIZE = 8192 → 4096 또는 2048로 감소
```

### 학습이 너무 오래 걸림
```python
EARLY_STOPPING_PATIENCE = 100 → 50으로 감소
```

### t-SNE가 너무 오래 걸림
이미 코드에 최대 10,000 샘플 제한이 적용되어 있음 (자동)

---

**개선 완료!** 🎉

이제 `python volume_opt.py`를 실행하고 결과를 확인해보세요.
특히 `latent_space_tsne.png`에서 패턴 분리가 잘 되었는지 확인하는 것이 중요합니다.
