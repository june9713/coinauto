"""
특정 시간의 백테스트 결과 분석 프로그램
2025-11-03 00:39와 00:42 시점의 백테스트 결과를 분석합니다.
"""
import pandas as pd
import json

def analyze_backtest_decision(csv_path, backtest_config_path, target_times):
    """
    백테스트 결과 분석

    Parameters:
    - csv_path: 히스토리 CSV 파일 경로
    - backtest_config_path: 백테스트 설정 파일 경로
    - target_times: 분석할 시간 리스트 (예: ['2025-11-03 00:39:00', '2025-11-03 00:42:00'])
    """
    # 백테스트 설정 로드
    with open(backtest_config_path, 'r') as f:
        config = json.load(f)

    print("="*80)
    print("백테스트 설정")
    print("="*80)
    print(f"거래량 윈도우 (volume_window): {config['volume_window']}")
    print(f"이동평균 윈도우 (ma_window): {config['ma_window']}")
    print(f"거래량 배수 (volume_multiplier): {config['volume_multiplier']}")
    print(f"매수 현금 비율 (buy_cash_ratio): {config['buy_cash_ratio']}")
    print(f"보유 기간 (hold_period): {config['hold_period']}")
    print(f"이익실현 목표 (profit_target): {config['profit_target']}%")
    print(f"손절 기준 (stop_loss): {config['stop_loss']}%")
    print(f"가격 슬리퍼지 (price_slippage): {config['price_slippage']}")
    print()

    # CSV 파일 로드
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()  # 컬럼명 공백 제거

    # 첫 번째 컬럼이 인덱스인 경우 처리
    if df.columns[0] == '':
        df = df.set_index(df.columns[0])
        df.index.name = 'date'
    elif 'Unnamed: 0' in df.columns:
        df = df.set_index('Unnamed: 0')
        df.index.name = 'date'

    # 날짜 인덱스를 datetime으로 변환
    df.index = pd.to_datetime(df.index)

    print("="*80)
    print("데이터 정보")
    print("="*80)
    print(f"총 캔들 수: {len(df)}")
    print(f"데이터 기간: {df.index[0]} ~ {df.index[-1]}")
    print()

    # 거래량 평균 계산 (55개 이전 캔들의 평균)
    volume_window = config['volume_window']
    ma_window = config['ma_window']
    volume_multiplier = config['volume_multiplier']

    # 이동평균 계산 (9개 이전 캔들의 종가 평균)
    df['volume_avg'] = df['volume'].shift(1).rolling(window=volume_window, min_periods=1).mean()
    df['ma'] = df['close'].shift(1).rolling(window=ma_window, min_periods=1).mean()

    # 첫 번째 값 설정
    if len(df) > 0:
        df.loc[df.index[0], 'volume_avg'] = 0.0
        df.loc[df.index[0], 'ma'] = df.loc[df.index[0], 'close']

    # 조건 계산
    df['condition_b'] = df['volume'] >= (df['volume_avg'] * volume_multiplier)
    df['condition_d'] = df['close'] > df['ma']
    df['condition_e'] = df['open'] < df['close']  # 양봉
    df['all_conditions'] = df['condition_b'] & df['condition_d'] & df['condition_e']

    print("="*80)
    print("시간별 백테스트 결과 분석")
    print("="*80)

    for target_time in target_times:
        target_dt = pd.to_datetime(target_time)

        print(f"\n시간: {target_time}")
        print("-"*80)

        if target_dt not in df.index:
            print(f"  경고: {target_time} 시간의 캔들이 CSV에 없습니다.")
            print(f"  CSV에 있는 가장 가까운 시간을 확인하세요:")

            # 가장 가까운 시간 찾기
            time_diff = abs(df.index - target_dt)
            closest_idx = time_diff.argmin()
            closest_time = df.index[closest_idx]
            print(f"    가장 가까운 시간: {closest_time} (차이: {time_diff[closest_idx]})")
            continue

        row = df.loc[target_dt]
        idx = df.index.get_loc(target_dt)

        print(f"  캔들 정보:")
        print(f"    시가: {row['open']:,.0f}")
        print(f"    고가: {row['high']:,.0f}")
        print(f"    저가: {row['low']:,.0f}")
        print(f"    종가: {row['close']:,.0f}")
        print(f"    거래량: {row['volume']:.8f}")
        print()

        print(f"  매수 조건 분석:")
        print(f"    거래량 평균 (A): {row['volume_avg']:.8f}")
        print(f"    거래량 임계값 (A * {volume_multiplier}): {row['volume_avg'] * volume_multiplier:.8f}")
        print(f"    현재 거래량 (B): {row['volume']:.8f}")
        print(f"    조건 B (거래량 >= 임계값): {row['condition_b']} {'✓' if row['condition_b'] else '✗'}")
        print()

        print(f"    이동평균 (C): {row['ma']:,.2f}")
        print(f"    현재 종가: {row['close']:,.0f}")
        print(f"    조건 D (종가 > 이동평균): {row['condition_d']} {'✓' if row['condition_d'] else '✗'}")
        print()

        print(f"    시가: {row['open']:,.0f}")
        print(f"    종가: {row['close']:,.0f}")
        print(f"    조건 E (양봉, 시가 < 종가): {row['condition_e']} {'✓' if row['condition_e'] else '✗'}")
        print()

        print(f"  전체 조건 만족 (B & D & E): {row['all_conditions']} {'✓✓✓' if row['all_conditions'] else '✗✗✗'}")
        print()

        if row['all_conditions']:
            print(f"  결론: 매수 대기 (wait)")
            print(f"    → 다음 캔들(00:{target_dt.minute + 3:02d}:00)의 오픈가 + {config['price_slippage']:,}원에 매수 예정")
            print(f"    → 매수 금액: 현금의 {config['buy_cash_ratio']*100:.0f}%")

            # 다음 캔들이 있는지 확인
            if idx + 1 < len(df):
                next_candle = df.iloc[idx + 1]
                next_time = df.index[idx + 1]
                buy_price = next_candle['open'] + config['price_slippage']
                print(f"    → 다음 캔들 시간: {next_time}")
                print(f"    → 다음 캔들 오픈가: {next_candle['open']:,.0f}")
                print(f"    → 예상 매수가: {buy_price:,.0f}")
        else:
            # 미충족 조건 확인
            failed_conditions = []
            if not row['condition_b']:
                failed_conditions.append("B (거래량)")
            if not row['condition_d']:
                failed_conditions.append("D (이동평균)")
            if not row['condition_e']:
                failed_conditions.append("E (양봉)")

            print(f"  결론: 대기 (none) - 조건 미충족")
            print(f"    → 미충족 조건: {', '.join(failed_conditions)}")
        print()

if __name__ == "__main__":
    csv_path = "/home/jay/workspace/python/2025/coinauto/datas/2025-11-03/history_vw55_mw9_vm1.4_bcr0.9_hp15_pt17.6_sl-28.6_slip1000_btc_3m_20251103_00.csv"
    backtest_config_path = "/home/jay/workspace/python/2025/coinauto/backtest_conditions.json"

    # 분석할 시간
    target_times = [
        '2025-11-03 00:39:00',
        '2025-11-03 00:42:00'
    ]

    analyze_backtest_decision(csv_path, backtest_config_path, target_times)
