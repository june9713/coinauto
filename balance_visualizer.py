"""
잔고 변동 이력 시각화 모듈
CSV 파일에서 잔고 이력을 읽어 그래프를 생성합니다.
"""
import traceback
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager, rc
import os
from config import Config


class BalanceVisualizer:
    """잔고 변동 이력 시각화 클래스"""

    def __init__(self):
        """초기화"""
        # 한글 폰트 설정 (한글이 깨지지 않도록)
        try:
            # Config에서 설정된 폰트 경로 사용
            font_path = Config.FONT_PATH

            if os.path.exists(font_path):
                # 폰트 파일이 있으면 해당 폰트 사용
                font_manager.fontManager.addfont(font_path)
                font_prop = font_manager.FontProperties(fname=font_path)
                font_name = font_prop.get_name()

                # 폰트 설정 - 여러 방식으로 확실하게 적용
                rc('font', family=font_name)
                plt.rcParams['font.family'] = font_name
                plt.rcParams['font.sans-serif'] = [font_name]

                print(f"한글 폰트 설정 완료: {font_name}")
            else:
                # 폰트 파일이 없으면 OS별 기본 폰트 사용
                if os.name == 'nt':  # Windows
                    font_name = 'Malgun Gothic'
                elif os.name == 'posix':  # Linux, Mac
                    font_name = 'DejaVu Sans'

                rc('font', family=font_name)
                print(f"경고: 폰트 파일을 찾을 수 없습니다 ({font_path}). 기본 폰트 사용: {font_name}")

            plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
        except Exception as e:
            print(f"경고: 폰트 설정 실패 ({e}). 기본 폰트 사용")

    def load_balance_history_from_csv(self, ticker='BTC', condition_dict=None):
        """
        백테스트 결과 CSV에서 잔고 변동 이력을 로드합니다.

        Parameters:
        - ticker (str): 암호화폐 티커
        - condition_dict (dict, optional): 조건 딕셔너리

        Returns:
        - pd.DataFrame: 잔고 변동 이력 데이터프레임
        """
        try:
            # 백테스트 결과 파일 경로
            result_file = Config.get_backtest_result_file_path(ticker=ticker, condition_dict=condition_dict)

            if not os.path.exists(result_file):
                print(f"백테스트 결과 파일이 없습니다: {result_file}")
                return None

            # CSV 로드
            df = pd.read_csv(result_file)

            # 필요한 컬럼 확인
            required_columns = ['execution_time', 'current_krw_balance', 'current_btc_balance', 'total_balance']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                print(f"경고: 필요한 컬럼이 없습니다: {missing_columns}")
                return None

            # execution_time을 datetime으로 변환
            df['execution_time'] = pd.to_datetime(df['execution_time'])

            # 잔고 정보가 있는 행만 필터링
            df = df[df['current_krw_balance'].notna() & df['current_btc_balance'].notna()]

            if len(df) == 0:
                print("잔고 정보가 있는 데이터가 없습니다.")
                return None

            # execution_time 기준으로 정렬
            df = df.sort_values('execution_time')

            # 중복 제거 (같은 시간에 여러 거래가 있을 수 있음 - 마지막 것만 유지)
            df = df.drop_duplicates(subset=['execution_time'], keep='last')

            # 최대 1일 데이터만 필터링
            now = pd.Timestamp.now()
            one_day_ago = now - pd.Timedelta(days=1)
            df = df[df['execution_time'] >= one_day_ago]

            if len(df) == 0:
                print("최근 1일 이내의 잔고 정보가 있는 데이터가 없습니다.")
                return None

            print(f"잔고 이력 로드 완료: {len(df)}개 데이터 (최근 1일)")
            print(f"  기간: {df['execution_time'].iloc[0]} ~ {df['execution_time'].iloc[-1]}")

            return df

        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            return None

    def plot_balance_history(self, ticker='BTC', condition_dict=None, output_path='./images/balance_history.jpg'):
        """
        잔고 변동 이력 그래프를 생성합니다.

        Parameters:
        - ticker (str): 암호화폐 티커
        - condition_dict (dict, optional): 조건 딕셔너리
        - output_path (str): 출력 이미지 경로

        Returns:
        - str: 생성된 이미지 경로 (성공 시) 또는 None (실패 시)
        """
        try:
            # 잔고 이력 로드
            df = self.load_balance_history_from_csv(ticker=ticker, condition_dict=condition_dict)

            if df is None or len(df) == 0:
                print("잔고 이력 데이터가 없어 그래프를 생성할 수 없습니다.")
                return None

            # 출력 디렉토리 확인 및 생성
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"출력 폴더 생성: {output_dir}")

            # 그래프 생성
            fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

            # 1. 총 자산 (KRW + BTC 가치)
            ax1 = axes[0]
            ax1.plot(df['execution_time'], df['total_balance'],
                    color='#667eea', linewidth=2, marker='o', markersize=4)
            ax1.set_ylabel('Total Balance (KRW)', fontsize=12, fontweight='bold')
            ax1.set_title(f'{ticker} Balance History - Total Assets', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

            # 2. KRW 잔고
            ax2 = axes[1]
            ax2.plot(df['execution_time'], df['current_krw_balance'],
                    color='#10b981', linewidth=2, marker='s', markersize=4)
            ax2.set_ylabel('KRW Balance', fontsize=12, fontweight='bold')
            ax2.set_title('KRW Balance History', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

            # 3. BTC 잔고
            ax3 = axes[2]
            ax3.plot(df['execution_time'], df['current_btc_balance'],
                    color='#f59e0b', linewidth=2, marker='^', markersize=4)
            ax3.set_ylabel(f'{ticker} Balance', fontsize=12, fontweight='bold')
            ax3.set_title(f'{ticker} Balance History', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Time', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.8f}'))

            # x축 날짜 포맷 설정
            for ax in axes:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                ax.tick_params(axis='x', rotation=45)

            plt.tight_layout()

            # 이미지 저장
            plt.savefig(output_path, dpi=Config.GRAPH_DPI, bbox_inches='tight')
            plt.close()

            print(f"잔고 이력 그래프 저장 완료: {output_path}")

            return output_path

        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            return None


if __name__ == "__main__":
    # 테스트
    visualizer = BalanceVisualizer()
    visualizer.plot_balance_history(ticker='BTC', output_path='./images/balance_history.jpg')
