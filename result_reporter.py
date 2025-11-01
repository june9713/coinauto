"""
결과 리포터 모듈
백테스트 결과 출력 기능 제공
"""
import traceback
import pandas as pd


class ResultReporter:
    """백테스트 결과 출력 클래스"""
    
    def __init__(self):
        """초기화"""
        pass
    
    def print_backtest_results(self, result):
        """
        백테스트 결과 출력
        
        Parameters:
        - result (dict): 백테스트 결과 딕셔너리
        """
        try:
            trades = result['trades']
            
            print("\n" + "="*80)
            print("백테스트 결과")
            print("="*80)
            print(f"초기 자본: {result['initial_capital']:,.0f}원")
            print(f"최종 자산: {result['final_asset']:,.0f}원")
            print(f"총 수익률: {result['total_return']:+.2f}%")
            print(f"\n총 거래 횟수: {result['total_trades']}회")
            print(f"매수 횟수: {result['buy_count']}회")
            print(f"매도 횟수: {result['sell_count']}회")
            print("\n거래 내역:")
            print("-"*80)
            
            total_profit = 0.0
            profitable_trades = 0
            losing_trades = 0
            
            for trade in trades:
                # 날짜를 초 단위까지 포함하여 출력
                if isinstance(trade['date'], pd.Timestamp):
                    date_str = trade['date'].strftime('%Y-%m-%d %H:%M:%S')
                else:
                    date_obj = pd.to_datetime(trade['date'])
                    date_str = date_obj.strftime('%Y-%m-%d %H:%M:%S')
                action = trade['action']
                price = trade['price']
                
                print(f"\n[{date_str}] {action}")
                print(f"  가격: {price:,.0f}원")
                
                if 'amount' in trade:
                    print(f"  수량: {trade['amount']:.8f} BTC")
                    print(f"  총액: {trade['total_value']:,.0f}원")
                
                if 'buy_price' in trade:
                    buy_price = trade['buy_price']
                    # 매수 날짜도 초 단위까지 포함하여 출력
                    if isinstance(trade['buy_date'], pd.Timestamp):
                        buy_date_str = trade['buy_date'].strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        buy_date_obj = pd.to_datetime(trade['buy_date'])
                        buy_date_str = buy_date_obj.strftime('%Y-%m-%d %H:%M:%S')
                    profit = trade['profit']
                    profit_percent = trade['profit_percent']
                    total_profit += profit
                    
                    print(f"  매수 가격: {buy_price:,.0f}원 (매수일: {buy_date_str})")
                    print(f"  수익: {profit:,.0f}원 ({profit_percent:+.2f}%)")
                    
                    if profit > 0:
                        profitable_trades += 1
                    else:
                        losing_trades += 1
                
                if 'total_asset' in trade:
                    print(f"  거래 후 총 자산: {trade['total_asset']:,.0f}원")
                
                if 'angle' in trade and trade['angle'] is not None:
                    print(f"  추세선 각도: {trade['angle']:.2f}°")
            
            # 거래 간격 계산
            trade_intervals = self._calculate_trade_intervals(trades)
            
            # 평균 거래 간격 계산
            avg_interval = 0.0
            if len(trade_intervals) > 0:
                avg_interval = sum(trade_intervals) / len(trade_intervals)
            
            # 최소/최대 거래 간격
            min_interval = min(trade_intervals) if len(trade_intervals) > 0 else 0
            max_interval = max(trade_intervals) if len(trade_intervals) > 0 else 0
            
            print("\n" + "-"*80)
            print(f"총 수익: {total_profit:,.0f}원")
            print(f"수익 거래: {profitable_trades}회")
            print(f"손실 거래: {losing_trades}회")
            if profitable_trades + losing_trades > 0:
                win_rate = (profitable_trades / (profitable_trades + losing_trades)) * 100
                print(f"승률: {win_rate:.2f}%")
            
            print("\n거래 간격 분석:")
            if len(trade_intervals) > 0:
                print(f"  평균 거래 간격: {avg_interval:.1f}일")
                print(f"  최소 거래 간격: {min_interval}일")
                print(f"  최대 거래 간격: {max_interval}일")
                print(f"  거래 간격 개수: {len(trade_intervals)}개")
            else:
                print(f"  거래가 1회 이하여서 간격을 계산할 수 없습니다.")
            
            print("="*80)
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def _calculate_trade_intervals(self, trades):
        """
        거래 간격 계산
        
        Parameters:
        - trades (list): 거래 리스트
        
        Returns:
        - list: 거래 간격 리스트 (일수)
        """
        try:
            trade_dates = []
            for trade in trades:
                if isinstance(trade['date'], pd.Timestamp):
                    trade_dates.append(trade['date'])
                else:
                    trade_dates.append(pd.to_datetime(trade['date']))
            
            trade_intervals = []  # 거래 간격 (일수)
            if len(trade_dates) > 1:
                # 거래 날짜를 정렬
                sorted_dates = sorted(trade_dates)
                for i in range(1, len(sorted_dates)):
                    interval_days = (sorted_dates[i] - sorted_dates[i-1]).days
                    trade_intervals.append(interval_days)
            
            return trade_intervals
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def get_summary(self, result):
        """
        백테스트 결과 요약 딕셔너리 반환
        
        Parameters:
        - result (dict): 백테스트 결과 딕셔너리
        
        Returns:
        - dict: 요약 정보 딕셔너리
        """
        try:
            trades = result['trades']
            total_profit = 0.0
            profitable_trades = 0
            losing_trades = 0
            
            for trade in trades:
                if 'profit' in trade:
                    profit = trade['profit']
                    total_profit += profit
                    if profit > 0:
                        profitable_trades += 1
                    else:
                        losing_trades += 1
            
            win_rate = 0.0
            if profitable_trades + losing_trades > 0:
                win_rate = (profitable_trades / (profitable_trades + losing_trades)) * 100
            
            trade_intervals = self._calculate_trade_intervals(trades)
            avg_interval = sum(trade_intervals) / len(trade_intervals) if len(trade_intervals) > 0 else 0.0
            
            return {
                'initial_capital': result['initial_capital'],
                'final_asset': result['final_asset'],
                'total_return': result['total_return'],
                'total_trades': result['total_trades'],
                'buy_count': result['buy_count'],
                'sell_count': result['sell_count'],
                'total_profit': total_profit,
                'profitable_trades': profitable_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'avg_trade_interval': avg_interval
            }
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise

