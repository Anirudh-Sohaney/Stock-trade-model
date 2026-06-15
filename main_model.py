"""
main_model.py - Core signal engine.
360 decision rules from ensemble distillation.
Zero ML dependencies at runtime.
"""

import numpy as np


class HardcodedTrader:
    """
    Stage 3 (Optimized): High-risk, high-return.
    OR-logic rule evaluation - any matching path triggers BUY.
    """

    def __init__(self):
        self.stage = 3
        self.rule_count = 0

    def generate_signal(self, f):
        """
        f: dict of feature_name -> value
        Returns: "BUY" or "HOLD"
        """

        if (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.600505
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.771319
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.078953
                and not np.isnan(f.get('atr_percentile', np.nan)) and f.get('atr_percentile', np.nan) <= 0.172203
            ):
            return "BUY"  # path: mean_ret=3.226%, n=5, risk_score=6, worst=2.64%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.184859
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.582647
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.603804
                and not np.isnan(f.get('consec_green', np.nan)) and f.get('consec_green', np.nan) > 2.500000
            ):
            return "BUY"  # path: mean_ret=3.715%, n=5, risk_score=7, worst=2.50%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.095735
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.522399
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) > 0.065242
                and not np.isnan(f.get('accel_3', np.nan)) and f.get('accel_3', np.nan) <= -3.820164
            ):
            return "BUY"  # path: mean_ret=1.385%, n=5, risk_score=8, worst=0.98%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.172446
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) <= 0.825679
                and not np.isnan(f.get('obv', np.nan)) and f.get('obv', np.nan) > 60379536.000000
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) > 18.500000
            ):
            return "BUY"  # path: mean_ret=0.923%, n=6, risk_score=6, worst=0.57%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.187251
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.826150
                and not np.isnan(f.get('hv_5', np.nan)) and f.get('hv_5', np.nan) <= 0.475065
                and not np.isnan(f.get('rsi_14', np.nan)) and f.get('rsi_14', np.nan) <= 23.005942
            ):
            return "BUY"  # path: mean_ret=1.538%, n=5, risk_score=10, worst=0.82%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.099156
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.807449
                and not np.isnan(f.get('hv_10', np.nan)) and f.get('hv_10', np.nan) <= 1.506217
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) <= 1.380874
            ):
            return "BUY"  # path: mean_ret=0.961%, n=92, risk_score=6, worst=0.29%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.185193
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.011422
                and not np.isnan(f.get('consec_green', np.nan)) and f.get('consec_green', np.nan) > 2.500000
                and not np.isnan(f.get('trend_persistence', np.nan)) and f.get('trend_persistence', np.nan) > 0.500000
            ):
            return "BUY"  # path: mean_ret=3.377%, n=5, risk_score=7, worst=1.58%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.269325
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.843705
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.071773
                and not np.isnan(f.get('hv_5', np.nan)) and f.get('hv_5', np.nan) <= 0.574867
            ):
            return "BUY"  # path: mean_ret=1.775%, n=6, risk_score=7, worst=1.04%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.184972
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) <= 0.762533
                and not np.isnan(f.get('obv', np.nan)) and f.get('obv', np.nan) > 63574836.000000
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) > 40.500000
            ):
            return "BUY"  # path: mean_ret=0.912%, n=5, risk_score=6, worst=0.57%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.187251
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) <= 0.801724
                and not np.isnan(f.get('obv', np.nan)) and f.get('obv', np.nan) > 63367692.000000
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) > 18.500000
            ):
            return "BUY"  # path: mean_ret=0.912%, n=5, risk_score=6, worst=0.57%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.184859
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) <= 0.744884
                and not np.isnan(f.get('obv', np.nan)) and f.get('obv', np.nan) > 63507998.000000
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) > 40.500000
            ):
            return "BUY"  # path: mean_ret=0.912%, n=5, risk_score=6, worst=0.57%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.104876
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.822526
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) > 0.650991
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) > 1.639247
            ):
            return "BUY"  # path: mean_ret=2.864%, n=7, risk_score=6, worst=1.11%

        elif (
                not np.isnan(f.get('volume_ratio', np.nan)) and f.get('volume_ratio', np.nan) > 3.205501
                and not np.isnan(f.get('macd_hist', np.nan)) and f.get('macd_hist', np.nan) <= -0.526443
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.224122
                and not np.isnan(f.get('atr_21', np.nan)) and f.get('atr_21', np.nan) > 5.316673
            ):
            return "BUY"  # path: mean_ret=1.452%, n=6, risk_score=9, worst=0.20%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.099156
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.807449
                and not np.isnan(f.get('hv_10', np.nan)) and f.get('hv_10', np.nan) <= 1.506217
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.380874
            ):
            return "BUY"  # path: mean_ret=1.434%, n=32, risk_score=6, worst=-0.49%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.187712
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.012873
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 41.500000
                and not np.isnan(f.get('consec_green', np.nan)) and f.get('consec_green', np.nan) > 2.500000
            ):
            return "BUY"  # path: mean_ret=2.767%, n=14, risk_score=7, worst=1.01%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.104669
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.474437
                and not np.isnan(f.get('atr_21', np.nan)) and f.get('atr_21', np.nan) > 7.041185
                and not np.isnan(f.get('body_to_range', np.nan)) and f.get('body_to_range', np.nan) <= 0.235990
            ):
            return "BUY"  # path: mean_ret=2.423%, n=7, risk_score=6, worst=1.14%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.287699
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.441326
                and not np.isnan(f.get('atr_pct', np.nan)) and f.get('atr_pct', np.nan) <= 0.558473
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.118845
            ):
            return "BUY"  # path: mean_ret=1.047%, n=189, risk_score=7, worst=-0.17%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.269817
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.810907
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.629292
                and not np.isnan(f.get('stoch_k', np.nan)) and f.get('stoch_k', np.nan) > 91.905201
            ):
            return "BUY"  # path: mean_ret=4.052%, n=6, risk_score=8, worst=0.29%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.246102
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) <= 0.849558
                and not np.isnan(f.get('obv', np.nan)) and f.get('obv', np.nan) > 60704656.000000
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) > 18.500000
            ):
            return "BUY"  # path: mean_ret=0.809%, n=10, risk_score=5, worst=-0.04%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.556468
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.415792
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.051188
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) <= 0.702407
            ):
            return "BUY"  # path: mean_ret=1.235%, n=68, risk_score=6, worst=0.29%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.569586
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 20.425514
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.091344
                and not np.isnan(f.get('hv_20', np.nan)) and f.get('hv_20', np.nan) <= 1.359296
            ):
            return "BUY"  # path: mean_ret=1.108%, n=123, risk_score=6, worst=-0.49%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.185269
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.005469
                and not np.isnan(f.get('consec_green', np.nan)) and f.get('consec_green', np.nan) > 2.500000
                and not np.isnan(f.get('hv_5', np.nan)) and f.get('hv_5', np.nan) > 1.778911
            ):
            return "BUY"  # path: mean_ret=2.562%, n=10, risk_score=7, worst=1.01%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.108071
                and not np.isnan(f.get('hv_5', np.nan)) and f.get('hv_5', np.nan) > 1.315906
                and not np.isnan(f.get('volume_ratio', np.nan)) and f.get('volume_ratio', np.nan) > 6.357044
                and not np.isnan(f.get('stoch_d', np.nan)) and f.get('stoch_d', np.nan) <= 50.414199
            ):
            return "BUY"  # path: mean_ret=1.574%, n=9, risk_score=10, worst=0.00%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.148352
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.308300
                and not np.isnan(f.get('atr_pct', np.nan)) and f.get('atr_pct', np.nan) <= 0.567760
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) <= 1.123791
            ):
            return "BUY"  # path: mean_ret=0.873%, n=144, risk_score=6, worst=-0.17%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.489351
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.647976
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.164335
                and not np.isnan(f.get('macd_hist', np.nan)) and f.get('macd_hist', np.nan) > 1.958789
            ):
            return "BUY"  # path: mean_ret=2.216%, n=11, risk_score=6, worst=0.52%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.184859
                and not np.isnan(f.get('compression_score', np.nan)) and f.get('compression_score', np.nan) <= 0.531778
                and not np.isnan(f.get('hv_5', np.nan)) and f.get('hv_5', np.nan) > 1.821920
                and not np.isnan(f.get('consec_red', np.nan)) and f.get('consec_red', np.nan) <= 0.111014
            ):
            return "BUY"  # path: mean_ret=2.493%, n=11, risk_score=8, worst=0.66%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.187846
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.715557
                and not np.isnan(f.get('ret_2', np.nan)) and f.get('ret_2', np.nan) > -1.292887
                and not np.isnan(f.get('dist_prev_close', np.nan)) and f.get('dist_prev_close', np.nan) <= -1.832045
            ):
            return "BUY"  # path: mean_ret=1.916%, n=15, risk_score=7, worst=0.09%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.248991
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) <= 0.837842
                and not np.isnan(f.get('atr_21', np.nan)) and f.get('atr_21', np.nan) > 7.852953
                and not np.isnan(f.get('hv_10', np.nan)) and f.get('hv_10', np.nan) <= 1.689018
            ):
            return "BUY"  # path: mean_ret=0.562%, n=7, risk_score=5, worst=-0.10%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.215190
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.038889
                and not np.isnan(f.get('consec_green', np.nan)) and f.get('consec_green', np.nan) > 2.500000
                and not np.isnan(f.get('atr_percentile', np.nan)) and f.get('atr_percentile', np.nan) <= 0.500000
            ):
            return "BUY"  # path: mean_ret=2.880%, n=12, risk_score=6, worst=-0.89%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.215190
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.011379
                and not np.isnan(f.get('consec_green', np.nan)) and f.get('consec_green', np.nan) > 2.500000
                and not np.isnan(f.get('atr_percentile', np.nan)) and f.get('atr_percentile', np.nan) <= 0.500000
            ):
            return "BUY"  # path: mean_ret=2.880%, n=12, risk_score=6, worst=-0.89%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.176189
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 14.385687
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.256680
                and not np.isnan(f.get('atr_pct', np.nan)) and f.get('atr_pct', np.nan) <= 0.547628
            ):
            return "BUY"  # path: mean_ret=0.962%, n=312, risk_score=6, worst=-0.17%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.370170
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.145184
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.406753
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) > 1.310389
            ):
            return "BUY"  # path: mean_ret=2.556%, n=34, risk_score=6, worst=0.18%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.187251
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.745614
                and not np.isnan(f.get('hv_5', np.nan)) and f.get('hv_5', np.nan) <= 0.476685
                and not np.isnan(f.get('obv_slope', np.nan)) and f.get('obv_slope', np.nan) <= -5.937530
            ):
            return "BUY"  # path: mean_ret=1.231%, n=10, risk_score=7, worst=0.47%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.215440
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.845284
                and not np.isnan(f.get('hv_5', np.nan)) and f.get('hv_5', np.nan) <= 0.475065
                and not np.isnan(f.get('obv', np.nan)) and f.get('obv', np.nan) > 1300069.500000
            ):
            return "BUY"  # path: mean_ret=0.994%, n=5, risk_score=5, worst=0.01%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.374693
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.121986
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 21.565150
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) <= 0.504040
            ):
            return "BUY"  # path: mean_ret=0.635%, n=28, risk_score=6, worst=-0.17%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.029412
                and not np.isnan(f.get('ret_3', np.nan)) and f.get('ret_3', np.nan) > -1.656641
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.279605
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 17.010938
            ):
            return "BUY"  # path: mean_ret=1.515%, n=15, risk_score=7, worst=-0.27%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.178492
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.250520
                and not np.isnan(f.get('atr_pct', np.nan)) and f.get('atr_pct', np.nan) <= 1.007099
                and not np.isnan(f.get('atr_pct', np.nan)) and f.get('atr_pct', np.nan) > 0.658338
            ):
            return "BUY"  # path: mean_ret=1.901%, n=160, risk_score=6, worst=-0.49%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.532705
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 20.212102
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.081662
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.402501
            ):
            return "BUY"  # path: mean_ret=1.878%, n=107, risk_score=6, worst=-0.49%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.148352
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.308300
                and not np.isnan(f.get('atr_pct', np.nan)) and f.get('atr_pct', np.nan) <= 0.567760
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.123791
            ):
            return "BUY"  # path: mean_ret=1.136%, n=124, risk_score=6, worst=-0.00%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.269328
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) <= 0.802182
                and not np.isnan(f.get('atr_21', np.nan)) and f.get('atr_21', np.nan) > 7.908667
                and not np.isnan(f.get('hv_10', np.nan)) and f.get('hv_10', np.nan) <= 1.699170
            ):
            return "BUY"  # path: mean_ret=0.573%, n=6, risk_score=5, worst=-0.10%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.419141
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.385087
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 20.221133
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.077142
            ):
            return "BUY"  # path: mean_ret=1.869%, n=98, risk_score=6, worst=-0.49%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.249211
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.749766
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.572154
                and not np.isnan(f.get('stoch_k', np.nan)) and f.get('stoch_k', np.nan) > 91.905201
            ):
            return "BUY"  # path: mean_ret=3.918%, n=5, risk_score=8, worst=0.29%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.269324
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.708011
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.567162
                and not np.isnan(f.get('stoch_k', np.nan)) and f.get('stoch_k', np.nan) > 93.962139
            ):
            return "BUY"  # path: mean_ret=3.791%, n=5, risk_score=8, worst=0.29%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.217463
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 26.994065
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.756439
                and not np.isnan(f.get('consec_red', np.nan)) and f.get('consec_red', np.nan) <= 0.036573
            ):
            return "BUY"  # path: mean_ret=2.837%, n=9, risk_score=6, worst=-0.51%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.072284
                and not np.isnan(f.get('roc_5', np.nan)) and f.get('roc_5', np.nan) <= -1.261401
                and not np.isnan(f.get('accel_3', np.nan)) and f.get('accel_3', np.nan) <= -1.717556
            ):
            return "BUY"  # path: mean_ret=1.835%, n=43, risk_score=8, worst=0.00%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.104876
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.822526
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) <= 0.650991
                and not np.isnan(f.get('ema_9_50_spread', np.nan)) and f.get('ema_9_50_spread', np.nan) <= 1.659529
            ):
            return "BUY"  # path: mean_ret=1.202%, n=188, risk_score=6, worst=-0.01%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.187846
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.599448
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.603804
                and not np.isnan(f.get('ema_50', np.nan)) and f.get('ema_50', np.nan) <= 1.485255
            ):
            return "BUY"  # path: mean_ret=3.840%, n=6, risk_score=6, worst=1.13%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.319486
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 20.198726
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) <= 0.318912
                and not np.isnan(f.get('compression_score', np.nan)) and f.get('compression_score', np.nan) > 0.410621
            ):
            return "BUY"  # path: mean_ret=0.630%, n=36, risk_score=5, worst=-0.05%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.205821
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.088713
                and not np.isnan(f.get('atr_percentile', np.nan)) and f.get('atr_percentile', np.nan) > 0.088583
                and not np.isnan(f.get('consec_green', np.nan)) and f.get('consec_green', np.nan) > 2.500000
            ):
            return "BUY"  # path: mean_ret=2.658%, n=15, risk_score=6, worst=-0.51%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.571882
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 24.954226
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.099659
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) <= 0.832543
            ):
            return "BUY"  # path: mean_ret=0.700%, n=172, risk_score=6, worst=-0.01%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.569586
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 20.425514
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.091344
                and not np.isnan(f.get('hv_20', np.nan)) and f.get('hv_20', np.nan) > 1.359296
            ):
            return "BUY"  # path: mean_ret=2.080%, n=115, risk_score=6, worst=-0.36%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.190855
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.790052
                and not np.isnan(f.get('hv_5', np.nan)) and f.get('hv_5', np.nan) > 0.476941
                and not np.isnan(f.get('rsi_7_slope', np.nan)) and f.get('rsi_7_slope', np.nan) <= -65.949272
            ):
            return "BUY"  # path: mean_ret=1.225%, n=5, risk_score=10, worst=0.24%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.186099
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.554877
                and not np.isnan(f.get('rsi_7_slope', np.nan)) and f.get('rsi_7_slope', np.nan) <= -65.949272
            ):
            return "BUY"  # path: mean_ret=1.225%, n=5, risk_score=9, worst=0.24%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.287699
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.441326
                and not np.isnan(f.get('atr_pct', np.nan)) and f.get('atr_pct', np.nan) > 0.558473
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.133347
            ):
            return "BUY"  # path: mean_ret=2.069%, n=212, risk_score=7, worst=-0.49%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.038014
                and not np.isnan(f.get('ema12_slope', np.nan)) and f.get('ema12_slope', np.nan) > -0.202223
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.501965
                and not np.isnan(f.get('lower_wick_pct', np.nan)) and f.get('lower_wick_pct', np.nan) <= 0.104944
            ):
            return "BUY"  # path: mean_ret=1.288%, n=28, risk_score=8, worst=-0.49%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.319486
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 20.198726
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) <= 0.318912
                and not np.isnan(f.get('compression_score', np.nan)) and f.get('compression_score', np.nan) <= 0.410621
            ):
            return "BUY"  # path: mean_ret=0.708%, n=98, risk_score=5, worst=-0.14%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.059756
                and not np.isnan(f.get('atr_pct', np.nan)) and f.get('atr_pct', np.nan) > 0.745928
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 38.701579
                and not np.isnan(f.get('atr_norm_candle', np.nan)) and f.get('atr_norm_candle', np.nan) > 1.264936
            ):
            return "BUY"  # path: mean_ret=1.963%, n=54, risk_score=6, worst=-0.49%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.201790
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.898074
                and not np.isnan(f.get('consec_green', np.nan)) and f.get('consec_green', np.nan) > 2.524149
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) <= 1.317663
            ):
            return "BUY"  # path: mean_ret=1.786%, n=17, risk_score=6, worst=-0.61%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.836405
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.128764
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) <= 0.667015
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.507755
            ):
            return "BUY"  # path: mean_ret=1.159%, n=258, risk_score=6, worst=-0.17%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.104876
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.822526
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) > 0.650991
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) <= 1.639247
            ):
            return "BUY"  # path: mean_ret=2.198%, n=106, risk_score=6, worst=-0.49%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.184874
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.566539
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.116729
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) > 0.116570
            ):
            return "BUY"  # path: mean_ret=0.446%, n=5, risk_score=8, worst=0.25%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.135042
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 23.131360
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.954907
                and not np.isnan(f.get('bb_width', np.nan)) and f.get('bb_width', np.nan) <= 0.021378
            ):
            return "BUY"  # path: mean_ret=1.070%, n=196, risk_score=6, worst=-0.17%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.029412
                and not np.isnan(f.get('ret_3', np.nan)) and f.get('ret_3', np.nan) <= -1.656641
                and not np.isnan(f.get('macd_above_signal', np.nan)) and f.get('macd_above_signal', np.nan) <= 0.544498
                and not np.isnan(f.get('ret_3', np.nan)) and f.get('ret_3', np.nan) <= -4.514460
            ):
            return "BUY"  # path: mean_ret=2.507%, n=5, risk_score=10, worst=1.09%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.528728
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.209294
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.083607
                and not np.isnan(f.get('bb_width', np.nan)) and f.get('bb_width', np.nan) <= 0.019182
            ):
            return "BUY"  # path: mean_ret=0.969%, n=295, risk_score=5, worst=-0.23%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.178492
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.250520
                and not np.isnan(f.get('atr_pct', np.nan)) and f.get('atr_pct', np.nan) <= 1.007099
                and not np.isnan(f.get('atr_pct', np.nan)) and f.get('atr_pct', np.nan) <= 0.658338
            ):
            return "BUY"  # path: mean_ret=1.087%, n=393, risk_score=6, worst=-0.32%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.500298
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.225079
                and not np.isnan(f.get('atr_pct', np.nan)) and f.get('atr_pct', np.nan) > 0.458334
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.105742
            ):
            return "BUY"  # path: mean_ret=1.962%, n=189, risk_score=6, worst=-0.49%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.148352
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.308300
                and not np.isnan(f.get('atr_pct', np.nan)) and f.get('atr_pct', np.nan) > 0.567760
                and not np.isnan(f.get('gap_up_pct', np.nan)) and f.get('gap_up_pct', np.nan) <= 9.407795
            ):
            return "BUY"  # path: mean_ret=1.993%, n=252, risk_score=8, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.099156
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.807449
                and not np.isnan(f.get('hv_10', np.nan)) and f.get('hv_10', np.nan) > 1.506217
                and not np.isnan(f.get('ema21_slope', np.nan)) and f.get('ema21_slope', np.nan) > -0.840324
            ):
            return "BUY"  # path: mean_ret=2.003%, n=151, risk_score=7, worst=-0.36%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.185060
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.577406
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.403565
                and not np.isnan(f.get('consec_green', np.nan)) and f.get('consec_green', np.nan) > 2.500000
            ):
            return "BUY"  # path: mean_ret=2.778%, n=8, risk_score=7, worst=-0.89%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.187311
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.825794
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.403639
                and not np.isnan(f.get('consec_green', np.nan)) and f.get('consec_green', np.nan) > 2.500000
            ):
            return "BUY"  # path: mean_ret=2.778%, n=8, risk_score=7, worst=-0.89%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.349119
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) > 0.571821
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.506634
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.178092
            ):
            return "BUY"  # path: mean_ret=1.917%, n=337, risk_score=6, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.135042
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 23.131360
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.954907
                and not np.isnan(f.get('bb_width', np.nan)) and f.get('bb_width', np.nan) > 0.021378
            ):
            return "BUY"  # path: mean_ret=1.874%, n=244, risk_score=6, worst=-0.49%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.556468
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.415792
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.051188
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) > 0.702407
            ):
            return "BUY"  # path: mean_ret=2.322%, n=36, risk_score=6, worst=-0.49%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.188029
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.802133
                and not np.isnan(f.get('hv_5', np.nan)) and f.get('hv_5', np.nan) <= 0.475065
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 17.000000
            ):
            return "BUY"  # path: mean_ret=1.066%, n=21, risk_score=6, worst=0.01%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.176189
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 14.385687
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.256680
                and not np.isnan(f.get('atr_pct', np.nan)) and f.get('atr_pct', np.nan) > 0.547628
            ):
            return "BUY"  # path: mean_ret=1.885%, n=352, risk_score=6, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.761899
                and not np.isnan(f.get('vwap_dev', np.nan)) and f.get('vwap_dev', np.nan) > 1.200634
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.236942
                and not np.isnan(f.get('volume_ratio', np.nan)) and f.get('volume_ratio', np.nan) > 1.875302
            ):
            return "BUY"  # path: mean_ret=2.426%, n=38, risk_score=8, worst=-0.07%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.763561
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.192861
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.488694
                and not np.isnan(f.get('consec_green', np.nan)) and f.get('consec_green', np.nan) > 2.191431
            ):
            return "BUY"  # path: mean_ret=2.948%, n=6, risk_score=7, worst=-0.89%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.630165
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.076550
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.772281
                and not np.isnan(f.get('cmf', np.nan)) and f.get('cmf', np.nan) <= -0.434603
            ):
            return "BUY"  # path: mean_ret=2.361%, n=7, risk_score=8, worst=0.00%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.172414
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.705229
                and not np.isnan(f.get('obv', np.nan)) and f.get('obv', np.nan) <= 146154520.000000
                and not np.isnan(f.get('consec_green', np.nan)) and f.get('consec_green', np.nan) > 2.500000
            ):
            return "BUY"  # path: mean_ret=1.847%, n=28, risk_score=7, worst=0.18%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.185193
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.676432
                and not np.isnan(f.get('volume_ratio', np.nan)) and f.get('volume_ratio', np.nan) > 5.881305
                and not np.isnan(f.get('volume_percentile', np.nan)) and f.get('volume_percentile', np.nan) <= 0.976967
            ):
            return "BUY"  # path: mean_ret=4.576%, n=6, risk_score=10, worst=0.42%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.270244
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.793960
                and not np.isnan(f.get('volume_ratio', np.nan)) and f.get('volume_ratio', np.nan) > 5.823412
                and not np.isnan(f.get('volume_percentile', np.nan)) and f.get('volume_percentile', np.nan) <= 0.977927
            ):
            return "BUY"  # path: mean_ret=4.576%, n=6, risk_score=9, worst=0.42%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.790128
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.067393
                and not np.isnan(f.get('hv_5', np.nan)) and f.get('hv_5', np.nan) > 2.979953
                and not np.isnan(f.get('bb_expansion', np.nan)) and f.get('bb_expansion', np.nan) > -0.007764
            ):
            return "BUY"  # path: mean_ret=1.990%, n=84, risk_score=7, worst=0.03%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.798872
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.096907
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 14.699968
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.494234
            ):
            return "BUY"  # path: mean_ret=1.580%, n=270, risk_score=6, worst=-0.49%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.038014
                and not np.isnan(f.get('ema12_slope', np.nan)) and f.get('ema12_slope', np.nan) <= -0.202223
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 18.796113
                and not np.isnan(f.get('dist_from_ema9', np.nan)) and f.get('dist_from_ema9', np.nan) <= -1.454414
            ):
            return "BUY"  # path: mean_ret=1.685%, n=43, risk_score=8, worst=-0.27%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.170765
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 14.390732
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.725726
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) <= 1.233221
            ):
            return "BUY"  # path: mean_ret=1.054%, n=244, risk_score=6, worst=-0.32%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.457739
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.114677
                and not np.isnan(f.get('dist_from_ema9', np.nan)) and f.get('dist_from_ema9', np.nan) > -1.033344
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.015968
            ):
            return "BUY"  # path: mean_ret=1.444%, n=176, risk_score=7, worst=-0.49%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.192459
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 14.211308
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.791656
                and not np.isnan(f.get('hv_20', np.nan)) and f.get('hv_20', np.nan) <= 2.017181
            ):
            return "BUY"  # path: mean_ret=1.248%, n=531, risk_score=6, worst=-0.49%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.170765
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 14.390732
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.725726
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.233221
            ):
            return "BUY"  # path: mean_ret=1.709%, n=388, risk_score=6, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.108071
                and not np.isnan(f.get('hv_5', np.nan)) and f.get('hv_5', np.nan) <= 1.315906
                and not np.isnan(f.get('atr_21', np.nan)) and f.get('atr_21', np.nan) <= 8.178898
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 14.806511
            ):
            return "BUY"  # path: mean_ret=0.739%, n=174, risk_score=6, worst=-0.49%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.228515
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 14.505838
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.304712
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.064071
            ):
            return "BUY"  # path: mean_ret=1.592%, n=138, risk_score=7, worst=-0.49%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.215190
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.825794
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 23.500000
                and not np.isnan(f.get('consec_green', np.nan)) and f.get('consec_green', np.nan) > 2.500000
            ):
            return "BUY"  # path: mean_ret=2.158%, n=18, risk_score=6, worst=-0.88%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.208421
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.845115
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 23.500000
                and not np.isnan(f.get('consec_green', np.nan)) and f.get('consec_green', np.nan) > 2.500000
            ):
            return "BUY"  # path: mean_ret=2.158%, n=18, risk_score=6, worst=-0.88%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.301716
                and not np.isnan(f.get('rel_candle_size', np.nan)) and f.get('rel_candle_size', np.nan) > 2.012011
                and not np.isnan(f.get('atr_pct', np.nan)) and f.get('atr_pct', np.nan) > 0.868017
                and not np.isnan(f.get('consec_red', np.nan)) and f.get('consec_red', np.nan) <= 0.769942
            ):
            return "BUY"  # path: mean_ret=2.850%, n=11, risk_score=5, worst=-0.07%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.525075
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.119449
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 23.590046
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) <= 0.849595
            ):
            return "BUY"  # path: mean_ret=1.073%, n=421, risk_score=6, worst=-0.17%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.500382
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.156374
                and not np.isnan(f.get('atr_pct', np.nan)) and f.get('atr_pct', np.nan) > 1.039539
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 20.836899
            ):
            return "BUY"  # path: mean_ret=2.231%, n=88, risk_score=6, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.071743
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.374173
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 42.729150
                and not np.isnan(f.get('roc_3', np.nan)) and f.get('roc_3', np.nan) > -8.816112
            ):
            return "BUY"  # path: mean_ret=1.757%, n=143, risk_score=7, worst=-3.55%

        elif (
                not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 18.266629
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.118968
                and not np.isnan(f.get('compression_score', np.nan)) and f.get('compression_score', np.nan) <= 0.495685
                and not np.isnan(f.get('ema_9_above_21', np.nan)) and f.get('ema_9_above_21', np.nan) <= 0.800649
            ):
            return "BUY"  # path: mean_ret=1.334%, n=250, risk_score=9, worst=-0.36%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.251754
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) <= 0.858151
                and not np.isnan(f.get('obv', np.nan)) and f.get('obv', np.nan) > 60015244.000000
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) > 40.500000
            ):
            return "BUY"  # path: mean_ret=0.711%, n=11, risk_score=5, worst=-0.27%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.337800
                and not np.isnan(f.get('compression_score', np.nan)) and f.get('compression_score', np.nan) <= 0.435541
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.172010
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 4.683465
            ):
            return "BUY"  # path: mean_ret=2.256%, n=10, risk_score=7, worst=0.78%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.299267
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 21.176227
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.155155
                and not np.isnan(f.get('atr_pct', np.nan)) and f.get('atr_pct', np.nan) > 1.013949
            ):
            return "BUY"  # path: mean_ret=2.329%, n=123, risk_score=7, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.215190
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.032190
                and not np.isnan(f.get('atr_percentile', np.nan)) and f.get('atr_percentile', np.nan) > 0.088583
                and not np.isnan(f.get('consec_green', np.nan)) and f.get('consec_green', np.nan) > 2.500000
            ):
            return "BUY"  # path: mean_ret=2.246%, n=20, risk_score=6, worst=-0.88%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.185186
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.749700
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.048132
                and not np.isnan(f.get('hv_5', np.nan)) and f.get('hv_5', np.nan) <= 0.574867
            ):
            return "BUY"  # path: mean_ret=1.641%, n=7, risk_score=8, worst=0.29%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.745442
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.127219
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 14.836313
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.367373
            ):
            return "BUY"  # path: mean_ret=1.550%, n=405, risk_score=6, worst=-0.49%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.135042
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 23.131360
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) > 13.954907
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) <= 0.643605
            ):
            return "BUY"  # path: mean_ret=0.587%, n=158, risk_score=6, worst=-0.15%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.747005
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.134370
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 14.763164
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.283655
            ):
            return "BUY"  # path: mean_ret=1.518%, n=438, risk_score=6, worst=-0.49%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.286574
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.913583
                and not np.isnan(f.get('hv_5', np.nan)) and f.get('hv_5', np.nan) <= 3.376076
                and not np.isnan(f.get('hv_20', np.nan)) and f.get('hv_20', np.nan) <= 0.838186
            ):
            return "BUY"  # path: mean_ret=0.766%, n=292, risk_score=5, worst=-0.66%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.215440
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.403635
                and not np.isnan(f.get('consec_green', np.nan)) and f.get('consec_green', np.nan) > 2.500000
                and not np.isnan(f.get('bb_width', np.nan)) and f.get('bb_width', np.nan) > 0.033756
            ):
            return "BUY"  # path: mean_ret=3.031%, n=7, risk_score=6, worst=-0.89%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.557622
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.201782
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 23.705636
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) <= 0.501507
            ):
            return "BUY"  # path: mean_ret=0.726%, n=582, risk_score=5, worst=-0.41%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.501016
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.257230
                and not np.isnan(f.get('stoch_k', np.nan)) and f.get('stoch_k', np.nan) > 72.467794
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.197489
            ):
            return "BUY"  # path: mean_ret=2.580%, n=26, risk_score=8, worst=0.06%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.556468
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.415792
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) > 0.051188
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.121549
            ):
            return "BUY"  # path: mean_ret=1.553%, n=275, risk_score=6, worst=-0.36%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.270496
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 14.901191
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.111696
                and not np.isnan(f.get('atr_pct', np.nan)) and f.get('atr_pct', np.nan) <= 0.629982
            ):
            return "BUY"  # path: mean_ret=0.921%, n=670, risk_score=5, worst=-0.66%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.758703
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.237471
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 22.488874
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) <= 0.530063
            ):
            return "BUY"  # path: mean_ret=0.737%, n=712, risk_score=5, worst=-0.48%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.145446
                and not np.isnan(f.get('ret_2', np.nan)) and f.get('ret_2', np.nan) <= -0.946545
                and not np.isnan(f.get('accel_3', np.nan)) and f.get('accel_3', np.nan) <= -1.960444
                and not np.isnan(f.get('ema_9_50_spread', np.nan)) and f.get('ema_9_50_spread', np.nan) > -5.166297
            ):
            return "BUY"  # path: mean_ret=1.646%, n=120, risk_score=9, worst=-1.51%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.266462
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) <= 0.823565
                and not np.isnan(f.get('obv', np.nan)) and f.get('obv', np.nan) > 66086282.000000
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) > 18.500000
            ):
            return "BUY"  # path: mean_ret=0.712%, n=8, risk_score=5, worst=-0.04%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.500298
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.225079
                and not np.isnan(f.get('atr_pct', np.nan)) and f.get('atr_pct', np.nan) <= 0.458334
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.323465
            ):
            return "BUY"  # path: mean_ret=0.748%, n=582, risk_score=4, worst=-0.66%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.189842
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.707711
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.421303
                and not np.isnan(f.get('consec_green', np.nan)) and f.get('consec_green', np.nan) > 2.500000
            ):
            return "BUY"  # path: mean_ret=2.672%, n=7, risk_score=7, worst=-0.89%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.490182
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.419130
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.191069
                and not np.isnan(f.get('consec_green', np.nan)) and f.get('consec_green', np.nan) > 2.154597
            ):
            return "BUY"  # path: mean_ret=2.672%, n=7, risk_score=7, worst=-0.89%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.370170
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.145184
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.406753
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) <= 1.310389
            ):
            return "BUY"  # path: mean_ret=1.404%, n=473, risk_score=6, worst=-0.49%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.185007
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.096252
                and not np.isnan(f.get('consec_green', np.nan)) and f.get('consec_green', np.nan) > 2.500000
                and not np.isnan(f.get('volume_percentile', np.nan)) and f.get('volume_percentile', np.nan) > 0.812860
            ):
            return "BUY"  # path: mean_ret=2.221%, n=11, risk_score=9, worst=-0.89%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.185614
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.825560
                and not np.isnan(f.get('consec_green', np.nan)) and f.get('consec_green', np.nan) > 2.500000
                and not np.isnan(f.get('bb_width', np.nan)) and f.get('bb_width', np.nan) <= 0.055277
            ):
            return "BUY"  # path: mean_ret=1.879%, n=17, risk_score=7, worst=-0.89%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.215190
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.096252
                and not np.isnan(f.get('consec_green', np.nan)) and f.get('consec_green', np.nan) > 2.500000
                and not np.isnan(f.get('volume_percentile', np.nan)) and f.get('volume_percentile', np.nan) > 0.795585
            ):
            return "BUY"  # path: mean_ret=2.074%, n=16, risk_score=7, worst=-0.89%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.178492
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.250520
                and not np.isnan(f.get('atr_pct', np.nan)) and f.get('atr_pct', np.nan) > 1.007099
                and not np.isnan(f.get('rsi_14_slope', np.nan)) and f.get('rsi_14_slope', np.nan) <= 5.875799
            ):
            return "BUY"  # path: mean_ret=2.042%, n=107, risk_score=10, worst=-0.99%

        elif (
                not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 39.817142
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.917585
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.147998
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.406490
            ):
            return "BUY"  # path: mean_ret=1.478%, n=517, risk_score=6, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.647139
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.781781
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.257564
                and not np.isnan(f.get('atr_pct', np.nan)) and f.get('atr_pct', np.nan) <= 0.804885
            ):
            return "BUY"  # path: mean_ret=1.056%, n=784, risk_score=5, worst=-0.66%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.854451
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.645487
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.161131
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 20.024674
            ):
            return "BUY"  # path: mean_ret=1.474%, n=585, risk_score=6, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.299267
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 21.176227
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.155155
                and not np.isnan(f.get('atr_pct', np.nan)) and f.get('atr_pct', np.nan) <= 1.013949
            ):
            return "BUY"  # path: mean_ret=1.188%, n=577, risk_score=7, worst=-0.49%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.528728
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.209294
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.083607
                and not np.isnan(f.get('bb_width', np.nan)) and f.get('bb_width', np.nan) > 0.019182
            ):
            return "BUY"  # path: mean_ret=1.615%, n=541, risk_score=5, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.794795
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.369749
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.155450
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.570624
            ):
            return "BUY"  # path: mean_ret=1.482%, n=556, risk_score=6, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.185614
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.825560
                and not np.isnan(f.get('consec_green', np.nan)) and f.get('consec_green', np.nan) > 2.500000
                and not np.isnan(f.get('bb_width', np.nan)) and f.get('bb_width', np.nan) > 0.055277
            ):
            return "BUY"  # path: mean_ret=1.900%, n=8, risk_score=7, worst=0.61%

        elif (
                not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 18.266629
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.118968
                and not np.isnan(f.get('compression_score', np.nan)) and f.get('compression_score', np.nan) <= 0.495685
                and not np.isnan(f.get('ema_9_above_21', np.nan)) and f.get('ema_9_above_21', np.nan) > 0.800649
            ):
            return "BUY"  # path: mean_ret=1.408%, n=123, risk_score=9, worst=-0.49%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.546278
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.209199
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.784560
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) > 1.602229
            ):
            return "BUY"  # path: mean_ret=2.359%, n=24, risk_score=5, worst=-0.13%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.417464
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.165427
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) > 0.846600
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.214924
            ):
            return "BUY"  # path: mean_ret=1.964%, n=227, risk_score=5, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.201790
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.898074
                and not np.isnan(f.get('consec_green', np.nan)) and f.get('consec_green', np.nan) > 2.524149
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.317663
            ):
            return "BUY"  # path: mean_ret=2.651%, n=11, risk_score=6, worst=-0.89%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.520971
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.137026
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 23.676181
                and not np.isnan(f.get('ret_2', np.nan)) and f.get('ret_2', np.nan) <= -1.185416
            ):
            return "BUY"  # path: mean_ret=1.589%, n=304, risk_score=7, worst=-0.37%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.812159
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.187130
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 24.591414
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.443532
            ):
            return "BUY"  # path: mean_ret=1.424%, n=721, risk_score=6, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.095735
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.522399
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) > 0.065242
                and not np.isnan(f.get('accel_3', np.nan)) and f.get('accel_3', np.nan) > -3.820164
            ):
            return "BUY"  # path: mean_ret=2.082%, n=92, risk_score=8, worst=-0.36%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.192459
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 14.211308
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.791656
                and not np.isnan(f.get('hv_20', np.nan)) and f.get('hv_20', np.nan) > 2.017181
            ):
            return "BUY"  # path: mean_ret=1.797%, n=223, risk_score=6, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.253225
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 22.829847
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) <= 0.893788
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.470493
            ):
            return "BUY"  # path: mean_ret=1.126%, n=842, risk_score=5, worst=-0.53%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.275861
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 24.357290
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.394278
                and not np.isnan(f.get('consec_red', np.nan)) and f.get('consec_red', np.nan) <= 0.532896
            ):
            return "BUY"  # path: mean_ret=2.020%, n=39, risk_score=6, worst=0.13%

        elif (
                not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 18.184916
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.412190
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.467725
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.193268
            ):
            return "BUY"  # path: mean_ret=1.407%, n=759, risk_score=6, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.459736
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 14.875110
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.190607
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.550774
            ):
            return "BUY"  # path: mean_ret=1.411%, n=746, risk_score=6, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.565664
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.190644
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 22.462448
                and not np.isnan(f.get('minute', np.nan)) and f.get('minute', np.nan) > 18.167995
            ):
            return "BUY"  # path: mean_ret=1.411%, n=746, risk_score=6, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.298929
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 23.461942
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.948766
                and not np.isnan(f.get('atr_pct', np.nan)) and f.get('atr_pct', np.nan) <= 0.651775
            ):
            return "BUY"  # path: mean_ret=0.896%, n=814, risk_score=5, worst=-0.66%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.319486
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 20.198726
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) > 0.318912
                and not np.isnan(f.get('atr_pct', np.nan)) and f.get('atr_pct', np.nan) <= 0.544148
            ):
            return "BUY"  # path: mean_ret=0.801%, n=598, risk_score=4, worst=-0.66%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.546278
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.209199
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.784560
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) <= 1.602229
            ):
            return "BUY"  # path: mean_ret=1.358%, n=812, risk_score=5, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.376396
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.343018
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) > 0.689416
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.238918
            ):
            return "BUY"  # path: mean_ret=1.805%, n=383, risk_score=5, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.215175
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.943668
                and not np.isnan(f.get('consec_green', np.nan)) and f.get('consec_green', np.nan) > 2.500000
                and not np.isnan(f.get('macd_line', np.nan)) and f.get('macd_line', np.nan) <= 0.701029
            ):
            return "BUY"  # path: mean_ret=1.853%, n=27, risk_score=6, worst=-0.89%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.343249
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 22.508130
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.247737
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 2.181778
            ):
            return "BUY"  # path: mean_ret=1.820%, n=290, risk_score=5, worst=-0.81%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.571882
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 24.954226
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.099659
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.832543
            ):
            return "BUY"  # path: mean_ret=1.570%, n=334, risk_score=6, worst=-0.49%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.205632
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.845284
                and not np.isnan(f.get('hv_5', np.nan)) and f.get('hv_5', np.nan) <= 0.475065
                and not np.isnan(f.get('cmf', np.nan)) and f.get('cmf', np.nan) > -0.145368
            ):
            return "BUY"  # path: mean_ret=1.223%, n=17, risk_score=7, worst=0.01%

        elif (
                not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 17.768191
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.204092
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 14.932998
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.895499
            ):
            return "BUY"  # path: mean_ret=1.387%, n=816, risk_score=5, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.417464
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.165427
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) <= 0.846600
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.276297
            ):
            return "BUY"  # path: mean_ret=1.046%, n=909, risk_score=5, worst=-0.90%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.552732
                and not np.isnan(f.get('atr_norm_candle', np.nan)) and f.get('atr_norm_candle', np.nan) <= 2.377904
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.366712
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.149640
            ):
            return "BUY"  # path: mean_ret=1.559%, n=224, risk_score=6, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.616658
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.207835
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 14.807431
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.836696
            ):
            return "BUY"  # path: mean_ret=1.385%, n=830, risk_score=5, worst=-0.99%

        elif (
                not np.isnan(f.get('gap_up_pct', np.nan)) and f.get('gap_up_pct', np.nan) > 0.975461
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.311480
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) <= 2.851209
            ):
            return "BUY"  # path: mean_ret=1.125%, n=97, risk_score=5, worst=-0.31%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.647139
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.781781
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.257564
                and not np.isnan(f.get('atr_pct', np.nan)) and f.get('atr_pct', np.nan) > 0.804885
            ):
            return "BUY"  # path: mean_ret=1.856%, n=336, risk_score=5, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.525075
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.119449
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 23.590046
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) > 0.849595
            ):
            return "BUY"  # path: mean_ret=2.050%, n=178, risk_score=6, worst=-0.49%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) > 0.029412
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.198214
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 24.565426
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.485536
            ):
            return "BUY"  # path: mean_ret=1.383%, n=738, risk_score=6, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.111589
                and not np.isnan(f.get('ema5_slope', np.nan)) and f.get('ema5_slope', np.nan) <= -0.439714
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 43.403218
                and not np.isnan(f.get('atr_percentile', np.nan)) and f.get('atr_percentile', np.nan) > 0.154384
            ):
            return "BUY"  # path: mean_ret=1.502%, n=342, risk_score=7, worst=-1.51%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.287699
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.441326
                and not np.isnan(f.get('atr_pct', np.nan)) and f.get('atr_pct', np.nan) <= 0.558473
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) > 0.118845
            ):
            return "BUY"  # path: mean_ret=0.756%, n=464, risk_score=5, worst=-0.66%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.270496
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 14.901191
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.111696
                and not np.isnan(f.get('atr_pct', np.nan)) and f.get('atr_pct', np.nan) > 0.629982
            ):
            return "BUY"  # path: mean_ret=1.682%, n=529, risk_score=5, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.337730
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.201030
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.637327
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 18.497474
            ):
            return "BUY"  # path: mean_ret=1.662%, n=485, risk_score=5, worst=-1.40%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.381138
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 19.782387
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.320789
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) <= 0.811215
            ):
            return "BUY"  # path: mean_ret=0.986%, n=667, risk_score=4, worst=-0.66%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.374693
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.121986
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 21.565150
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.504040
            ):
            return "BUY"  # path: mean_ret=1.513%, n=456, risk_score=6, worst=-0.49%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.532705
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 20.212102
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.081662
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) <= 1.402501
            ):
            return "BUY"  # path: mean_ret=1.208%, n=81, risk_score=6, worst=0.29%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.286574
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.913583
                and not np.isnan(f.get('hv_5', np.nan)) and f.get('hv_5', np.nan) <= 3.376076
                and not np.isnan(f.get('hv_20', np.nan)) and f.get('hv_20', np.nan) > 0.838186
            ):
            return "BUY"  # path: mean_ret=1.269%, n=661, risk_score=5, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.280579
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 22.465120
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.153679
                and not np.isnan(f.get('consec_red', np.nan)) and f.get('consec_red', np.nan) <= 0.443868
            ):
            return "BUY"  # path: mean_ret=2.378%, n=25, risk_score=6, worst=-0.88%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.215900
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.039163
                and not np.isnan(f.get('consec_green', np.nan)) and f.get('consec_green', np.nan) > 2.500000
                and not np.isnan(f.get('body_to_range', np.nan)) and f.get('body_to_range', np.nan) <= 0.575540
            ):
            return "BUY"  # path: mean_ret=2.131%, n=19, risk_score=6, worst=-0.89%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.215274
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.013502
                and not np.isnan(f.get('consec_green', np.nan)) and f.get('consec_green', np.nan) > 2.500000
                and not np.isnan(f.get('body_to_range', np.nan)) and f.get('body_to_range', np.nan) <= 0.575540
            ):
            return "BUY"  # path: mean_ret=2.131%, n=19, risk_score=6, worst=-0.89%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.184754
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.012190
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 42.500000
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 22.500000
            ):
            return "BUY"  # path: mean_ret=1.570%, n=645, risk_score=6, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.311094
                and not np.isnan(f.get('rel_candle_size', np.nan)) and f.get('rel_candle_size', np.nan) > 1.862558
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 2.391335
                and not np.isnan(f.get('is_green', np.nan)) and f.get('is_green', np.nan) > 0.189628
            ):
            return "BUY"  # path: mean_ret=2.330%, n=47, risk_score=4, worst=-0.80%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.204135
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.815365
                and not np.isnan(f.get('hv_5', np.nan)) and f.get('hv_5', np.nan) <= 0.475065
                and not np.isnan(f.get('atr_21', np.nan)) and f.get('atr_21', np.nan) > 1.058383
            ):
            return "BUY"  # path: mean_ret=0.907%, n=6, risk_score=5, worst=-0.05%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.105494
                and not np.isnan(f.get('ret_2', np.nan)) and f.get('ret_2', np.nan) <= -1.199425
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 42.500000
            ):
            return "BUY"  # path: mean_ret=1.536%, n=314, risk_score=6, worst=-3.55%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.856080
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.143358
                and not np.isnan(f.get('dist_from_ema9', np.nan)) and f.get('dist_from_ema9', np.nan) > -0.815798
                and not np.isnan(f.get('cci_20', np.nan)) and f.get('cci_20', np.nan) > 81.667117
            ):
            return "BUY"  # path: mean_ret=1.387%, n=127, risk_score=9, worst=-0.49%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.220154
                and not np.isnan(f.get('obv', np.nan)) and f.get('obv', np.nan) <= 39481710.954434
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 14.261101
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.448020
            ):
            return "BUY"  # path: mean_ret=1.355%, n=880, risk_score=5, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.368150
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 22.861726
                and not np.isnan(f.get('hv_20', np.nan)) and f.get('hv_20', np.nan) > 1.142759
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.163803
            ):
            return "BUY"  # path: mean_ret=1.601%, n=537, risk_score=6, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.630165
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.076550
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.772281
                and not np.isnan(f.get('cmf', np.nan)) and f.get('cmf', np.nan) > -0.434603
            ):
            return "BUY"  # path: mean_ret=2.055%, n=90, risk_score=8, worst=-5.07%

        elif (
                not np.isnan(f.get('volume_ratio', np.nan)) and f.get('volume_ratio', np.nan) > 1.234892
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.346303
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 23.156691
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.133373
            ):
            return "BUY"  # path: mean_ret=1.496%, n=416, risk_score=8, worst=-0.49%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.807842
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.053807
                and not np.isnan(f.get('ema12_slope', np.nan)) and f.get('ema12_slope', np.nan) <= -0.293610
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 42.868626
            ):
            return "BUY"  # path: mean_ret=1.320%, n=114, risk_score=7, worst=-3.55%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.270703
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.570209
                and not np.isnan(f.get('stoch_d', np.nan)) and f.get('stoch_d', np.nan) > 60.758401
            ):
            return "BUY"  # path: mean_ret=1.845%, n=135, risk_score=4, worst=-0.68%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.552732
                and not np.isnan(f.get('atr_norm_candle', np.nan)) and f.get('atr_norm_candle', np.nan) > 2.377904
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.088692
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) <= 0.634204
            ):
            return "BUY"  # path: mean_ret=1.158%, n=128, risk_score=6, worst=-1.40%

        elif (
                not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 17.666028
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.911948
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.229599
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 20.585824
            ):
            return "BUY"  # path: mean_ret=1.349%, n=937, risk_score=5, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.186018
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.011422
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 42.500000
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 23.500000
            ):
            return "BUY"  # path: mean_ret=1.546%, n=675, risk_score=6, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.375164
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 2.095816
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.205977
                and not np.isnan(f.get('consec_red', np.nan)) and f.get('consec_red', np.nan) <= 0.723198
            ):
            return "BUY"  # path: mean_ret=2.359%, n=6, risk_score=6, worst=-0.51%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.185193
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.011422
                and not np.isnan(f.get('consec_green', np.nan)) and f.get('consec_green', np.nan) > 2.500000
                and not np.isnan(f.get('trend_persistence', np.nan)) and f.get('trend_persistence', np.nan) <= 0.500000
            ):
            return "BUY"  # path: mean_ret=1.871%, n=9, risk_score=11, worst=-0.89%

        elif (
                not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 18.288873
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.963221
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.178527
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 21.000736
            ):
            return "BUY"  # path: mean_ret=1.358%, n=859, risk_score=6, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.270589
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.048521
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.125069
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 18.500000
            ):
            return "BUY"  # path: mean_ret=1.606%, n=561, risk_score=7, worst=-1.08%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.170765
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 14.390732
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) > 13.725726
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) <= 0.588477
            ):
            return "BUY"  # path: mean_ret=0.477%, n=325, risk_score=6, worst=-0.35%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.557785
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 14.734000
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.233426
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.134475
            ):
            return "BUY"  # path: mean_ret=1.347%, n=951, risk_score=5, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.269128
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 23.000000
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.104687
            ):
            return "BUY"  # path: mean_ret=1.367%, n=482, risk_score=6, worst=-0.49%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.447853
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 23.000000
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.104687
            ):
            return "BUY"  # path: mean_ret=1.367%, n=482, risk_score=5, worst=-0.49%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.184989
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.777540
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 41.500000
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 22.500000
            ):
            return "BUY"  # path: mean_ret=1.448%, n=809, risk_score=6, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.190476
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.816062
                and not np.isnan(f.get('hv_5', np.nan)) and f.get('hv_5', np.nan) <= 0.479547
                and not np.isnan(f.get('ema_20', np.nan)) and f.get('ema_20', np.nan) > 110.753391
            ):
            return "BUY"  # path: mean_ret=1.093%, n=18, risk_score=6, worst=-0.05%

        elif (
                not np.isnan(f.get('gap_up_pct', np.nan)) and f.get('gap_up_pct', np.nan) > 0.975461
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.420852
                and not np.isnan(f.get('atr_pct', np.nan)) and f.get('atr_pct', np.nan) > 1.483416
            ):
            return "BUY"  # path: mean_ret=2.244%, n=15, risk_score=5, worst=0.06%

        elif (
                not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 41.987552
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.875038
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.086418
                and not np.isnan(f.get('hv_5', np.nan)) and f.get('hv_5', np.nan) > 1.618487
            ):
            return "BUY"  # path: mean_ret=1.546%, n=384, risk_score=6, worst=-2.86%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) > 0.059756
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.229751
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 14.108514
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.358259
            ):
            return "BUY"  # path: mean_ret=1.305%, n=816, risk_score=5, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.205878
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.016505
                and not np.isnan(f.get('volume_ratio', np.nan)) and f.get('volume_ratio', np.nan) <= 3.775799
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 23.500000
            ):
            return "BUY"  # path: mean_ret=1.488%, n=696, risk_score=5, worst=-1.03%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) > 0.104669
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.167702
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 29.702089
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 21.746332
            ):
            return "BUY"  # path: mean_ret=1.231%, n=395, risk_score=6, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.278599
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 24.000000
                and not np.isnan(f.get('accel_3', np.nan)) and f.get('accel_3', np.nan) <= -2.279068
            ):
            return "BUY"  # path: mean_ret=1.377%, n=212, risk_score=6, worst=-0.85%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.563902
                and not np.isnan(f.get('volume_percentile', np.nan)) and f.get('volume_percentile', np.nan) <= 0.701118
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 14.189567
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.091266
            ):
            return "BUY"  # path: mean_ret=0.927%, n=138, risk_score=6, worst=-0.02%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.858157
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.674610
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.248113
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.313129
            ):
            return "BUY"  # path: mean_ret=1.310%, n=1055, risk_score=5, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.367803
                and not np.isnan(f.get('atr_norm_candle', np.nan)) and f.get('atr_norm_candle', np.nan) > 1.421319
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 23.062138
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.186445
            ):
            return "BUY"  # path: mean_ret=1.265%, n=807, risk_score=6, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.228515
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 14.505838
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.304712
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) > 0.064071
            ):
            return "BUY"  # path: mean_ret=1.311%, n=789, risk_score=5, worst=-0.99%

        elif (
                not np.isnan(f.get('bb_pct', np.nan)) and f.get('bb_pct', np.nan) > 0.060753
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 41.809363
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.097644
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.718823
            ):
            return "BUY"  # path: mean_ret=1.392%, n=337, risk_score=7, worst=-1.27%

        elif (
                not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 16.350324
                and not np.isnan(f.get('volume_percentile', np.nan)) and f.get('volume_percentile', np.nan) <= 0.796904
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.092152
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.082727
            ):
            return "BUY"  # path: mean_ret=1.750%, n=97, risk_score=6, worst=-0.27%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.184859
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.744884
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 40.500000
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 23.500000
            ):
            return "BUY"  # path: mean_ret=1.398%, n=899, risk_score=6, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.187251
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.801724
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) > 0.050700
                and not np.isnan(f.get('consec_green', np.nan)) and f.get('consec_green', np.nan) > 2.500000
            ):
            return "BUY"  # path: mean_ret=1.732%, n=27, risk_score=7, worst=-1.18%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.215190
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.837572
                and not np.isnan(f.get('volume_ratio', np.nan)) and f.get('volume_ratio', np.nan) > 5.721458
                and not np.isnan(f.get('volume_percentile', np.nan)) and f.get('volume_percentile', np.nan) <= 0.977927
            ):
            return "BUY"  # path: mean_ret=3.949%, n=7, risk_score=9, worst=0.19%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.185574
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.744884
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 18.500000
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 23.500000
            ):
            return "BUY"  # path: mean_ret=1.394%, n=905, risk_score=6, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.439856
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 14.003494
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.174075
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) <= 0.717216
            ):
            return "BUY"  # path: mean_ret=0.827%, n=826, risk_score=6, worst=-0.35%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.185641
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.679738
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 41.500000
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 22.500000
            ):
            return "BUY"  # path: mean_ret=1.395%, n=885, risk_score=6, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.184972
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.704415
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 41.500000
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 23.500000
            ):
            return "BUY"  # path: mean_ret=1.380%, n=933, risk_score=6, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.184972
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.762533
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) > 0.048062
                and not np.isnan(f.get('consec_green', np.nan)) and f.get('consec_green', np.nan) > 2.500000
            ):
            return "BUY"  # path: mean_ret=1.710%, n=28, risk_score=7, worst=-1.18%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.049298
                and not np.isnan(f.get('atr_pct', np.nan)) and f.get('atr_pct', np.nan) > 0.700554
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 18.500000
            ):
            return "BUY"  # path: mean_ret=1.449%, n=171, risk_score=5, worst=-0.49%

        elif (
                not np.isnan(f.get('gap_up_pct', np.nan)) and f.get('gap_up_pct', np.nan) > 0.975461
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.311480
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 2.851209
            ):
            return "BUY"  # path: mean_ret=2.136%, n=36, risk_score=5, worst=-0.49%

        elif (
                not np.isnan(f.get('gap_up_pct', np.nan)) and f.get('gap_up_pct', np.nan) > 1.266588
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.595896
                and not np.isnan(f.get('cmf', np.nan)) and f.get('cmf', np.nan) > 0.183522
            ):
            return "BUY"  # path: mean_ret=1.763%, n=36, risk_score=7, worst=-0.48%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.187251
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.744590
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 18.500000
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 23.500000
            ):
            return "BUY"  # path: mean_ret=1.384%, n=915, risk_score=6, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.215262
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.057513
                and not np.isnan(f.get('consec_green', np.nan)) and f.get('consec_green', np.nan) > 2.500000
                and not np.isnan(f.get('candle_body_pct', np.nan)) and f.get('candle_body_pct', np.nan) <= 57.554028
            ):
            return "BUY"  # path: mean_ret=2.004%, n=18, risk_score=7, worst=-0.89%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.243420
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.992768
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.772216
                and not np.isnan(f.get('stoch_d', np.nan)) and f.get('stoch_d', np.nan) > 52.301881
            ):
            return "BUY"  # path: mean_ret=1.965%, n=147, risk_score=5, worst=-4.60%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.270703
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 22.500000
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.372338
            ):
            return "BUY"  # path: mean_ret=1.533%, n=720, risk_score=4, worst=-1.03%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.298929
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 23.461942
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.948766
                and not np.isnan(f.get('atr_pct', np.nan)) and f.get('atr_pct', np.nan) > 0.651775
            ):
            return "BUY"  # path: mean_ret=1.593%, n=580, risk_score=5, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.270703
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 23.500000
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.372338
            ):
            return "BUY"  # path: mean_ret=1.527%, n=744, risk_score=4, worst=-1.03%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.215175
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.054510
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 42.500000
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 22.500000
            ):
            return "BUY"  # path: mean_ret=1.518%, n=756, risk_score=5, worst=-1.03%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.376396
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.343018
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) <= 0.689416
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) <= 0.451376
            ):
            return "BUY"  # path: mean_ret=0.657%, n=717, risk_score=4, worst=-0.82%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.823121
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.264425
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 26.740535
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.963293
            ):
            return "BUY"  # path: mean_ret=1.275%, n=1159, risk_score=5, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.286574
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.913583
                and not np.isnan(f.get('hv_5', np.nan)) and f.get('hv_5', np.nan) > 3.376076
                and not np.isnan(f.get('is_green', np.nan)) and f.get('is_green', np.nan) > 0.107271
            ):
            return "BUY"  # path: mean_ret=1.902%, n=74, risk_score=5, worst=-0.74%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.368150
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 22.861726
                and not np.isnan(f.get('hv_20', np.nan)) and f.get('hv_20', np.nan) <= 1.142759
                and not np.isnan(f.get('minute', np.nan)) and f.get('minute', np.nan) > 18.371447
            ):
            return "BUY"  # path: mean_ret=0.735%, n=796, risk_score=4, worst=-1.00%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.402456
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.199045
                and not np.isnan(f.get('ema5_slope', np.nan)) and f.get('ema5_slope', np.nan) > -0.217666
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 21.355488
            ):
            return "BUY"  # path: mean_ret=1.227%, n=436, risk_score=7, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.172369
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.812452
                and not np.isnan(f.get('hv_5', np.nan)) and f.get('hv_5', np.nan) <= 0.476941
                and not np.isnan(f.get('cmf', np.nan)) and f.get('cmf', np.nan) > -0.174126
            ):
            return "BUY"  # path: mean_ret=1.108%, n=14, risk_score=8, worst=0.01%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.184980
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) <= 0.801467
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 18.500000
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 24.500000
            ):
            return "BUY"  # path: mean_ret=0.589%, n=581, risk_score=6, worst=-0.71%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.269128
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 24.000000
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.104687
            ):
            return "BUY"  # path: mean_ret=1.269%, n=554, risk_score=6, worst=-0.49%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.275861
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 24.357290
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.394278
                and not np.isnan(f.get('consec_red', np.nan)) and f.get('consec_red', np.nan) > 0.532896
            ):
            return "BUY"  # path: mean_ret=1.222%, n=1193, risk_score=6, worst=-0.99%

        elif (
                not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 17.874445
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.274959
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 25.875065
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.840651
            ):
            return "BUY"  # path: mean_ret=1.252%, n=1226, risk_score=5, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.038014
                and not np.isnan(f.get('ema12_slope', np.nan)) and f.get('ema12_slope', np.nan) <= -0.202223
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 18.796113
                and not np.isnan(f.get('dist_from_ema9', np.nan)) and f.get('dist_from_ema9', np.nan) > -1.454414
            ):
            return "BUY"  # path: mean_ret=1.270%, n=62, risk_score=8, worst=0.09%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.215440
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.403635
                and not np.isnan(f.get('consec_green', np.nan)) and f.get('consec_green', np.nan) <= 2.500000
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 39.500000
            ):
            return "BUY"  # path: mean_ret=1.507%, n=702, risk_score=6, worst=-1.86%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.176189
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 14.385687
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) > 13.256680
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) <= 0.835505
            ):
            return "BUY"  # path: mean_ret=0.542%, n=486, risk_score=6, worst=-0.35%

        elif (
                not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 17.971331
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.205870
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 14.966715
                and not np.isnan(f.get('atr_pct', np.nan)) and f.get('atr_pct', np.nan) <= 0.624342
            ):
            return "BUY"  # path: mean_ret=0.716%, n=969, risk_score=5, worst=-0.62%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.585735
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 24.083664
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.292111
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.275532
            ):
            return "BUY"  # path: mean_ret=1.248%, n=1232, risk_score=5, worst=-0.99%

        elif (
                not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 16.350324
                and not np.isnan(f.get('volume_percentile', np.nan)) and f.get('volume_percentile', np.nan) <= 0.796904
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.092152
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) <= 1.082727
            ):
            return "BUY"  # path: mean_ret=0.650%, n=678, risk_score=6, worst=-0.49%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.278599
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 24.000000
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.072284
            ):
            return "BUY"  # path: mean_ret=1.332%, n=279, risk_score=6, worst=-0.49%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.269130
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 23.500000
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.099345
                and not np.isnan(f.get('consec_red', np.nan)) and f.get('consec_red', np.nan) <= 0.500000
            ):
            return "BUY"  # path: mean_ret=2.388%, n=22, risk_score=6, worst=-1.17%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.187614
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.600365
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 18.500000
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 23.500000
            ):
            return "BUY"  # path: mean_ret=1.302%, n=1046, risk_score=6, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.205821
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) <= 1.054510
                and not np.isnan(f.get('obv', np.nan)) and f.get('obv', np.nan) <= 42613538.000000
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 23.500000
            ):
            return "BUY"  # path: mean_ret=0.722%, n=761, risk_score=5, worst=-1.08%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.205821
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.658472
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 18.500000
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 22.500000
            ):
            return "BUY"  # path: mean_ret=1.335%, n=1043, risk_score=5, worst=-1.03%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.184980
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.801467
                and not np.isnan(f.get('consec_green', np.nan)) and f.get('consec_green', np.nan) > 2.500000
                and not np.isnan(f.get('stoch_k', np.nan)) and f.get('stoch_k', np.nan) <= 83.829502
            ):
            return "BUY"  # path: mean_ret=1.575%, n=23, risk_score=8, worst=-1.18%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.187252
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.576708
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 18.500000
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 23.500000
            ):
            return "BUY"  # path: mean_ret=1.288%, n=1063, risk_score=6, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.185186
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.749700
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) > 0.048132
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.664082
            ):
            return "BUY"  # path: mean_ret=1.716%, n=385, risk_score=6, worst=-3.48%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.600505
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.771319
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.078953
                and not np.isnan(f.get('atr_percentile', np.nan)) and f.get('atr_percentile', np.nan) > 0.172203
            ):
            return "BUY"  # path: mean_ret=2.108%, n=97, risk_score=6, worst=-5.07%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.253225
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 22.829847
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) > 0.893788
                and not np.isnan(f.get('consec_red', np.nan)) and f.get('consec_red', np.nan) <= 0.144657
            ):
            return "BUY"  # path: mean_ret=2.592%, n=11, risk_score=6, worst=-0.88%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) > 0.067602
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.235985
                and not np.isnan(f.get('bb_pct', np.nan)) and f.get('bb_pct', np.nan) > 0.822346
                and not np.isnan(f.get('consec_red', np.nan)) and f.get('consec_red', np.nan) <= 0.252131
            ):
            return "BUY"  # path: mean_ret=2.030%, n=24, risk_score=8, worst=-0.88%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.186099
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.554877
                and not np.isnan(f.get('rsi_7_slope', np.nan)) and f.get('rsi_7_slope', np.nan) > -65.949272
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 23.500000
            ):
            return "BUY"  # path: mean_ret=1.283%, n=1066, risk_score=6, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.105494
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) <= 0.792610
                and not np.isnan(f.get('atr_7', np.nan)) and f.get('atr_7', np.nan) > 3.179267
            ):
            return "BUY"  # path: mean_ret=0.884%, n=87, risk_score=5, worst=-0.41%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.714063
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.289429
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 27.836560
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.221339
            ):
            return "BUY"  # path: mean_ret=1.213%, n=1328, risk_score=5, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.215190
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.825794
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 23.500000
                and not np.isnan(f.get('consec_green', np.nan)) and f.get('consec_green', np.nan) <= 2.500000
            ):
            return "BUY"  # path: mean_ret=1.357%, n=993, risk_score=6, worst=-1.08%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.278387
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.857792
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.133692
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 41.500000
            ):
            return "BUY"  # path: mean_ret=1.415%, n=878, risk_score=7, worst=-3.78%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.439856
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 14.003494
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.174075
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) > 0.717216
            ):
            return "BUY"  # path: mean_ret=1.511%, n=510, risk_score=6, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.280579
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 22.465120
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.153679
                and not np.isnan(f.get('consec_red', np.nan)) and f.get('consec_red', np.nan) > 0.443868
            ):
            return "BUY"  # path: mean_ret=1.413%, n=936, risk_score=6, worst=-1.04%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.286574
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.913583
                and not np.isnan(f.get('hv_5', np.nan)) and f.get('hv_5', np.nan) > 3.376076
                and not np.isnan(f.get('is_green', np.nan)) and f.get('is_green', np.nan) <= 0.107271
            ):
            return "BUY"  # path: mean_ret=1.407%, n=281, risk_score=5, worst=-0.85%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.863972
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.146996
                and not np.isnan(f.get('ret_1', np.nan)) and f.get('ret_1', np.nan) > -0.559540
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 21.897002
            ):
            return "BUY"  # path: mean_ret=1.310%, n=282, risk_score=7, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.714621
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.070965
                and not np.isnan(f.get('dist_prev_close', np.nan)) and f.get('dist_prev_close', np.nan) <= -1.506826
                and not np.isnan(f.get('roc_3', np.nan)) and f.get('roc_3', np.nan) > -1.560925
            ):
            return "BUY"  # path: mean_ret=2.338%, n=12, risk_score=7, worst=0.41%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.215169
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.724404
                and not np.isnan(f.get('volume_ratio', np.nan)) and f.get('volume_ratio', np.nan) > 5.641571
                and not np.isnan(f.get('volume_percentile', np.nan)) and f.get('volume_percentile', np.nan) <= 0.976967
            ):
            return "BUY"  # path: mean_ret=3.594%, n=8, risk_score=9, worst=0.19%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.267898
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.823757
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.137242
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 41.500000
            ):
            return "BUY"  # path: mean_ret=1.375%, n=975, risk_score=7, worst=-3.78%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.319486
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 20.198726
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) > 0.318912
                and not np.isnan(f.get('atr_pct', np.nan)) and f.get('atr_pct', np.nan) > 0.544148
            ):
            return "BUY"  # path: mean_ret=1.449%, n=824, risk_score=4, worst=-1.72%

        elif (
                not np.isnan(f.get('atr_percentile', np.nan)) and f.get('atr_percentile', np.nan) > 0.251362
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.909161
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.057127
                and not np.isnan(f.get('ret_1', np.nan)) and f.get('ret_1', np.nan) <= -1.626278
            ):
            return "BUY"  # path: mean_ret=1.885%, n=39, risk_score=7, worst=-3.03%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.610818
                and not np.isnan(f.get('compression_score', np.nan)) and f.get('compression_score', np.nan) <= 0.795046
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.186987
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 14.475948
            ):
            return "BUY"  # path: mean_ret=1.191%, n=1062, risk_score=7, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.186054
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.599289
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.593572
                and not np.isnan(f.get('macd_signal', np.nan)) and f.get('macd_signal', np.nan) > -9.122427
            ):
            return "BUY"  # path: mean_ret=1.649%, n=469, risk_score=6, worst=-5.07%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.072619
                and not np.isnan(f.get('roc_5', np.nan)) and f.get('roc_5', np.nan) <= -1.259694
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 18.500000
            ):
            return "BUY"  # path: mean_ret=1.307%, n=343, risk_score=6, worst=-0.74%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.217078
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.087856
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) <= 1.901267
                and not np.isnan(f.get('consec_green', np.nan)) and f.get('consec_green', np.nan) > 2.500000
            ):
            return "BUY"  # path: mean_ret=1.831%, n=14, risk_score=6, worst=-0.89%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.381138
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 19.782387
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.320789
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) > 0.811215
            ):
            return "BUY"  # path: mean_ret=1.678%, n=236, risk_score=4, worst=-1.72%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.059756
                and not np.isnan(f.get('atr_pct', np.nan)) and f.get('atr_pct', np.nan) > 0.745928
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) > 38.701579
                and not np.isnan(f.get('hv_5', np.nan)) and f.get('hv_5', np.nan) <= 0.968097
            ):
            return "BUY"  # path: mean_ret=1.541%, n=9, risk_score=6, worst=-0.36%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) > 0.104876
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.237831
                and not np.isnan(f.get('mean_rev_score', np.nan)) and f.get('mean_rev_score', np.nan) <= -1.681216
                and not np.isnan(f.get('consec_green', np.nan)) and f.get('consec_green', np.nan) <= 2.582965
            ):
            return "BUY"  # path: mean_ret=1.211%, n=106, risk_score=7, worst=-0.25%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.270703
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 23.500000
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.120928
            ):
            return "BUY"  # path: mean_ret=1.402%, n=984, risk_score=4, worst=-1.31%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.287699
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.441326
                and not np.isnan(f.get('atr_pct', np.nan)) and f.get('atr_pct', np.nan) > 0.558473
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) > 0.133347
            ):
            return "BUY"  # path: mean_ret=1.365%, n=451, risk_score=5, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.758703
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.237471
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 22.488874
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) > 0.530063
            ):
            return "BUY"  # path: mean_ret=1.455%, n=873, risk_score=5, worst=-1.03%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.809892
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.325905
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.348527
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) <= 2.291855
            ):
            return "BUY"  # path: mean_ret=0.996%, n=1255, risk_score=4, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.809892
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.325905
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.348527
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 2.291855
            ):
            return "BUY"  # path: mean_ret=1.596%, n=350, risk_score=4, worst=-1.72%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.178492
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.250520
                and not np.isnan(f.get('atr_pct', np.nan)) and f.get('atr_pct', np.nan) > 1.007099
                and not np.isnan(f.get('rsi_14_slope', np.nan)) and f.get('rsi_14_slope', np.nan) > 5.875799
            ):
            return "BUY"  # path: mean_ret=2.054%, n=14, risk_score=6, worst=0.06%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.375164
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 2.095816
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.205977
                and not np.isnan(f.get('consec_red', np.nan)) and f.get('consec_red', np.nan) > 0.723198
            ):
            return "BUY"  # path: mean_ret=1.781%, n=305, risk_score=6, worst=-5.07%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.238967
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.708484
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.709070
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 22.500000
            ):
            return "BUY"  # path: mean_ret=1.290%, n=1228, risk_score=5, worst=-1.03%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.557622
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.201782
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 23.705636
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) > 0.501507
            ):
            return "BUY"  # path: mean_ret=1.368%, n=890, risk_score=5, worst=-1.08%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.121204
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) > 0.867061
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 17.921181
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) > 15.460836
            ):
            return "BUY"  # path: mean_ret=1.207%, n=373, risk_score=6, worst=-0.66%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) > 0.136956
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.431690
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.171820
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.255307
            ):
            return "BUY"  # path: mean_ret=1.153%, n=654, risk_score=5, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.269130
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 23.500000
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.099345
                and not np.isnan(f.get('consec_red', np.nan)) and f.get('consec_red', np.nan) > 0.500000
            ):
            return "BUY"  # path: mean_ret=1.376%, n=987, risk_score=6, worst=-1.31%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.329539
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 25.590565
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.788507
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.713187
            ):
            return "BUY"  # path: mean_ret=1.497%, n=548, risk_score=4, worst=-1.72%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.552732
                and not np.isnan(f.get('atr_norm_candle', np.nan)) and f.get('atr_norm_candle', np.nan) > 2.377904
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.088692
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) > 0.634204
            ):
            return "BUY"  # path: mean_ret=1.808%, n=60, risk_score=6, worst=-5.07%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.280579
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 22.465120
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) <= 1.153679
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.116122
            ):
            return "BUY"  # path: mean_ret=0.995%, n=216, risk_score=7, worst=-0.17%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.171160
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.692901
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) > 0.098190
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 14.622088
            ):
            return "BUY"  # path: mean_ret=1.209%, n=529, risk_score=6, worst=-0.99%

        elif (
                not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 38.137902
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.937298
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.253298
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 21.959517
            ):
            return "BUY"  # path: mean_ret=1.195%, n=1455, risk_score=5, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.215190
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) <= 0.724413
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.103497
                and not np.isnan(f.get('atr_7', np.nan)) and f.get('atr_7', np.nan) > 3.256989
            ):
            return "BUY"  # path: mean_ret=0.883%, n=71, risk_score=7, worst=-0.41%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.609705
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.309840
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 24.578176
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.783240
            ):
            return "BUY"  # path: mean_ret=1.151%, n=1487, risk_score=4, worst=-1.72%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.185193
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.011422
                and not np.isnan(f.get('consec_green', np.nan)) and f.get('consec_green', np.nan) <= 2.500000
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 18.500000
            ):
            return "BUY"  # path: mean_ret=1.357%, n=1072, risk_score=7, worst=-1.86%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.626765
                and not np.isnan(f.get('bb_pct', np.nan)) and f.get('bb_pct', np.nan) > 0.853451
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.304547
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 17.649322
            ):
            return "BUY"  # path: mean_ret=1.234%, n=412, risk_score=7, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) > 0.104876
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.237831
                and not np.isnan(f.get('mean_rev_score', np.nan)) and f.get('mean_rev_score', np.nan) <= -1.681216
                and not np.isnan(f.get('consec_green', np.nan)) and f.get('consec_green', np.nan) > 2.582965
            ):
            return "BUY"  # path: mean_ret=2.017%, n=19, risk_score=7, worst=-0.88%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.251754
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.858151
                and not np.isnan(f.get('stoch_k', np.nan)) and f.get('stoch_k', np.nan) > 79.048161
                and not np.isnan(f.get('rsi_7', np.nan)) and f.get('rsi_7', np.nan) > 56.600735
            ):
            return "BUY"  # path: mean_ret=1.717%, n=68, risk_score=7, worst=-4.21%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.187846
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.599448
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.603804
                and not np.isnan(f.get('ema_50', np.nan)) and f.get('ema_50', np.nan) > 1.485255
            ):
            return "BUY"  # path: mean_ret=1.646%, n=476, risk_score=6, worst=-5.07%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.188033
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.489618
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 23.000000
            ):
            return "BUY"  # path: mean_ret=1.163%, n=1286, risk_score=5, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.762514
                and not np.isnan(f.get('ema12_slope', np.nan)) and f.get('ema12_slope', np.nan) <= 0.103698
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.095562
                and not np.isnan(f.get('ret_3', np.nan)) and f.get('ret_3', np.nan) <= -1.668663
            ):
            return "BUY"  # path: mean_ret=1.548%, n=219, risk_score=8, worst=-5.05%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.184874
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.691705
                and not np.isnan(f.get('hv_5', np.nan)) and f.get('hv_5', np.nan) > 0.475065
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.626883
            ):
            return "BUY"  # path: mean_ret=1.703%, n=456, risk_score=6, worst=-5.07%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.520971
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.137026
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 23.676181
                and not np.isnan(f.get('ret_2', np.nan)) and f.get('ret_2', np.nan) > -1.185416
            ):
            return "BUY"  # path: mean_ret=1.089%, n=440, risk_score=7, worst=-0.49%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.240499
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.812356
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) > 0.048062
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 22.500000
            ):
            return "BUY"  # path: mean_ret=1.301%, n=1028, risk_score=5, worst=-1.03%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.184859
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.582647
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.603804
                and not np.isnan(f.get('consec_green', np.nan)) and f.get('consec_green', np.nan) <= 2.500000
            ):
            return "BUY"  # path: mean_ret=1.672%, n=467, risk_score=7, worst=-5.07%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.185269
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.005469
                and not np.isnan(f.get('consec_green', np.nan)) and f.get('consec_green', np.nan) <= 2.500000
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.603233
            ):
            return "BUY"  # path: mean_ret=1.672%, n=467, risk_score=7, worst=-5.07%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.704755
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.489683
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.145559
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 24.925962
            ):
            return "BUY"  # path: mean_ret=1.157%, n=982, risk_score=6, worst=-0.97%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.603095
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.074308
                and not np.isnan(f.get('ema5_slope', np.nan)) and f.get('ema5_slope', np.nan) > -0.239002
                and not np.isnan(f.get('obv', np.nan)) and f.get('obv', np.nan) > 50565170.674098
            ):
            return "BUY"  # path: mean_ret=0.931%, n=9, risk_score=7, worst=0.03%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) > 0.108071
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.389404
                and not np.isnan(f.get('cci_20', np.nan)) and f.get('cci_20', np.nan) > 147.118013
                and not np.isnan(f.get('breakout_score', np.nan)) and f.get('breakout_score', np.nan) > 1.831375
            ):
            return "BUY"  # path: mean_ret=2.162%, n=46, risk_score=11, worst=-1.08%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.286623
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 24.728486
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.960302
                and not np.isnan(f.get('consec_red', np.nan)) and f.get('consec_red', np.nan) <= 0.426587
            ):
            return "BUY"  # path: mean_ret=1.855%, n=39, risk_score=6, worst=-1.17%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.248991
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.837842
                and not np.isnan(f.get('hv_5', np.nan)) and f.get('hv_5', np.nan) > 0.475065
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 23.500000
            ):
            return "BUY"  # path: mean_ret=1.295%, n=1218, risk_score=5, worst=-1.31%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.217078
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.087856
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.901267
                and not np.isnan(f.get('macd_signal', np.nan)) and f.get('macd_signal', np.nan) > -9.122427
            ):
            return "BUY"  # path: mean_ret=1.686%, n=395, risk_score=5, worst=-5.07%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.627947
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.223194
                and not np.isnan(f.get('compression_score', np.nan)) and f.get('compression_score', np.nan) <= 0.882097
                and not np.isnan(f.get('hv_5', np.nan)) and f.get('hv_5', np.nan) > 3.557244
            ):
            return "BUY"  # path: mean_ret=1.460%, n=403, risk_score=6, worst=-1.55%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.136956
                and not np.isnan(f.get('hv_5', np.nan)) and f.get('hv_5', np.nan) > 1.396229
                and not np.isnan(f.get('rel_candle_size', np.nan)) and f.get('rel_candle_size', np.nan) <= 2.355903
                and not np.isnan(f.get('ret_1', np.nan)) and f.get('ret_1', np.nan) <= -2.022767
            ):
            return "BUY"  # path: mean_ret=2.350%, n=12, risk_score=7, worst=-3.03%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.271027
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 26.543697
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 23.140044
                and not np.isnan(f.get('atr_pct', np.nan)) and f.get('atr_pct', np.nan) <= 0.692593
            ):
            return "BUY"  # path: mean_ret=0.713%, n=1446, risk_score=5, worst=-0.72%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.098639
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) <= 1.242526
                and not np.isnan(f.get('atr_pct', np.nan)) and f.get('atr_pct', np.nan) > 1.081424
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 39.684247
            ):
            return "BUY"  # path: mean_ret=1.261%, n=194, risk_score=6, worst=-0.74%

        elif (
                not np.isnan(f.get('gap_up_pct', np.nan)) and f.get('gap_up_pct', np.nan) > 0.975461
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.420852
                and not np.isnan(f.get('accel_3', np.nan)) and f.get('accel_3', np.nan) > 2.345077
            ):
            return "BUY"  # path: mean_ret=1.782%, n=59, risk_score=7, worst=-1.08%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.184859
                and not np.isnan(f.get('compression_score', np.nan)) and f.get('compression_score', np.nan) <= 0.531778
                and not np.isnan(f.get('hv_5', np.nan)) and f.get('hv_5', np.nan) > 1.821920
                and not np.isnan(f.get('consec_red', np.nan)) and f.get('consec_red', np.nan) > 0.111014
            ):
            return "BUY"  # path: mean_ret=1.456%, n=441, risk_score=8, worst=-5.05%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.215190
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.096252
                and not np.isnan(f.get('consec_green', np.nan)) and f.get('consec_green', np.nan) > 2.500000
                and not np.isnan(f.get('volume_percentile', np.nan)) and f.get('volume_percentile', np.nan) <= 0.795585
            ):
            return "BUY"  # path: mean_ret=2.507%, n=5, risk_score=6, worst=-0.88%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.215190
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.091468
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 2.081342
                and not np.isnan(f.get('bb_expansion', np.nan)) and f.get('bb_expansion', np.nan) > -0.006911
            ):
            return "BUY"  # path: mean_ret=1.774%, n=322, risk_score=6, worst=-5.07%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.526640
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.387635
                and not np.isnan(f.get('bb_pct', np.nan)) and f.get('bb_pct', np.nan) > 0.874866
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.304059
            ):
            return "BUY"  # path: mean_ret=1.257%, n=377, risk_score=7, worst=-1.58%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.216075
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.599759
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) > 0.048717
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 23.500000
            ):
            return "BUY"  # path: mean_ret=1.194%, n=1185, risk_score=5, worst=-1.08%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.299267
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 21.176227
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) > 0.155155
                and not np.isnan(f.get('is_green', np.nan)) and f.get('is_green', np.nan) > 0.746332
            ):
            return "BUY"  # path: mean_ret=1.152%, n=490, risk_score=5, worst=-1.48%

        elif (
                not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 17.971331
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.205870
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 14.966715
                and not np.isnan(f.get('atr_pct', np.nan)) and f.get('atr_pct', np.nan) > 0.624342
            ):
            return "BUY"  # path: mean_ret=1.364%, n=814, risk_score=5, worst=-1.14%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.032937
                and not np.isnan(f.get('roc_3', np.nan)) and f.get('roc_3', np.nan) <= -1.759810
                and not np.isnan(f.get('macd_above_signal', np.nan)) and f.get('macd_above_signal', np.nan) <= 0.553317
                and not np.isnan(f.get('atr_14', np.nan)) and f.get('atr_14', np.nan) <= 5.141143
            ):
            return "BUY"  # path: mean_ret=1.805%, n=30, risk_score=9, worst=-3.03%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.253225
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 22.829847
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) > 0.893788
                and not np.isnan(f.get('consec_red', np.nan)) and f.get('consec_red', np.nan) > 0.144657
            ):
            return "BUY"  # path: mean_ret=1.622%, n=421, risk_score=6, worst=-1.03%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.270703
                and not np.isnan(f.get('hv_5', np.nan)) and f.get('hv_5', np.nan) > 3.283650
                and not np.isnan(f.get('atr_percentile', np.nan)) and f.get('atr_percentile', np.nan) <= 0.375000
            ):
            return "BUY"  # path: mean_ret=1.376%, n=66, risk_score=4, worst=-0.68%

        elif (
                not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 18.266629
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.118968
                and not np.isnan(f.get('compression_score', np.nan)) and f.get('compression_score', np.nan) > 0.495685
                and not np.isnan(f.get('ema5_slope', np.nan)) and f.get('ema5_slope', np.nan) <= -0.310906
            ):
            return "BUY"  # path: mean_ret=1.127%, n=435, risk_score=8, worst=-0.74%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.205632
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.845284
                and not np.isnan(f.get('hv_5', np.nan)) and f.get('hv_5', np.nan) > 0.475065
                and not np.isnan(f.get('volume_ratio', np.nan)) and f.get('volume_ratio', np.nan) > 7.048796
            ):
            return "BUY"  # path: mean_ret=1.986%, n=19, risk_score=9, worst=-0.14%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.249211
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.819114
                and not np.isnan(f.get('hv_5', np.nan)) and f.get('hv_5', np.nan) <= 0.474859
                and not np.isnan(f.get('gap_down_pct', np.nan)) and f.get('gap_down_pct', np.nan) > 0.024716
            ):
            return "BUY"  # path: mean_ret=0.814%, n=11, risk_score=7, worst=-0.65%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.766275
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.068616
                and not np.isnan(f.get('ret_5', np.nan)) and f.get('ret_5', np.nan) <= -1.406868
                and not np.isnan(f.get('atr_norm_candle', np.nan)) and f.get('atr_norm_candle', np.nan) <= 2.246919
            ):
            return "BUY"  # path: mean_ret=1.323%, n=245, risk_score=7, worst=-5.05%

        elif (
                not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 35.059244
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.637576
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.163135
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 14.175644
            ):
            return "BUY"  # path: mean_ret=1.103%, n=1207, risk_score=6, worst=-0.99%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.204106
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.845284
                and not np.isnan(f.get('hv_5', np.nan)) and f.get('hv_5', np.nan) <= 0.475065
                and not np.isnan(f.get('rsi_7_slope', np.nan)) and f.get('rsi_7_slope', np.nan) > -0.696422
            ):
            return "BUY"  # path: mean_ret=0.901%, n=15, risk_score=5, worst=-1.61%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.269325
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.843705
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) > 0.071773
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 22.500000
            ):
            return "BUY"  # path: mean_ret=1.228%, n=1106, risk_score=5, worst=-1.03%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.215190
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.016369
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.111111
                and not np.isnan(f.get('atr_percentile', np.nan)) and f.get('atr_percentile', np.nan) > 0.089567
            ):
            return "BUY"  # path: mean_ret=1.597%, n=535, risk_score=7, worst=-6.91%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.269328
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.802182
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) > 0.046673
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 22.500000
            ):
            return "BUY"  # path: mean_ret=1.234%, n=1245, risk_score=5, worst=-1.03%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.714621
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.070965
                and not np.isnan(f.get('dist_prev_close', np.nan)) and f.get('dist_prev_close', np.nan) <= -1.506826
                and not np.isnan(f.get('roc_3', np.nan)) and f.get('roc_3', np.nan) <= -1.560925
            ):
            return "BUY"  # path: mean_ret=1.583%, n=55, risk_score=7, worst=-5.05%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.286623
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 24.728486
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.960302
                and not np.isnan(f.get('consec_red', np.nan)) and f.get('consec_red', np.nan) > 0.426587
            ):
            return "BUY"  # path: mean_ret=1.249%, n=1362, risk_score=6, worst=-1.53%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.228515
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) > 14.505838
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.148876
                and not np.isnan(f.get('obv', np.nan)) and f.get('obv', np.nan) > 77723758.447900
            ):
            return "BUY"  # path: mean_ret=0.301%, n=6, risk_score=7, worst=-0.18%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.251039
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.845284
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.125260
                and not np.isnan(f.get('hv_5', np.nan)) and f.get('hv_5', np.nan) <= 0.481943
            ):
            return "BUY"  # path: mean_ret=1.240%, n=14, risk_score=7, worst=-1.61%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.413870
                and not np.isnan(f.get('compression_score', np.nan)) and f.get('compression_score', np.nan) > 0.686605
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.114120
                and not np.isnan(f.get('ema5_slope', np.nan)) and f.get('ema5_slope', np.nan) <= -0.490803
            ):
            return "BUY"  # path: mean_ret=1.376%, n=109, risk_score=8, worst=-3.55%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.135042
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 23.131360
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) > 13.954907
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) > 0.643605
            ):
            return "BUY"  # path: mean_ret=1.429%, n=128, risk_score=6, worst=-0.44%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.863972
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) > 0.146996
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.287322
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.042415
            ):
            return "BUY"  # path: mean_ret=1.055%, n=798, risk_score=5, worst=-0.90%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.375164
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) <= 2.095816
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 22.112649
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.224889
            ):
            return "BUY"  # path: mean_ret=1.008%, n=1164, risk_score=5, worst=-1.03%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.530694
                and not np.isnan(f.get('compression_score', np.nan)) and f.get('compression_score', np.nan) <= 0.635363
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.331117
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 20.646125
            ):
            return "BUY"  # path: mean_ret=1.103%, n=1432, risk_score=5, worst=-1.72%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.493463
                and not np.isnan(f.get('atr_norm_candle', np.nan)) and f.get('atr_norm_candle', np.nan) > 2.050874
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.214889
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) > 0.795367
            ):
            return "BUY"  # path: mean_ret=1.722%, n=162, risk_score=5, worst=-5.07%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.413357
                and not np.isnan(f.get('rel_candle_size', np.nan)) and f.get('rel_candle_size', np.nan) > 1.777023
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 2.239608
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.228283
            ):
            return "BUY"  # path: mean_ret=1.747%, n=278, risk_score=5, worst=-5.07%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.251704
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.745614
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) > 0.091547
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 22.500000
            ):
            return "BUY"  # path: mean_ret=1.189%, n=1026, risk_score=5, worst=-1.03%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.653189
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.331522
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 14.091562
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.396376
            ):
            return "BUY"  # path: mean_ret=1.105%, n=1650, risk_score=4, worst=-1.72%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.205883
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.845284
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) > 0.048127
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.550006
            ):
            return "BUY"  # path: mean_ret=1.550%, n=518, risk_score=5, worst=-3.48%

        elif (
                not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 18.266629
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) > 0.118968
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.189939
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 23.557574
            ):
            return "BUY"  # path: mean_ret=0.977%, n=731, risk_score=6, worst=-1.08%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.270252
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.849576
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.125118
                and not np.isnan(f.get('hv_5', np.nan)) and f.get('hv_5', np.nan) <= 0.476941
            ):
            return "BUY"  # path: mean_ret=1.314%, n=12, risk_score=7, worst=-1.61%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.188033
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.732085
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 24.000000
            ):
            return "BUY"  # path: mean_ret=1.060%, n=1516, risk_score=5, worst=-0.99%

        elif (
                not np.isnan(f.get('cmf', np.nan)) and f.get('cmf', np.nan) <= -0.163053
                and not np.isnan(f.get('ema12_slope', np.nan)) and f.get('ema12_slope', np.nan) <= -0.437965
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.048889
            ):
            return "BUY"  # path: mean_ret=1.866%, n=32, risk_score=8, worst=-3.03%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.267955
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.835977
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.136966
                and not np.isnan(f.get('hv_5', np.nan)) and f.get('hv_5', np.nan) <= 0.475065
            ):
            return "BUY"  # path: mean_ret=1.265%, n=13, risk_score=7, worst=-1.61%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.217463
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 26.994065
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) <= 1.756439
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.879130
            ):
            return "BUY"  # path: mean_ret=1.052%, n=729, risk_score=5, worst=-1.14%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.098639
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.242526
                and not np.isnan(f.get('minute', np.nan)) and f.get('minute', np.nan) <= 25.768422
                and not np.isnan(f.get('vwap_dev', np.nan)) and f.get('vwap_dev', np.nan) > -9.976878
            ):
            return "BUY"  # path: mean_ret=1.224%, n=63, risk_score=6, worst=-3.55%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.376396
                and not np.isnan(f.get('hour', np.nan)) and f.get('hour', np.nan) <= 13.343018
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) <= 0.689416
                and not np.isnan(f.get('atr_norm', np.nan)) and f.get('atr_norm', np.nan) > 0.451376
            ):
            return "BUY"  # path: mean_ret=0.914%, n=519, risk_score=4, worst=-1.00%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.189716
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.857888
                and not np.isnan(f.get('hv_5', np.nan)) and f.get('hv_5', np.nan) <= 0.475065
                and not np.isnan(f.get('compression_score', np.nan)) and f.get('compression_score', np.nan) <= 0.665736
            ):
            return "BUY"  # path: mean_ret=0.976%, n=24, risk_score=7, worst=-1.61%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.215296
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.654196
                and not np.isnan(f.get('obv', np.nan)) and f.get('obv', np.nan) <= 144762840.000000
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.714514
            ):
            return "BUY"  # path: mean_ret=1.638%, n=507, risk_score=5, worst=-5.07%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.215113
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.801724
                and not np.isnan(f.get('hv_5', np.nan)) and f.get('hv_5', np.nan) <= 0.476941
                and not np.isnan(f.get('volume_ratio', np.nan)) and f.get('volume_ratio', np.nan) > 1.326014
            ):
            return "BUY"  # path: mean_ret=0.984%, n=21, risk_score=7, worst=-1.61%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.261874
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.857815
                and not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) > 0.125118
                and not np.isnan(f.get('time_bar', np.nan)) and f.get('time_bar', np.nan) <= 22.500000
            ):
            return "BUY"  # path: mean_ret=1.159%, n=791, risk_score=5, worst=-1.03%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.215190
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.844391
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.421303
                and not np.isnan(f.get('consec_green', np.nan)) and f.get('consec_green', np.nan) > 2.500000
            ):
            return "BUY"  # path: mean_ret=2.250%, n=11, risk_score=6, worst=-0.89%

        elif (
                not np.isnan(f.get('price_position', np.nan)) and f.get('price_position', np.nan) <= 0.214719
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 0.606418
                and not np.isnan(f.get('spread_proxy', np.nan)) and f.get('spread_proxy', np.nan) > 1.415450
                and not np.isnan(f.get('consec_green', np.nan)) and f.get('consec_green', np.nan) > 2.500000
            ):
            return "BUY"  # path: mean_ret=2.250%, n=11, risk_score=6, worst=-0.89%

        return "HOLD"


if __name__ == "__main__":
    trader = HardcodedTrader()
    test = {"rsi_14": 30.0, "volume_ratio": 1.5}
    print(f"Signal: {trader.generate_signal(test)}")
