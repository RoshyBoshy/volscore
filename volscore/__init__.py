"""
VolSSCORE: A Python package for implementing the Barclays short-straddle strategy for overpriced volatility.

This package provides tools for screening stocks based on volatility metrics,
comparing implied volatility (IV) with realized volatility (RV), and identifying
potential opportunities where options are overpriced relative to historical movement.
"""

__version__ = '0.1.0'
__author__ = 'VolSSCORE Team'

from volscore.data_ingestion import (
    fetch_stock_universe,
    fetch_historical_prices,
    fetch_option_chains,
    load_sp500
)

from volscore.metrics import (
    calculate_realized_volatility,
    calculate_implied_volatility,
    calculate_vol_premium,
    calculate_sector_premium,
    calculate_hist_percentile,
    calculate_volscore
)

from volscore.screening import (
    screen_stocks,
    filter_by_vol_premium,
    filter_by_hist_percentile, 
    filter_by_open_interest,
    filter_by_spread
)

from volscore.reporting import (
    export_to_csv,
    print_results_table,
    generate_volatility_chart,
    generate_sector_comparison
)