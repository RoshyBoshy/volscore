# VolSSCORE: Volatility Short-Straddle Strategy Screening Platform

VolSSCORE is a Python-based screening platform that implements the Barclays short-straddle strategy for identifying opportunities in overpriced volatility. This tool helps traders and investors identify stocks where options may be overpriced relative to their historical price movement.

## 🚀 Features

- **Volatility Metrics Calculation**: Calculate realized volatility (RV), at-the-money implied volatility (IV), and the VolSSCORE metric
- **Advanced Screening**: Filter stocks based on customizable criteria including vol_premium, hist_percentile, spreads, and open interest
- **Data Integration**: Seamless integration with yfinance for retrieving stock universe, historical prices, and option chains
- **Visualization**: Generate insightful charts and reports for analysis
- **Command-line Interface**: Run complete workflows from the command-line
- **Efficient Caching**: Built-in caching system to avoid excessive API calls
- **Comprehensive Reports**: Generate detailed reports in various formats including CSV and interactive HTML dashboards

## 📊 The VolSSCORE Metric

The core of the system is the VolSSCORE metric, which combines multiple volatility factors to identify potentially overpriced options:

1. **Volatility Premium** (IV - RV): The spread between implied volatility and realized volatility
2. **Sector Premium** (IV - sector_avg_IV): How a stock's IV compares to its sector peers
3. **Historical Percentile** (IV vs past_1yr_IV): Current IV relative to its historical range

These components are normalized to a [0,1] range and combined with the following weights:

- 50% vol_premium
- 30% sector_premium
- 20% hist_percentile

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installing from PyPI

```bash
pip install volscore
```

### Installing from Source

Clone the repository and install in development mode:

```bash
git clone https://github.com/yourusername/volscore.git
cd volscore
pip install -e .
```

## 🖥️ Usage

### Command-line Interface

VolSSCORE provides a comprehensive command-line interface for running various tasks:

#### Running the Full Workflow

```bash
volscore full-workflow --sector "Technology" --min-volscore 0.7 --min-vol-premium 0.05
```

#### Fetching Market Data

```bash
volscore fetch-data --sector "Technology" --period "1y" --interval "1d"
```

#### Screening Stocks

```bash
volscore screen --min-volscore 0.8 --min-vol-premium 0.1 --filter-spreads --save-csv
```

#### Analyzing a Single Stock

```bash
volscore analyze AAPL --period "1y"
```

### Using as a Python Library

VolSSCORE can also be imported and used in your own Python projects:

```python
import pandas as pd
from volscore.data_ingestion import fetch_stock_universe, fetch_historical_prices, fetch_option_chains
from volscore.metrics import calculate_volscore
from volscore.screening import screen_stocks
from volscore.reporting import generate_volatility_chart

# Load universe and fetch data
universe = fetch_stock_universe(sector="Technology")
tickers = universe['ticker'].tolist()
price_data = fetch_historical_prices(tickers, period="1y")
option_data = fetch_option_chains(tickers)

# Apply screening
filters = [
    {'column': 'vol_premium', 'operator': '>', 'value': 0.05},
    {'column': 'hist_percentile', 'operator': '>=', 'value': 80}
]
results = screen_stocks(metrics_data, filters=filters, min_volscore=0.7)

# Generate visualization for top result
top_ticker = results.iloc[0]['ticker']
generate_volatility_chart(results, price_data, top_ticker, show_plot=True)
```

## 📁 Project Structure

```
volscore/
│
├── volscore/             # Main package directory
│   ├── __init__.py       # Package initialization
│   ├── data_ingestion.py # Data fetching functionality
│   ├── metrics.py        # Volatility metrics calculations
│   ├── screening.py      # Stock screening functionality
│   ├── reporting.py      # Reporting and visualization tools
│   └── cli.py            # Command-line interface
│
├── tests/                # Unit tests
│   ├── test_metrics.py   # Tests for metrics calculations
│   └── test_screening.py # Tests for screening functionality
│
├── data/                 # Data storage (created by the package)
│   ├── prices/           # Cached price data
│   └── options/          # Cached option data
│
├── reports/              # Generated reports (created by the package)
│
├── setup.py              # Package setup file
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

## 🧪 Testing

Run unit tests using pytest:

```bash
# Run all tests
pytest

# Run specific test module
pytest tests/test_metrics.py

# Run with coverage report
pytest --cov=volscore
```

## 📈 Example Output

When running the full workflow, VolSSCORE will display a table with the top opportunities:

```
Top 10 VolSSCORE Opportunities
+--------+---------------------------+------+------+--------------+----------------+----------+
| ticker | sector                    | iv   | rv   | vol_premium  | hist_percentile | volscore |
+--------+---------------------------+------+------+--------------+----------------+----------+
| XYZ    | Information Technology    | 35.0% | 20.0% | 15.0%       | 92%            | 0.89     |
| ABC    | Health Care               | 40.0% | 25.0% | 15.0%       | 95%            | 0.87     |
| DEF    | Communication Services    | 45.0% | 30.0% | 15.0%       | 90%            | 0.86     |
| GHI    | Consumer Discretionary    | 30.0% | 18.0% | 12.0%       | 88%            | 0.81     |
| ...    | ...                       | ...   | ...   | ...         | ...            | ...      |
+--------+---------------------------+------+------+--------------+----------------+----------+
```

The package also generates various visual reports including:

- Volatility charts for individual stocks
- Sector comparison charts
- Volatility premium distribution histograms
- Interactive dashboards

## 🔄 Workflow Diagram

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Stock Universe  │────▶│ Historical      │────▶│ Option Chain    │
│ Selection       │     │ Price Data      │     │ Data            │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │                      │
         │                       ▼                      │
         │              ┌─────────────────┐             │
         │              │ Calculate       │             │
         └─────────────▶│ Volatility     │◀────────────┘
                        │ Metrics        │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ Apply Screening │
                        │ Filters         │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ Generate        │
                        │ Reports         │
                        └─────────────────┘
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by the Barclays short-straddle strategy for overpriced volatility
- Powered by yfinance for market data access
- Visualization capabilities enhanced by Matplotlib, Seaborn, and Plotly

## 📞 Contact

For questions, feature requests, or support, please open an issue on GitHub.
