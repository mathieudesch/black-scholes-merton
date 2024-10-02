# Stock Option Analyzer

## Description

The Stock Option Analyzer is a Python program designed to identify potentially underpriced stock options in the market. It uses the Black-Scholes-Merton model to calculate theoretical option prices and compares them with market prices. The program can analyze options for the entire S&P 500 index or for a specific stock, allowing users to find potential trading opportunities.

Key features:
- Analyze S&P 500 stocks or a specific stock
- Choose between mark, ask, or bid prices for comparison
- Analyze call options, put options, or both
- Set custom filters for price range and open interest
- Parallel processing for efficient analysis of multiple stocks
- Generates a detailed output file with underpriced options

## Dependencies

This program requires Python 3.7 or later. The following Python libraries are needed:

- yfinance
- pandas
- numpy
- torch (PyTorch)
- fredapi

You can install these dependencies using pip:

```
pip install yfinance pandas numpy torch fredapi
```

Note: The PyTorch installation may vary depending on your system. Please refer to the [official PyTorch website](https://pytorch.org/get-started/locally/) for the most appropriate installation method for your setup.

## FRED API Key

This program uses data from the Federal Reserve Economic Data (FRED) to obtain risk-free rates. To use this feature, you need to obtain a free API key from FRED:

1. Go to the [FRED website](https://fred.stlouisfed.org/)
2. Create an account or sign in
3. Go to your account settings and request an API key

Once you have your API key, you need to paste it in here as a string

```python
os.environ['FRED_API_KEY'] = 'PUT YOUR API KEY HERE'
```

## Usage

To run the program:

1. Ensure all dependencies are installed and your FRED API key is set.
2. Run the script:
   ```
   python stock_option_analyzer.py
   ```
3. Follow the prompts to select your analysis options:
   - Choose between S&P 500 analysis or a specific stock
   - Select the price type (mark, ask, or bid)
   - Choose option type (calls, puts, or both)
   - Set minimum and maximum price filters
   - Set minimum open interest filter
4. The program will run the analysis and generate an `output.txt` file with the results.

## Output

The program generates an `output.txt` file containing:
- Analysis parameters
- List of potentially underpriced options, sorted by expiration date and percentage difference between theoretical and market prices
- Details for each option including ticker, expiration date, strike price, market price, theoretical price, percentage difference, open interest, implied volatility, and days to expiration

## Note

This program is for educational and research purposes only. It does not constitute financial advice. Always do your own research and consider seeking advice from a qualified financial professional before making investment decisions.

## Contributing

Contributions to improve the Stock Option Analyzer are welcome. Please feel free to submit pull requests or open issues to suggest improvements or report bugs.

## License

This project is licensed under the Apache License, Version 2.0. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
