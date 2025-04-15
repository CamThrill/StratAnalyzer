import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly.offline import plot
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QPushButton, QLabel, QVBoxLayout, QWidget, QComboBox, QMessageBox, QTabWidget, QTextEdit, QCheckBox, QHBoxLayout, QScrollArea
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineSettings  # Import QWebEngineSettings
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView
import plotly.io as pio
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene
from PyQt5.QtGui import QPixmap


# Thread for asynchronous file loading
class FileLoaderThread(QThread):
    loaded = pyqtSignal(pd.DataFrame, str)  # Signal to emit when file is loaded
    error = pyqtSignal(str)  # Signal to emit when an error occurs

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            df = pd.read_excel(self.file_path)
            self.loaded.emit(df, self.file_path)
        except Exception as e:
            self.error.emit(str(e))

class StrategyAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()

        # Window setup
        self.setWindowTitle("NinjaTrader Strategy Analyzer")
        self.setGeometry(100, 100, 1000, 700)

        # Initialize UI components
        self.init_ui()

        # Data Storage
        self.dataframes = {}

    def toggle_multi_select(self):
        """Toggle multi-select mode for strategy checkboxes."""
        for checkbox in self.strategy_checkboxes.values():
            checkbox.setCheckable(not checkbox.isCheckable())

    def toggle_select_all(self):
        """Toggle selection of all strategies."""
        if not self.strategy_checkboxes:
            return

        # Check if all strategies are currently selected
        all_checked = all(checkbox.isChecked() for checkbox in self.strategy_checkboxes.values())

        # Toggle all checkboxes
        for checkbox in self.strategy_checkboxes.values():
            checkbox.setChecked(not all_checked)

        # Update the summary and metrics
        self.update_summary()
        self.update_metrics("All")

    def calculate_max_drawdown(self, profits):
            """Calculate the maximum drawdown from a series of profits."""
            cumulative_pnl = profits.cumsum()
            max_drawdown = (cumulative_pnl.cummax() - cumulative_pnl).max()
            return max_drawdown

    def calculate_sharpe_ratio(self, returns, risk_free_rate, years):
        """Calculate the annualized Sharpe ratio."""
        if len(returns) == 0:
            return 0

        # Calculate mean and standard deviation of returns
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # Annualize the returns and standard deviation
        if years > 0:
            annualized_mean_return = mean_return * 252  # Assuming 252 trading days in a year
            annualized_std_return = std_return * np.sqrt(252)
        else:
            annualized_mean_return = mean_return
            annualized_std_return = std_return

        # Calculate Sharpe ratio
        if annualized_std_return > 0:
            sharpe_ratio = (annualized_mean_return - risk_free_rate) / annualized_std_return
        else:
            sharpe_ratio = 0

        return sharpe_ratio

    def calculate_sortino_ratio(self, returns, risk_free_rate):
        """Calculate the Sortino ratio."""
        if len(returns) == 0:
            return 0

        # Calculate mean return adjusted for risk-free rate
        mean_return = np.mean(returns) - risk_free_rate

        # Calculate downside deviation (only returns below the risk-free rate)
        downside_returns = returns[returns < risk_free_rate]
        if len(downside_returns) == 0:
            return 0  # Avoid division by zero

        downside_deviation = np.std(downside_returns)

        # Calculate Sortino ratio
        if downside_deviation > 0:
            sortino_ratio = mean_return / downside_deviation
        else:
            sortino_ratio = 0

        return sortino_ratio

    def init_ui(self):
        """Initialize the user interface."""
        # Tab Widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Tabs
        self.summary_tab = QWidget()
        self.metrics_tab = QWidget()
        self.trades_tab = QWidget()
        self.streaks_tab = QWidget()
        self.correlation_tab = QWidget()

        self.tabs.addTab(self.summary_tab, "Performance Summary")
        self.tabs.addTab(self.metrics_tab, "Metrics")
        self.tabs.addTab(self.trades_tab, "Trades Breakdown")
        self.tabs.addTab(self.streaks_tab, "Streak Analysis")
        self.tabs.addTab(self.correlation_tab, "Correlation Matrix")

        # Initialize Tabs
        self.init_summary_tab()
        self.init_metrics_tab()
        self.init_trades_tab()
        self.init_streaks_tab()
        self.init_correlation_tab()

        # Load File Button
        self.load_button = QPushButton("📂 Load NinjaTrader XLSX File")
        self.load_button.clicked.connect(self.load_file)
        self.tabs.setCornerWidget(self.load_button, Qt.TopRightCorner)

        # Status Bar
        self.statusBar().showMessage("Ready")

    def init_summary_tab(self):
        """Initialize the Summary tab."""
        layout = QVBoxLayout()

        # Strategy Selection Section
        self.strategy_selection_layout = QVBoxLayout()
        self.strategy_checkboxes = {}  # Store checkboxes for each strategy

        # Select All Button
        self.select_all_button = QPushButton("Select All")
        self.select_all_button.clicked.connect(self.toggle_select_all)
        self.strategy_selection_layout.addWidget(self.select_all_button)

        # Scroll Area for Strategy Checkboxes
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_widget.setLayout(self.strategy_selection_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area, stretch=1)  # Lower stretch factor

        # Summary Label
        self.summary_label = QTextEdit()
        self.summary_label.setReadOnly(True)
        layout.addWidget(self.summary_label, stretch=1)  # Lower stretch factor

        # Graphics view for displaying the PnL graph as an image
        self.graphics_view = QGraphicsView()
        self.graphics_scene = QGraphicsScene()
        self.graphics_view.setScene(self.graphics_scene)
        layout.addWidget(self.graphics_view, stretch=4)  # Higher stretch factor

        # Button to toggle between individual PnL lines and total PnL
        self.toggle_total_pnl_button = QPushButton("Show Total PnL as One Line")
        self.toggle_total_pnl_button.clicked.connect(self.toggle_total_pnl)
        layout.addWidget(self.toggle_total_pnl_button)

        # Button to toggle x-axis between trades and dates
        self.toggle_x_axis_button = QPushButton("Show by Dates")
        self.toggle_x_axis_button.clicked.connect(self.toggle_x_axis)
        layout.addWidget(self.toggle_x_axis_button)

        self.summary_tab.setLayout(layout)

        # Variables to track graph modes
        self.show_total_pnl = False  # Tracks whether to show total PnL
        self.show_by_dates = False   # Tracks whether to show by dates or trades

    def toggle_total_pnl(self):
        """Toggle between showing individual PnL lines and total PnL."""
        self.show_total_pnl = not self.show_total_pnl
        if self.show_total_pnl:
            self.toggle_total_pnl_button.setText("Show Individual PnL Lines")
        else:
            self.toggle_total_pnl_button.setText("Show Total PnL as One Line")
        self.update_graph()

    def toggle_x_axis(self):
        """Toggle the x-axis between trades and dates."""
        self.show_by_dates = not self.show_by_dates
        if self.show_by_dates:
            self.toggle_x_axis_button.setText("Show by Trades")
        else:
            self.toggle_x_axis_button.setText("Show by Dates")
        self.update_graph()

    def toggle_select_all(self):
        """Toggle selection of all strategies."""
        if not self.strategy_checkboxes:
            return

        # Check if all strategies are currently selected
        all_checked = all(checkbox.isChecked() for checkbox in self.strategy_checkboxes.values())

        # Toggle all checkboxes
        for checkbox in self.strategy_checkboxes.values():
            checkbox.setChecked(not all_checked)

        # Update the summary and graph
        self.update_summary()
        self.update_graph()

    def init_metrics_tab(self):
        """Initialize the Metrics tab."""
        layout = QVBoxLayout()

        # Buttons for filtering trades
        button_layout = QHBoxLayout()
        self.all_trades_button = QPushButton("All Trades")
        self.long_trades_button = QPushButton("Long")
        self.short_trades_button = QPushButton("Short")

        self.all_trades_button.clicked.connect(lambda: self.update_metrics("All"))
        self.long_trades_button.clicked.connect(lambda: self.update_metrics("Long"))
        self.short_trades_button.clicked.connect(lambda: self.update_metrics("Short"))

        button_layout.addWidget(self.all_trades_button)
        button_layout.addWidget(self.long_trades_button)
        button_layout.addWidget(self.short_trades_button)
        layout.addLayout(button_layout)

        # Metrics display
        self.metrics_label = QTextEdit()
        self.metrics_label.setReadOnly(True)
        layout.addWidget(self.metrics_label)

        self.metrics_tab.setLayout(layout)

    def init_trades_tab(self):
        """Initialize the Trades tab."""
        layout = QVBoxLayout()
        self.trades_label = QTextEdit()
        self.trades_label.setReadOnly(True)
        layout.addWidget(self.trades_label)
        self.trades_tab.setLayout(layout)

    def init_streaks_tab(self):
        """Initialize the Streaks tab."""
        layout = QVBoxLayout()
        self.streaks_label = QTextEdit()
        self.streaks_label.setReadOnly(True)
        layout.addWidget(self.streaks_label)
        self.streaks_tab.setLayout(layout)

    def init_correlation_tab(self):
        """Initialize the Correlation tab."""
        layout = QVBoxLayout()
        self.correlation_button = QPushButton("Show Correlation Matrix")
        self.correlation_button.clicked.connect(self.show_correlation_matrix)
        layout.addWidget(self.correlation_button)
        self.correlation_tab.setLayout(layout)

    def load_file(self):
        """Load and process multiple XLSX files asynchronously."""
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Open NinjaTrader XLSX Files", "", "Excel Files (*.xlsx)")
        if file_paths:
            self.statusBar().showMessage("Loading files...")
            self.load_button.setEnabled(False)

            # Store threads in a list to prevent garbage collection
            self.threads = []  # List to store QThread objects

            # Load each file asynchronously
            for file_path in file_paths:
                thread = FileLoaderThread(file_path)
                thread.loaded.connect(self.on_file_loaded)
                thread.error.connect(self.on_file_error)
                thread.finished.connect(self.on_thread_finished)  # Clean up when thread finishes
                self.threads.append(thread)  # Add thread to the list
                thread.start()

    def on_thread_finished(self):
        """Clean up finished threads."""
        # Remove finished threads from the list
        self.threads = [thread for thread in self.threads if thread.isRunning()]
        if not self.threads:  # All threads are finished
            self.statusBar().showMessage("All files loaded.")
            self.load_button.setEnabled(True)

    def on_file_loaded(self, df, file_path):
        """Handle the loaded dataframe."""
        self.statusBar().showMessage(f"Loaded: {file_path}")
        self.process_data(df, file_path)

    def on_file_error(self, error_message):
        """Handle file loading errors."""
        self.statusBar().showMessage("Error loading file.")
        QMessageBox.critical(self, "Error", error_message)

    def process_data(self, df, file_path):
        """Process the loaded dataframe."""
        try:
            df.columns = df.columns.str.strip().str.lower()  # Normalize column names

            # Validate required columns
            required_columns = {"entry time", "exit time", "profit", "market pos."}
            if not required_columns.issubset(df.columns):
                QMessageBox.warning(self, "Error", f"File must contain the following columns: {required_columns}")
                return

            # Process data
            df["entry time"] = pd.to_datetime(df["entry time"])
            df["exit time"] = pd.to_datetime(df["exit time"])
            df["trade date"] = df["entry time"].dt.date  # Create trade_date column

            # Ensure "trade date" is a datetime column
            df["trade date"] = pd.to_datetime(df["trade date"])

            # Sort data by trade date
            df = df.sort_values("trade date")

            df["profit"] = df["profit"].astype(str)
            df["profit"] = df["profit"].str.replace('[\$,]', '', regex=True)
            df["profit"] = df["profit"].str.replace(r'\((.*?)\)', r'-\1', regex=True)
            df["profit"] = df["profit"].astype(float)

            # Map "Market pos." to "action" (Long -> buy, Short -> sell)
            df["action"] = df["market pos."].str.lower().map({"long": "buy", "short": "sell"})

            # Store the dataframe
            dataset_name = file_path.split("/")[-1].replace(".xlsx", "")  # Remove .xlsx from the name
            self.dataframes[dataset_name] = df

            # Add checkbox for the new strategy
            self.add_strategy_checkbox(dataset_name)

            # Update UI
            self.update_summary()
            self.update_metrics("All")  # Default to "All Trades" view
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process file: {str(e)}")

    def add_strategy_checkbox(self, strategy_name):
        """Add a checkbox for a new strategy."""
        checkbox = QCheckBox(strategy_name)
        checkbox.setChecked(True)  # Default to selected
        checkbox.stateChanged.connect(self.update_summary)
        self.strategy_checkboxes[strategy_name] = checkbox
        self.strategy_selection_layout.addWidget(checkbox)

    def update_summary(self):
        """Update the summary tab with calculated metrics."""
        if not self.dataframes:
            self.summary_label.setText("No data loaded. Please load a file.")
            return

        # Filter dataframes based on selected strategies
        selected_strategies = [name for name, checkbox in self.strategy_checkboxes.items() if checkbox.isChecked()]
        filtered_dataframes = {name: df for name, df in self.dataframes.items() if name in selected_strategies}

        if not filtered_dataframes:
            self.summary_label.setText("No strategies selected.")
            return

        # Concatenate all profits from selected strategies and sort by trade date
        all_data = pd.concat([df[["trade date", "profit"]] for df in filtered_dataframes.values()], axis=0)
        all_data = all_data.sort_values("trade date")  # Sort by trade date

        # Calculate cumulative PnL for max drawdown
        max_drawdown = self.calculate_max_drawdown(all_data["profit"])

        # Calculate total profits
        total_profits = all_data["profit"].sum()

        # Calculate total number of trades
        total_trades = len(all_data)

        # Calculate win rate
        if total_trades > 0:
            win_rate = (all_data["profit"] > 0).sum() / total_trades * 100
        else:
            win_rate = 0  # Default to 0 if there are no trades

        # Calculate start and end dates
        start_date = all_data["trade date"].min()
        end_date = all_data["trade date"].max()
        years = (end_date - start_date).days / 365.25

        # Calculate CAGR
        initial_capital = 10000  # Define initial capital (adjust as needed)
        ending_value = initial_capital + total_profits
        cagr = (ending_value / initial_capital) ** (1 / years) - 1 if years > 0 else 0

        # Calculate largest losing trade
        largest_losing_trade = all_data["profit"].min()

        # Calculate required capital
        required_capital = (max_drawdown * 2) + abs(largest_losing_trade)

        # Calculate Sharpe ratio
        risk_free_rate = 0  # Assume risk-free rate is 0% for simplicity
        returns = all_data["profit"] / initial_capital  # Convert profits to returns
        sharpe_ratio = self.calculate_sharpe_ratio(returns, risk_free_rate, years)

        # Calculate Sortino ratio
        sortino_ratio = self.calculate_sortino_ratio(returns, risk_free_rate)

        # Display summary metrics
        summary_text = (
            f"Portfolio Profits: ${total_profits:,.2f}\nBacktest Dates: {start_date.date()} to {end_date.date()}\n"
            f"CAGR: {cagr:.2%}\nTotal Trades: {total_trades}\nWin Rate: {win_rate:.2f}%\n"
            f"Sharpe Ratio: {sharpe_ratio:.2f}\nSortino Ratio: {sortino_ratio:.2f}\n"
            f"Max Drawdown: ${max_drawdown:,.2f}\nLargest Losing Trade: ${largest_losing_trade:,.2f}\n"
            f"Required Capital: ${required_capital:,.2f}"
        )
        self.summary_label.setText(summary_text)
        self.update_graph()

    def update_metrics(self, trade_type):
        """Update the metrics tab based on the selected trade type."""
        if not self.dataframes:
            return

        # Combine all dataframes into one
        combined_df = pd.concat(self.dataframes.values())

        # Filter trades based on type
        if trade_type == "Long":
            filtered_df = combined_df[combined_df["action"].str.lower() == "buy"]
        elif trade_type == "Short":
            filtered_df = combined_df[combined_df["action"].str.lower() == "sell"]
        else:
            filtered_df = combined_df  # All trades

        # Ensure trade_date is a datetime column
        filtered_df["trade date"] = pd.to_datetime(filtered_df["trade date"], errors="coerce")
        filtered_df = filtered_df.dropna(subset=["trade date"])

        # Sort by trade date
        filtered_df = filtered_df.sort_values("trade date")

        # Calculate cumulative profits for max drawdown
        cumulative_pnl = filtered_df["profit"].cumsum()
        max_drawdown = (cumulative_pnl.cummax() - cumulative_pnl).max()

        # Calculate other metrics
        total_net_profit = filtered_df["profit"].sum()
        win_rate = (filtered_df["profit"] > 0).mean() * 100
        avg_trade = filtered_df["profit"].mean()
        avg_winning_trade = filtered_df[filtered_df["profit"] > 0]["profit"].mean()
        avg_losing_trade = filtered_df[filtered_df["profit"] < 0]["profit"].mean()
        largest_winning_trade = filtered_df["profit"].max()
        largest_losing_trade = filtered_df["profit"].min()
        avg_trades_per_day = filtered_df.groupby(filtered_df["trade date"].dt.date).size().mean()

        # Calculate profit per month
        profit_per_month = filtered_df.groupby(filtered_df["trade date"].dt.to_period("M"))["profit"].sum().mean()

        # Display metrics
        metrics_text = (
            f"Total Net Profit: ${total_net_profit:,.2f}\n"
            f"Maximum Drawdown: ${max_drawdown:,.2f}\n"
            f"Win Rate: {win_rate:.2f}%\n"
            f"Average Trade: ${avg_trade:,.2f}\n"
            f"Average Winning Trade: ${avg_winning_trade:,.2f}\n"
            f"Average Losing Trade: ${avg_losing_trade:,.2f}\n"
            f"Largest Winning Trade: ${largest_winning_trade:,.2f}\n"
            f"Largest Losing Trade: ${largest_losing_trade:,.2f}\n"
            f"Average Trades Per Day: {avg_trades_per_day:.2f}\n"
            f"Profit Per Month: ${profit_per_month:,.2f}\n"
        )
        self.metrics_label.setText(metrics_text)

    def update_graph(self):
        """Update the PnL graph using Plotly and display it in a QGraphicsView."""
        if not self.dataframes:
            return

        # Filter dataframes based on selected strategies
        selected_strategies = [name for name, checkbox in self.strategy_checkboxes.items() if checkbox.isChecked()]
        filtered_dataframes = {name: df.copy() for name, df in self.dataframes.items() if name in selected_strategies}

        if not filtered_dataframes:
            return

        fig = go.Figure()

        if self.show_total_pnl:
            # Concatenate all profits and trade dates
            all_data = pd.concat(
                [df[["trade date", "profit"]] for df in filtered_dataframes.values()],
                axis=0,
                ignore_index=True,
            )

            # Ensure "trade date" is a datetime column
            all_data["trade date"] = pd.to_datetime(all_data["trade date"])

            # Group by trade date and sum the profits
            grouped_data = all_data.groupby("trade date", as_index=False)["profit"].sum()

            # Sort by trade date
            grouped_data = grouped_data.sort_values("trade date")

            # Calculate cumulative sum of profits
            grouped_data["cumulative_pnl"] = grouped_data["profit"].cumsum()

            # Determine x-axis data
            if self.show_by_dates:
                # Use trade dates for x-axis
                x_data = grouped_data["trade date"]
                x_label = "Date"
            else:
                # Use trade indices for x-axis
                x_data = grouped_data.index
                x_label = "Trade"

            # Plot the total PnL as one line
            fig.add_trace(go.Scatter(x=x_data, y=grouped_data["cumulative_pnl"], mode='lines', name='Total PnL'))
        else:
            # Plot individual PnL lines
            for name, df in filtered_dataframes.items():
                # Calculate net profit for the strategy
                net_profit = df["profit"].sum()
                # Format the legend name with net profit
                legend_name = f"{name}: ${net_profit:,.2f}"

                if self.show_by_dates:
                    # Use trade dates for x-axis
                    x_data = df["trade date"]
                    x_label = "Date"
                else:
                    # Use trade indices for x-axis
                    x_data = df.index
                    x_label = "Trade"

                # Add the trace with the updated legend name
                                # Add the trace with the updated legend name
                fig.add_trace(go.Scatter(x=x_data, y=df["profit"].cumsum(), mode='lines', name=legend_name))

        # Update layout
        fig.update_layout(
            title="Cumulative PnL",
            xaxis_title=x_label,
            yaxis_title="PnL",
            width=1200,  # Set the width of the figure
            height=600,  # Set the height of the figure
        )

        # Save the Plotly figure as an image
        pio.write_image(fig, "plotly_graph.png")

        # Load the image into the QGraphicsView
        self.graphics_scene.clear()  # Clear the existing scene
        pixmap = QPixmap("plotly_graph.png")
        self.graphics_scene.addPixmap(pixmap)

    def show_correlation_matrix(self):
        """Show the correlation matrix for strategies based on daily PnL."""
        if len(self.dataframes) < 2:
            QMessageBox.warning(self, "Warning", "Load at least two datasets for correlation analysis.")
            return

        try:
            pnl_data = {name: df.groupby(df["trade date"])["profit"].sum() for name, df in self.dataframes.items()}
            pnl_df = pd.DataFrame(pnl_data)

            if len(pnl_df.columns) > 1:
                corr_matrix = pnl_df.corr()
                plt.figure(figsize=(6, 4))
                sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
                plt.title("Correlation Matrix (Daily PnL)")
                plt.show()
            else:
                QMessageBox.warning(self, "Warning", "Insufficient data for correlation analysis.")
        except KeyError as e:
            QMessageBox.critical(self, "Error", f"Missing expected column: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while calculating the correlation matrix: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StrategyAnalyzer()
    window.show()
    sys.exit(app.exec_())