"""
PCA-Based Mean Reversion Strategy in FX Markets
Portfolio: Emerging Markets
Autor: basado en el paper de Arath Reyes (2025)

Requisitos:
    pip install yfinance pandas numpy scikit-learn matplotlib seaborn
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  Clase principal
# ─────────────────────────────────────────────

class MeanReversionStrategy:
    def __init__(self, tickers: list, start_date: str, end_date: str, threshold: float = 1.0):
        """
        Parameters
        ----------
        tickers    : Lista de tickers de Yahoo Finance (ej. ['BRL=X', 'MXN=X'])
        start_date : Fecha inicio 'YYYY-MM-DD'
        end_date   : Fecha fin   'YYYY-MM-DD'
        threshold  : Umbral z-score para generar señales (default 1.0)
        """
        self.tickers    = tickers
        self.start_date = start_date
        self.end_date   = end_date
        self.threshold  = threshold

        # Atributos que se llenan durante la ejecución
        self.fx_data          = None
        self.log_returns      = None
        self.residuals        = None
        self.signals          = None
        self.strategy_returns = None
        self.daily_pnl        = None
        self.cumulative_pnl   = None
        self.ann_return       = None
        self.ann_vol          = None
        self.sharpe_ratio     = None

    # ── 1. Carga de datos ──────────────────────────────────────────────────
    def load_data(self):
        print("  Descargando datos de Yahoo Finance...")
        self.fx_data = (
            yf.download(self.tickers, start=self.start_date, end=self.end_date, interval="1d")
            ["Close"]
            .dropna()
        )
        self.log_returns = np.log(self.fx_data / self.fx_data.shift(1)).dropna()
        print(f"  Datos cargados: {len(self.fx_data)} días, {len(self.tickers)} pares\n")

    # ── 2. Señales vía PCA ─────────────────────────────────────────────────
    def generate_signals(self):
        print("  Aplicando PCA y calculando residuales...")
        pca = PCA()
        pca.fit(self.log_returns)

        # Reconstrucción completa (todos los PCs) y residuales
        reconstructed  = pca.inverse_transform(pca.transform(self.log_returns))
        self.residuals = self.log_returns - reconstructed

        # Z-scores de los residuales
        z_scores = (self.residuals - self.residuals.mean()) / self.residuals.std()

        # Señales: +1 (compra) si z < -θ, -1 (venta) si z > +θ
        raw_signals  = z_scores.applymap(
            lambda x: -1 if x >  self.threshold else
                       1 if x < -self.threshold else 0
        )
        # Lag de 1 día para evitar look-ahead bias
        self.signals = raw_signals.shift(1).fillna(0)
        print(f"  Señales generadas con threshold = {self.threshold}\n")

    # ── 3. Backtest ────────────────────────────────────────────────────────
    def backtest(self):
        print("  Corriendo backtest...")
        self.strategy_returns = self.signals * self.log_returns
        self.daily_pnl        = self.strategy_returns.sum(axis=1)
        self.cumulative_pnl   = self.daily_pnl.cumsum()

        self.ann_return  = self.daily_pnl.mean() * 252
        self.ann_vol     = self.daily_pnl.std()  * np.sqrt(252)
        self.sharpe_ratio = self.ann_return / self.ann_vol
        print("  Backtest completado.\n")

    # ── 4. Gráficas ────────────────────────────────────────────────────────
    def plot_results(self, portfolio_name: str = "Portfolio"):
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        fig.suptitle(f"{portfolio_name} — PCA Mean Reversion Strategy", fontsize=14, fontweight="bold")

        # Matriz de correlación
        corr = self.log_returns.corr()
        sns.heatmap(corr, ax=axes[0], annot=True, fmt=".2f", cmap="coolwarm",
                    linewidths=0.5, cbar_kws={"shrink": 0.8})
        axes[0].set_title("Correlation Matrix")
        axes[0].tick_params(axis="x", rotation=45)

        # P&L acumulado
        axes[1].plot(self.cumulative_pnl, color="steelblue", linewidth=1.5, label="Cumulative PnL")
        axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[1].set_title("Cumulative PnL")
        axes[1].set_xlabel("Date")
        axes[1].set_ylabel("Cumulative Return")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    # ── 5. Pipeline completo ───────────────────────────────────────────────
    def run(self, portfolio_name: str = "Portfolio"):
        print(f"{'='*50}")
        print(f"  Portafolio: {portfolio_name}")
        print(f"{'='*50}")
        self.load_data()
        self.generate_signals()
        self.backtest()
        self.plot_results(portfolio_name)
        self.print_metrics()

    def print_metrics(self):
        print("─" * 35)
        print(f"  Annualized Return  : {self.ann_return:.2%}")
        print(f"  Annualized Vol     : {self.ann_vol:.2%}")
        print(f"  Sharpe Ratio       : {self.sharpe_ratio:.2f}")
        print("─" * 35, "\n")


# ─────────────────────────────────────────────
#  Configuración — Emerging Markets
# ─────────────────────────────────────────────

if __name__ == "__main__":

    EM_TICKERS = ["BRL=X", "MXN=X", "CNY=X", "KRW=X", "JPY=X", "SGD=X", "INR=X"]
    START_DATE = "2020-05-01"
    END_DATE   = "2025-05-01"
    THRESHOLD  = 1.0          # ← puedes cambiar este hiperparámetro

    em_strategy = MeanReversionStrategy(
        tickers    = EM_TICKERS,
        start_date = START_DATE,
        end_date   = END_DATE,
        threshold  = THRESHOLD
    )
    em_strategy.run(portfolio_name="Emerging Markets")