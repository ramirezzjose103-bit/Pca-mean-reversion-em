# PCA Mean Reversion Strategy — FX Emerging Markets

Estrategia cuantitativa de trading en divisas EM que usa PCA para aislar movimientos idiosincrásicos, generando señales long/short cuando una divisa se desvía estadísticamente del comportamiento común del grupo.

---

## Instalación

```bash
pip install yfinance pandas numpy scikit-learn matplotlib seaborn
```

## Uso

```bash
python pca_mean_reversion_em.py
```

---

## Metodología

1. **Datos** — Retornos logarítmicos diarios de 7 pares de divisas EM vs USD (2020–2025).
2. **PCA** — Descompone los retornos en factores comunes y calcula los residuales.
3. **Señales** — Z-score de los residuales; compra si `z < -1`, vende si `z > +1`.
4. **Backtest** — Señales retrasadas 1 día para evitar look-ahead bias.

---

## Portafolio

| Ticker | Par | País |
|--------|-----|------|
| BRL=X | BRL/USD | Brasil |
| MXN=X | MXN/USD | México |
| CNY=X | CNY/USD | China |
| KRW=X | KRW/USD | Corea del Sur |
| JPY=X | JPY/USD | Japón |
| SGD=X | SGD/USD | Singapur |
| INR=X | INR/USD | India |

---

## Resultados (backtest 2020–2025)

| Métrica | Valor |
|--------|-------|
| Annualized Return | 2.38% |
| Annualized Volatility | ~14% |
| Sharpe Ratio | 0.17 |

---

## Parámetros

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `threshold` | `1.0` | Umbral z-score para generar señales |
| `start_date` | `2020-05-01` | Inicio del backtest |
| `end_date` | `2025-05-01` | Fin del backtest |

---

## Referencia

Basado en: *PCA-Based Mean Reversion Strategy in FX Markets* — Arath Reyes (2025).
