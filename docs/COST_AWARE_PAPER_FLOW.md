# Flowchart — modello cost-aware nel Live Paper

Questo documento descrive il modello attualmente selezionato nel Live Paper:
**XGBoost cost-aware H15 — research**. È un esperimento virtuale: non invia
mai ordini al broker e non è stato promosso a una strategia reale.

```mermaid
flowchart TD
    subgraph Training[Training offline — una tantum]
        H[Storico XAUUSD M1\nBID/ASK + spread] --> F[FeatureEngine causale]
        F --> FT[Input: rendimenti, candele, ATR, volatilità,\nEMA, RSI, MACD, Bollinger, spread, ora e sessione]
        H --> L[Target cost-aware H15]
        L --> LT{Movimento eseguibile\ndopo BID/ASK + slippage?}
        LT -->|profitto LONG > 0,50| UP[UP]
        LT -->|profitto SHORT > 0,50| DOWN[DOWN]
        LT -->|altrimenti| NEUTRAL[NEUTRAL]
        FT --> XGB[XGBoost shallow\nclassificatore a 3 classi]
        UP --> XGB
        DOWN --> XGB
        NEUTRAL --> XGB
        XGB --> ART[Artefatto modello + manifest\nresearch-only]
    end

    subgraph Live[Live Paper — ogni candela M1 completata]
        MT5[MT5 broker\nTick BID/ASK + candele M1] --> CHECK{Tick fresco e\ncandela M1 completa?}
        CHECK -->|no| WAIT[Attendi / nessuna azione]
        CHECK -->|sì| LF[Stesso FeatureEngine\nsolo dati passati]
        ART --> PRED[XGBoost: P DOWN / P NEUTRAL / P UP]
        LF --> PRED
        PRED --> CLASS{Classe con\nprobabilità più alta}
        CLASS -->|UP| BUY[BUY\nscore interno 0,60]
        CLASS -->|DOWN| SELL[SELL\nscore interno 0,40]
        CLASS -->|NEUTRAL| HOLD[HOLD\nscore interno 0,50]
    end

    subgraph Paper[Motore virtuale — nessun ordine al broker]
        BUY --> GUARD{Limiti paper rispettati?}
        SELL --> GUARD
        HOLD --> NONE[NO TRADE\nnessuna nuova posizione]
        GUARD -->|spread, margine, rischio,\nlimite giornaliero o posizione: KO| BLOCK[NO TRADE / HOLD\ncon motivo]
        GUARD -->|OK| MODE{Modalità raffica}
        MODE -->|stesso verso, max 10 leg| OPEN[Apri una posizione virtuale\nalla quotazione eseguibile BID/ASK\n+ slippage configurato]
        OPEN --> MONITOR[Mark-to-market a ogni tick]
        MONITOR --> EXIT{SL, TP, inversione\no chiusura manuale?}
        EXIT -->|no| MONITOR
        EXIT -->|sì| CLOSE[Chiudi virtualmente\nBID per LONG / ASK per SHORT]
        CLOSE --> LEDGER[Ledger persistente:\ntrade, PnL, equity, drawdown, margine]
    end
```

## Cosa entra nel modello

Il modello non vede il futuro né “guarda” il grafico come una persona. Riceve
solo valori numerici calcolati dalle candele M1 già chiuse: variazioni di prezzo,
forma delle candele, volatilità, trend/medie, RSI/MACD, posizione nelle Bande di
Bollinger, spread e momento della giornata/sessione.

## Cosa produce davvero

Ad ogni candela M1 completa, XGBoost restituisce tre probabilità:

- `P(DOWN)`: ritiene più probabile un movimento SHORT eseguibile a 15 minuti;
- `P(NEUTRAL)`: non vede un movimento sufficiente dopo costi;
- `P(UP)`: ritiene più probabile un movimento LONG eseguibile a 15 minuti.

La classe più alta diventa `SELL`, `HOLD` o `BUY`. Per compatibilità con il
motore paper già esistente, la classe viene convertita internamente in uno score
0,40 / 0,50 / 0,60: **non è una probabilità binaria e non va letta come “60% di
probabilità di salire”**.

## Guardrail e risultati possibili

Anche quando il modello dice BUY o SELL, il motore può non entrare se lo spread
supera il limite, non c’è margine virtuale, è stato raggiunto il limite giornaliero
o ci sono già dieci posizioni concordi. In raffica può aprire al massimo una nuova
posizione per candela M1 completata.

Ogni posizione paper usa BID/ASK del broker, slippage e SL/TP. I risultati sono
solo virtuali e vengono salvati localmente: trade chiusi, PnL realizzato/aperto,
equity, margine e drawdown. Nessuna parte del flusso contiene API o codice di
invio ordini a MT5 o Trading212.
