"""
ORO — Sistema de tasas base condicionales honestas.
====================================================
Un solo archivo. Dos modos:

    python oro.py           -> alerta diaria
    python oro.py validar   -> validacion walk-forward

Diferencias clave frente al sistema del Nasdaq 100:

  * UN solo instrumento. No hay "escanear 100 y quedarse con el mejor",
    que era la principal fuente de espejismos del otro sistema.
  * Costos mucho menores (spread de XAU/USD ~1 pb por lado vs ~5 pb en
    acciones). Una ventaja bruta pequena aqui puede sobrevivir.
  * Se reporta la EXPECTATIVA NETA en cada alerta, no solo el % de
    acierto. Esa fue la leccion cara del Nasdaq: una tasa de acierto
    alta con expectativa neta negativa es una trampa.
  * Dos horizontes: intradia (~2 anos de datos) y diario (~20+ anos).
    El diario tiene 10x mas muestra; miralo con mas confianza.

No predice el futuro. Reporta frecuencias historicas condicionales con
su incertidumbre. No es asesoria de inversion.
"""

import os
import sys
import math
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# ==================== CONFIGURACION ====================
# Instrumento. GC=F = futuro de oro COMEX (historia diaria desde ~2000).
# Si falla, cae a GLD (ETF, desde 2004).
TICKER = "GC=F"
TICKER_FALLBACK = "GLD"

# Hora de la ventana intradia, en hora de Nueva York.
# 10:00 ET se elige por tres razones:
#   1. El feed de GC=F alinea sus barras EN PUNTO, no a la media hora.
#   2. Siempre cae DESPUES de tu alerta de las 8:20 COT (que es 9:20 ET en
#      horario de verano de EE.UU. y 8:20 ET en invierno).
#   3. Coincide con el fixing PM de Londres (15:00 alli), un evento real
#      de liquidez en el oro.
SESION_HORA_ET = 10
SESION_MIN_ET = 0

# Que cuenta como "subida" en la primera hora. El oro se mueve menos que
# una accion individual; 0.3% es un umbral razonable, no 2%.
UMBRAL_INTRADIA = 0.003

# Costo de ida y vuelta en terminos de retorno.
# XAU/USD en Pepperstone: spread ~0.20 USD sobre ~3700 = ~0.5 pb por lado.
# 0.0002 (2 pb) incluye spread + algo de slippage. AJUSTA a tu ejecucion real.
COSTO_IDA_VUELTA = 0.0002

# Condiciones activas. Empezamos con UNA por la leccion del Nasdaq:
# con 3 features las celdas quedaban con 15 dias y nada pasaba el filtro.
FEATURES = ("gap_bin",)
# Opciones extra: "trend_bin", "vol_regime_bin", "dow"

MIN_MUESTRA = 30      # dias comparables minimos para reportar algo
CONFIANZA = 0.95      # nivel de los intervalos
FDR_ALPHA = 0.10      # control de falsos hallazgos entre condiciones
LIFT_MINIMO = 0.05    # ventaja minima sobre la tasa base (5 pp)

ZONA = "America/Bogota"
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")

log = logging.getLogger("oro")


# ==================== ESTADISTICA HONESTA ====================
def wilson(exitos: int, n: int, conf: float = CONFIANZA):
    """Intervalo de Wilson. Penaliza muestras chicas automaticamente."""
    if n <= 0:
        return 0.0, 0.0, 0.0
    z = 1.959963984540054 if abs(conf - 0.95) < 1e-9 else 1.6448536269514722
    p = exitos / n
    den = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / den
    mitad = (z / den) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centro - mitad), min(1.0, centro + mitad), p


def p_binomial(exitos: int, n: int, p0: float) -> float:
    """P(X >= exitos | Binomial(n, p0)). Evidencia contra la tasa base."""
    if n <= 0:
        return 1.0
    from scipy.stats import binomtest
    p0 = min(max(p0, 1e-9), 1 - 1e-9)
    return float(binomtest(exitos, n, p0, alternative="greater").pvalue)


def benjamini_hochberg(pv: np.ndarray, alpha: float = FDR_ALPHA):
    """Control de tasa de falsos descubrimientos entre las condiciones probadas."""
    p = np.asarray(pv, dtype=float)
    m = len(p)
    if m == 0:
        return np.array([], bool), np.array([], float)
    orden = np.argsort(p)
    rank = p[orden]
    crit = (np.arange(1, m + 1) / m) * alpha
    bajo = rank <= crit
    umbral = rank[np.max(np.where(bajo)[0])] if bajo.any() else -1.0
    q = np.minimum.accumulate((rank * m / np.arange(1, m + 1))[::-1])[::-1]
    qq = np.empty(m)
    qq[orden] = np.clip(q, 0, 1)
    return p <= umbral, qq


def media_ic(x: np.ndarray, conf: float = CONFIANZA):
    """Media con intervalo de confianza. Para la expectativa por operacion."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return (float(x.mean()) if n else 0.0), 0.0, 0.0
    z = 1.959963984540054
    m = float(x.mean())
    ee = float(x.std(ddof=1) / math.sqrt(n))
    return m, m - z * ee, m + z * ee


# ==================== DATOS ====================
def bajar(ticker: str, intervalo: str, periodo: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(ticker, period=periodo, interval=intervalo,
                     auto_adjust=False, progress=False, threads=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(how="all")


def datos_oro():
    """Devuelve (diario_largo, primera_hora). Cae al ETF si el futuro falla."""
    for tk in (TICKER, TICKER_FALLBACK):
        try:
            d = bajar(tk, "1d", "max")
            h = bajar(tk, "60m", "730d")
            if len(d) > 500 and not h.empty:
                log.info("Datos de %s: %d dias, %d barras de 60m", tk, len(d), len(h))
                return d, h, tk
        except Exception as e:
            log.warning("Fallo %s: %s", tk, e)
    raise RuntimeError("No se pudo descargar historial de oro.")


def primera_hora(h60: pd.DataFrame) -> pd.DataFrame:
    """Extrae la barra de sesion -> open, max, cierre de esa hora.

    El futuro de oro cotiza casi 24h y su feed puede alinear las barras en
    punto o a la media hora. En vez de exigir una alineacion concreta,
    probamos las candidatas y nos quedamos con la que MAS sesiones tenga.
    Exigir un match exacto fue justo el bug de la primera version: encontraba
    3 barras sueltas y nunca activaba el plan B.
    """
    idx = h60.index
    idx = idx.tz_localize("UTC") if idx.tz is None else idx
    idx = idx.tz_convert("America/New_York")
    h = h60.set_index(idx)

    sel, elegida = None, None
    for mm in (SESION_MIN_ET, 0, 30):
        s = h[(h.index.hour == SESION_HORA_ET) & (h.index.minute == mm)]
        if sel is None or len(s) > len(sel):
            sel, elegida = s, (SESION_HORA_ET, mm)
    log.info("Barra de sesion elegida: %02d:%02d ET -> %d sesiones",
             elegida[0], elegida[1], len(sel))
    if len(sel) < 100:
        disponibles = (pd.Series(h.index.strftime("%H:%M")).value_counts().head(6))
        log.warning("Pocas sesiones (%d). Horarios mas frecuentes en el feed:\n%s",
                    len(sel), disponibles.to_string())
    out = pd.DataFrame(
        {"open": sel["Open"].values, "fh_max": sel["High"].values,
         "fh_close": sel["Close"].values},
        index=sel.index.normalize().tz_localize(None))
    out.index.name = "fecha"
    return out[~out.index.duplicated(keep="first")]


# ==================== CONDICIONES ====================
def _bin_gap(g):
    if pd.isna(g):    return "desconocido"
    if g <= -0.005:   return "gap_baja_fuerte"
    if g <= -0.0015:  return "gap_baja"
    if g < 0.0015:    return "plano"
    if g < 0.005:     return "gap_alza"
    return "gap_alza_fuerte"


def _bin_trend(t):
    if pd.isna(t):   return "desconocido"
    if t <= -0.02:   return "tend_baja_fuerte"
    if t <= -0.005:  return "tend_baja"
    if t < 0.005:    return "tend_neutral"
    if t < 0.02:     return "tend_alza"
    return "tend_alza_fuerte"


def indicadores(d: pd.DataFrame) -> pd.DataFrame:
    """Todo desplazado un dia: la condicion del dia D usa info hasta D-1."""
    c, hi, lo = d["Close"], d["High"], d["Low"]
    pc = c.shift(1)
    tr = pd.concat([(hi - lo), (hi - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    atr = (tr.ewm(alpha=1 / 14, adjust=False).mean() / c).shift(1)
    pct = atr.rolling(252, min_periods=60).apply(lambda w: float((w[-1] >= w).mean()), raw=True)
    return pd.DataFrame({
        "gap": d["Open"] / pc - 1.0,                  # apertura vs cierre previo
        "trend": c.shift(1) / c.shift(6) - 1.0,       # momentum a 5 dias
        "volr": pct,                                   # regimen de volatilidad
        "dow": pd.Series(d.index.dayofweek, index=d.index),
    })


def binear(ind: pd.DataFrame) -> pd.DataFrame:
    b = pd.DataFrame(index=ind.index)
    b["gap_bin"] = ind["gap"].map(_bin_gap)
    b["trend_bin"] = ind["trend"].map(_bin_trend)
    b["vol_regime_bin"] = ind["volr"].map(
        lambda p: "desconocido" if pd.isna(p) else
        ("vol_baja" if p < 0.34 else ("vol_media" if p < 0.67 else "vol_alta")))
    b["dow"] = ind["dow"].map(lambda x: ["Lun", "Mar", "Mie", "Jue", "Vie", "Sab", "Dom"][int(x)])
    return b


# ==================== MOTOR ====================
def construir(diario: pd.DataFrame, fh: pd.DataFrame | None, horizonte: str):
    """Une condiciones con resultado. horizonte: 'intradia' o 'diario'."""
    b = binear(indicadores(diario))
    b.index = pd.to_datetime(b.index).normalize()

    if horizonte == "intradia":
        f = fh.copy()
        f.index = pd.to_datetime(f.index).normalize()
        df = b.join(f, how="inner").dropna(subset=["open", "fh_max", "fh_close"])
        if df.empty:
            return df
        df["exito"] = (df["fh_max"] >= df["open"] * (1 + UMBRAL_INTRADIA)).astype(int)
        df["ret"] = df["fh_close"] / df["open"] - 1.0
    else:  # diario: comprar en la apertura, salir en el cierre del mismo dia
        df = b.join(diario[["Open", "Close"]], how="inner").dropna()
        if df.empty:
            return df
        df["ret"] = df["Close"] / df["Open"] - 1.0
        df["exito"] = (df["ret"] > 0).astype(int)
    return df.dropna(subset=["exito", "ret"])


def evaluar_hoy(df: pd.DataFrame, cond_hoy: dict):
    """Tasa condicional + expectativa neta para la condicion de hoy."""
    if df.empty:
        return None
    base_e, base_n = int(df["exito"].sum()), len(df)
    mask = np.ones(len(df), bool)
    for f in FEATURES:
        mask &= (df[f].values == cond_hoy.get(f, "desconocido"))
    sub = df.loc[mask]
    if len(sub) < MIN_MUESTRA:
        return {"n": len(sub), "suficiente": False}

    lo, hi, p = wilson(int(sub["exito"].sum()), len(sub))
    base = base_e / base_n
    m, mlo, mhi = media_ic(sub["ret"].values)
    return {
        "suficiente": True, "n": len(sub), "tasa": p, "lo": lo, "hi": hi,
        "base": base, "lift": p - base,
        "pval": p_binomial(int(sub["exito"].sum()), len(sub), base),
        "bruta": m, "bruta_lo": mlo, "bruta_hi": mhi,
        "neta": m - COSTO_IDA_VUELTA,
        "neta_lo": mlo - COSTO_IDA_VUELTA,
        "cond": {f: cond_hoy.get(f, "?") for f in FEATURES},
    }


def condicion_hoy(diario: pd.DataFrame) -> dict:
    return binear(indicadores(diario)).iloc[-1].to_dict()


# ==================== VALIDACION WALK-FORWARD ====================
def walk_forward(df: pd.DataFrame, train_frac: float = 0.6):
    """Aprende condiciones favorables en el 60% viejo, mide en el 40% nuevo."""
    if len(df) < max(150, MIN_MUESTRA * 4):
        return None
    df = df.sort_index()
    k = int(len(df) * train_frac)
    tr, te = df.iloc[:k], df.iloc[k:]
    base_tr = tr["exito"].mean()

    favorables = set()
    for clave, s in tr.groupby(list(FEATURES))["exito"]:
        if len(s) < MIN_MUESTRA:
            continue
        lo, _, _ = wilson(int(s.sum()), len(s))
        if lo > base_tr and (s.mean() - base_tr) >= LIFT_MINIMO:
            favorables.add(clave if isinstance(clave, tuple) else (clave,))
    if not favorables:
        return {"favorables": 0}

    claves_te = list(zip(*[te[f].values for f in FEATURES]))
    marc = np.array([c in favorables for c in claves_te])
    m = te.loc[marc]
    if m.empty:
        return {"favorables": len(favorables), "n_oos": 0}

    bruta, blo, bhi = media_ic(m["ret"].values)
    return {
        "favorables": len(favorables), "n_oos": len(m),
        "tasa_oos": m["exito"].mean(), "base_oos": te["exito"].mean(),
        "lift_oos": m["exito"].mean() - te["exito"].mean(),
        "bruta": bruta, "bruta_lo": blo, "bruta_hi": bhi,
        "neta": bruta - COSTO_IDA_VUELTA,
        "neta_lo": blo - COSTO_IDA_VUELTA,
    }


def veredicto(r):
    if r is None:
        return "SIN DATOS", "Historial insuficiente para validar."
    if r.get("favorables", 0) == 0:
        return ("SIN CONDICIONES FAVORABLES",
                "Ninguna condicion supero la tasa base ni en entrenamiento. "
                "No hay senal que validar. NO OPERAR.")
    if r.get("n_oos", 0) < MIN_MUESTRA:
        return ("MUESTRA INSUFICIENTE",
                f"Solo {r.get('n_oos', 0)} dias marcados fuera de muestra. No concluyente.")
    if r["lift_oos"] > 0 and r["neta_lo"] > 0:
        return ("EVIDENCIA A FAVOR",
                "La ventaja sobrevivio fuera de muestra Y el limite inferior de la "
                "expectativa neta es positivo. Es lo mas fuerte que este metodo puede "
                "dar. Aun asi: un solo corte. Empieza con tamano minimo.")
    if r["lift_oos"] > 0 and r["neta"] > 0:
        return ("EVIDENCIA DEBIL",
                "Expectativa neta positiva pero su intervalo cruza el cero: no se "
                "distingue de cero con confianza. Insuficiente para arriesgar capital.")
    return ("SIN EVIDENCIA - NO OPERAR",
            "La ventaja no sobrevivio fuera de muestra o se la comen los costos. "
            "Sirve como contexto, no como senal de entrada.")


# ==================== ALERTA ====================
def enviar(texto: str):
    print(texto)
    if not (TG_TOKEN and TG_CHAT):
        log.warning("Sin credenciales de Telegram: solo se imprimio.")
        return
    try:
        import requests
        r = requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                          json={"chat_id": TG_CHAT, "text": texto,
                                "disable_web_page_preview": True}, timeout=20)
        if r.status_code != 200:
            log.error("Telegram %s: %s", r.status_code, r.text[:200])
    except Exception as e:
        log.error("Error enviando a Telegram: %s", e)


def pc(x):
    return f"{100 * x:.0f}%"


def bloque(nombre, res, extra=""):
    L = [f"— {nombre} —"]
    if extra:
        L.append(f"  {extra}")
    if res is None:
        L.append("  sin datos")
        return L
    if not res.get("suficiente"):
        L.append(f"  solo {res['n']} dias comparables (< {MIN_MUESTRA}): sin lectura")
        return L
    L.append(f"  sube {pc(res['tasa'])}  (n={res['n']}, IC95% {pc(res['lo'])}-{pc(res['hi'])})")
    L.append(f"  base {pc(res['base'])}  ·  lift {100 * res['lift']:+.0f}pp  ·  p={res['pval']:.3f}")
    L.append(f"  expectativa bruta {100 * res['bruta']:+.3f}%  (IC {100 * res['bruta_lo']:+.3f}% a {100 * res['bruta_hi']:+.3f}%)")
    L.append(f"  costo {100 * COSTO_IDA_VUELTA:.3f}%  ->  NETA {100 * res['neta']:+.3f}%")
    if res["neta_lo"] > 0:
        L.append("  * el IC de la neta esta entero por encima de cero")
    elif res["neta"] <= 0:
        L.append("  * neta negativa: los costos se comen la ventaja")
    else:
        L.append("  * la neta cruza cero: no se distingue de cero")
    return L


def modo_diario():
    diario, h60, tk = datos_oro()
    fh = primera_hora(h60)
    cond = condicion_hoy(diario)
    gap = indicadores(diario)["gap"].iloc[-1]

    d_intra = construir(diario, fh, "intradia")
    d_dia = construir(diario, None, "diario")
    r_intra = evaluar_hoy(d_intra, cond)
    r_dia = evaluar_hoy(d_dia, cond)

    ahora = datetime.now(ZoneInfo(ZONA))
    u = f"{100 * UMBRAL_INTRADIA:.2f}".rstrip("0").rstrip(".")
    L = ["ORO — tasas base condicionales", ahora.strftime("%Y-%m-%d %H:%M ") + "COT",
         f"Instrumento: {tk}", ""]
    L.append(f"Condicion de hoy: {', '.join(str(cond.get(f)) for f in FEATURES)}")
    L.append(f"Gap de apertura: {100 * gap:+.2f}%" if pd.notna(gap) else "Gap: n/d")
    L.append("")
    L += bloque(f"INTRADIA: 1a hora tras {SESION_HORA_ET}:{SESION_MIN_ET:02d} ET, max >= +{u}%",
                r_intra, f"muestra total: {len(d_intra)} sesiones (~2 anos)")
    L.append("")
    L += bloque("DIARIO: cierre > apertura del mismo dia", r_dia,
                f"muestra total: {len(d_dia)} dias (mas confiable)")
    L.append("")
    L.append("-" * 32)
    L.append("Frecuencias HISTORICAS bajo condiciones parecidas, no probabilidades "
             "del futuro. Fijate en la NETA, no en el % de acierto. "
             "No es asesoria de inversion.")
    enviar("\n".join(L))


def modo_validar():
    diario, h60, tk = datos_oro()
    fh = primera_hora(h60)
    L = ["VALIDACION WALK-FORWARD — ORO", "=" * 32, f"Instrumento: {tk}", ""]
    for nombre, df in (("INTRADIA (1a hora)", construir(diario, fh, "intradia")),
                       ("DIARIO (apertura->cierre)", construir(diario, None, "diario"))):
        r = walk_forward(df)
        t, expl = veredicto(r)
        L.append(f"— {nombre} —  muestra: {len(df)}")
        if r and r.get("n_oos"):
            L.append(f"  lift fuera de muestra : {100 * r['lift_oos']:+.1f} pp  (n={r['n_oos']})")
            L.append(f"  tasa OOS {pc(r['tasa_oos'])} vs base {pc(r['base_oos'])}")
            L.append(f"  expect. bruta {100 * r['bruta']:+.3f}%  ·  costo {100 * COSTO_IDA_VUELTA:.3f}%")
            L.append(f"  expect. NETA  {100 * r['neta']:+.3f}%  (IC inf {100 * r['neta_lo']:+.3f}%)")
        L.append(f"  VEREDICTO: {t}")
        L.append(f"  {expl}")
        L.append("")
    L.append("-" * 32)
    L.append("Metodo: entrenar con el 60% mas antiguo, medir en el 40% final que el "
             "modelo nunca vio. No es asesoria de inversion.")
    enviar("\n".join(L))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if len(sys.argv) > 1 and sys.argv[1].lower().startswith("valid"):
            modo_validar()
        else:
            modo_diario()
    except Exception as e:
        import traceback
        traceback.print_exc()
        enviar(f"Sistema ORO fallo: {e}")
        sys.exit(1)
