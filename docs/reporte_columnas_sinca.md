# Reporte de exploración — Archivo SINCA

**Generado:** 2026-06-15 23:11:03
**Archivo:** `sinca_puente_alto_mp25_2022_2026.csv`
**Formato SINCA detectado:** B

---

## 1. Resumen general

| Métrica | Valor |
|---|---|
| Filas totales | 1,625 |
| Columnas totales | 5 |
| Filas duplicadas | 0 |
| Columnas con algún nulo | 3 |
| Granularidad | Horaria (requiere agregación diaria) |

---

## 2. Columnas disponibles

| Columna | Tipo pandas | Nulos | % Nulos | Ejemplo |
|---|---|---|---|---|
| `FECHA (YYMMDD)` | int64 | 0 | 0.0% | 220101 |
| `HORA (HHMM)` | int64 | 0 | 0.0% | 0 |
| `Registros validados` | float64 | 233 | 14.34% | 10.0 |
| `Registros preliminares` | float64 | 1,466 | 90.22% | 21.0 |
| `Registros no validados` | float64 | 1,625 | 100.0% | — |

---

## 3. Columnas relevantes para el proyecto

### Fecha / hora
- `FECHA (YYMMDD)`
- `HORA (HHMM)`

### MP 2,5 — material particulado fino
_No detectada._

### MP 10 — material particulado grueso
_No detectada._

### Estado de validación del registro
- `Registros validados`
- `Registros preliminares`
- `Registros no validados`

### Estación de monitoreo
_No detectada._

### Comuna / localidad
_No detectada._

### Otros gases (SO₂, NO₂, CO, O₃)
_No detectada._

---

## 4. Rango de fechas disponibles

### `FECHA (YYMMDD)`
- Registros válidos: **1,625**
- Desde: **2022-01-01**
- Hasta: **2026-06-13**

### `HORA (HHMM)`
- ⚠️ No parseable como fecha

---

## 5. Estadísticas descriptivas (columnas numéricas)

```
       FECHA (YYMMDD)  HORA (HHMM)  Registros validados  Registros preliminares  Registros no validados
count         1625.00       1625.0              1392.00                  159.00                     0.0
mean        238161.07          0.0                21.03                   21.01                     NaN
std          12951.84          0.0                12.50                   14.72                     NaN
min         220101.00          0.0                 2.00                    6.00                     NaN
25%         230211.00          0.0                12.00                   11.00                     NaN
50%         240323.00          0.0                17.00                   16.00                     NaN
75%         250503.00          0.0                27.00                   23.00                     NaN
max         260613.00          0.0                79.00                   78.00                     NaN
```

---

## 6. Primeras 5 filas

```
   FECHA (YYMMDD)  HORA (HHMM)  Registros validados  Registros preliminares  Registros no validados
0          220101            0                 10.0                     NaN                     NaN
1          220102            0                  5.0                     NaN                     NaN
2          220103            0                  7.0                     NaN                     NaN
3          220104            0                  9.0                     NaN                     NaN
4          220105            0                 12.0                     NaN                     NaN
```

---

## 7. Notas importantes para el ETL

### Formato detectado: B
- Fecha y hora en columnas **separadas**: `FECHA (YYMMDD)` y `HORA (HHMM)`
- Formato de fecha: YYMMDD (ej: 260101 = 2026-01-01)
- Formato de hora: HHMM (ej: 0000 = 00:00)
- Los valores de MP 2,5 están en 3 columnas según validación:
  - `Registros validados` — usar estos preferentemente
  - `Registros preliminares` — usar si validados es nulo
  - `Registros no validados` — evitar salvo que no haya alternativa
- El ETL deberá combinar las 3 columnas en una sola `mp25`

### Tareas antes de escribir extract_sinca.py
- [ ] Descargar datos históricos 2022–2025 desde SINCA
- [ ] Verificar si todos los archivos históricos tienen el mismo formato
- [ ] Confirmar función de agregación diaria (promedio vs máximo)
- [ ] Decidir prioridad entre registros validados / preliminares / no validados