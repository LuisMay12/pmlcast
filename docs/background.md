# Background: the Mexican power market and the CENACE data

Context for reading the rest of the project: what a nodal price is, where its
three components come from, when CENACE publishes it, and why the collector
asks for prices in windows of seven days and twenty nodes. Every rule below
is quoted from an official CENACE document; the sources are listed at the
end.

## The market in one paragraph

CENACE (*Centro Nacional de Control de Energía*) operates Mexico's
wholesale electricity market (MEM). It does not set a single national price:
it computes a price for every **NodoP**, a pricing node that stands for one
or several substations where energy is injected or withdrawn, and for every
hour. The data dictionary counts more than 2,500 active NodosP; the
catalogue snapshot this project uses (`v20260218`) lists 2,596. The national
grid (SIN) and the two isolated Baja California systems (BCA, BCS) are
priced separately.

## Two markets, two schedules

| | Day-ahead market (MDA) | Real-time market (MTR) |
|---|---|---|
| What it settles | The schedule for every hour of the next day; binding for each MWh scheduled | Deviations from that schedule |
| Offers | Open 7 days before the operating day, close at **10:00 the day before** | — |
| Results | Binding programmes issued **before 17:00 the day before** | Published **7 days after** the operating day |

The prices themselves reach the public download service "one day before the
operating day". In practice the published price file carries a later
version stamp: the SIN MDA file for 1 October 2026 is stamped
`v2026 09 30_18 45 02`, that is 18:45 on the previous evening.

That timeline is the whole case for PMLcast. By the time tomorrow's prices
are public, offers closed hours earlier. A forecast made in the morning, from
prices that are already public, is the only view of tomorrow a participant
can act on.

It is also why the project forecasts MDA and not MTR. MTR is published a
week late, so at forecast time the most recent real-time price is seven days
old, and a model cannot read its own recent history the way the MDA model
does. Forecasting MTR is planned as a second phase that conditions on the
already-published MDA price instead.

## Where the three components come from

Each price CENACE publishes is the sum of three components:

```
PML = energy + losses + congestion        (MXN/MWh)
```

They are not accounting categories added afterwards. They come out of the
optimisation CENACE solves for the day-ahead market (the AU-MDA model, a
security-constrained unit commitment). Once the on/off decisions are fixed,
the model is solved again with only continuous variables, and every
constraint gets a **dual variable**: a shadow price, the cost of tightening
that constraint by one unit. The Manual de Mercado de Energía de Corto Plazo
(section 4.4.8) builds each component from those shadow prices:

- **Energy.** The shadow price of the power-balance constraint at the
  system's **reference node**, which equals that of the whole system's
  balance. It is the cost of one more MWh with no network in the way, so it
  is **the same at every node of a system** in a given hour.
- **Congestion.** For every transmission constraint that is binding, its
  shadow price times the sensitivity of the flow on that line to an
  injection at the node (balanced at the reference node). It is zero when
  no line is at its limit, and it can be negative: a node whose injection
  relieves a congested line is paid less.
- **Losses.** The marginal effect of an injection at the node on the
  network's total losses, again balanced at the reference node. Lines heat
  up and waste a share of what they carry; a node far from generation needs
  more energy produced elsewhere to deliver one MWh, so its losses
  component is high. A node next to large plants can have a **negative**
  losses component, because injecting there shortens the distance energy
  has to travel.

The split between components depends on which node is chosen as reference,
but their sum, the price, does not.

### The same hour at three nodes

19 September 2026, 21:00, from this project's data:

| Node | Price | Energy | Losses | Congestion |
|---|---|---|---|---|
| 08MDP-230 Mérida Potencia | 4,372.09 | 3,992.57 | **+380.95** | −1.43 |
| 01TUL-400 Tula | 4,076.99 | 3,992.57 | +85.41 | −0.98 |
| 06ALT-400 Altamira | 3,762.81 | 3,992.57 | **−219.30** | −10.46 |

The energy component is identical at all three. What separates them is the
grid: Mérida sits at the end of long transmission lines into the Yucatán
peninsula and pays for the losses on the way, while Altamira is next to
large thermal plants on the Gulf coast. That difference, more than any
change in demand, is why the peninsular nodes are the expensive ones.

### Why the model reads the components

The model receives the three components alongside the price (inputs 2–4 of
each hour in `seq`). The energy component carries the system-wide signal,
shared by every node, while losses and congestion carry what is local to
one node. Giving the network both lets it tell a day that is expensive
everywhere from one that is expensive here because of the grid.

## The download service and the seven-day window

Prices come from **SW-PML**, CENACE's public web service. A request is a URL:

```
https://ws01.cenace.gob.mx:8082/SWPML/SIM/<system>/<market>/<nodes>/<y>/<m>/<d>/<y>/<m>/<d>/<format>
https://ws01.cenace.gob.mx:8082/SWPML/SIM/SIN/MDA/01PLO-115/2017/11/07/2017/11/07/XML
```

Each response lists, per node and hour, `pml`, `pml_ene`, `pml_per` and
`pml_cng`. The technical manual sets the limits the collector is built
around:

| Rule (Manual Técnico SW-PML) | Consequence in `pmlcast/cenace.py` |
|---|---|
| "La lista de NodosP podrá considerar de **1 a 20** NodosP" | `MAX_NODES_PER_REQUEST = 20`: nodes go out in batches of twenty |
| "El periodo de consulta podrá considerar de **1 a 7** Días de Operación" | `MAX_DAYS_PER_REQUEST = 7`: history is split into seven-day windows |
| MDA available from 29 Jan 2016 (SIN), 27 Jan 2016 (BCA), 23 Mar 2016 (BCS); MTR from 27 Jan 2017 | `config.EPOCHS`: 29 Jan 2016 for MDA and 27 Jan 2017 for MTR, the SIN dates, are the earliest the collector asks for |
| "El CENACE podrá deshabilitar el SW-PML sin previo aviso, por uso indebido" | Requests run one at a time with a pause, retries back off, and every response is kept in bronze so nothing is asked twice |

A wider range does not degrade gracefully: it is refused with the message
*"No se pueden mostrar datos con un lapso mayor a 7 dias"*.

The windows are anchored to the market's start date rather than to the
first day requested, so two runs that overlap ask for exactly the same
windows and the second finds them already stored. Collecting the 99 nodes
from 2016 to September 2026 means about 3,900 days, so 556 windows times 5
batches: roughly **2,800 requests**. At 3 to 19 seconds each, run in
sequence, that is a few hours to most of a night, which is why the
collection was run overnight and is cached rather than repeated.

The deployed service uses the same collector for a single node: when a
requested node's stored prices stop short, it asks for the missing days,
usually one or two requests.

## Sources

- CENACE. *Manual Técnico: Uso de Servicio Web para descarga de Precios
  Marginales Locales (SW-PML)*, 24 June 2022.
  <https://www.cenace.gob.mx/DocsMEM/2022-06-24%20Manual%20T%C3%A9cnico%20SW-PML.pdf>
- SENER. *Manual de Mercado de Energía de Corto Plazo*, Diario Oficial de
  la Federación, 17 June 2016. Sections 2.4 (MDA timeline) and 4.4.7–4.4.8
  (price components).
  <https://www.cenace.gob.mx/Docs/MercadoCortoPlazo/Manual%20de%20Mercado%20de%20Energ%C3%ADa%20de%20Corto%20Plazo%20(DOF%20SENER%2017-Jun-16).pdf>
- CENACE. *Datos Abiertos — Diccionario de Datos, Mercado de Energía de
  Corto Plazo*.
  <https://www.cenace.gob.mx/Docs/Cenace/Documentos/DatosAbiertos/DICCIONARIO/Diccionario_Datos_Abiertos_CENACE_MECP.pdf>
- CENACE. *Catálogo de NodosP*.
  <https://www.cenace.gob.mx/Paginas/SIM/NodosP.aspx>
