"""
Recallr — Cost Breakeven Dashboard
====================================
Run:  streamlit run costs_calculations.py

All cost logic, simulation, and UI live here.

Cost model
──────────
Naive approach: every session re-sends ALL past raw tokens as context.
    → per-session input tokens = (sessions_so_far + 1) * tokens_per_session
    → total cost is quadratic in number of sessions

Recallr: fixed 3-stage pipeline per session, independent of history size.
    → total cost is linear in number of sessions
"""

from dataclasses import dataclass
from typing import Optional
import streamlit as st
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
#  Per-session cost functions
#  Called once per session inside run_simulation() so that the quadratic
#  growth of the naive approach is captured accurately day by day.
# ─────────────────────────────────────────────────────────────────────────────

def naive_cost_per_session(
    sessions_already_done: int,
    num_of_chat_exchanges_per_session: int,
    num_of_tokens_per_chat_exchange: int,
    llm_cost_per_m_input: float,
) -> float:
    """
    Cost of one session under the naive approach.

    The naive approach has no memory system — it concatenates ALL past raw
    conversation tokens into the context window on every call.
    Session number (sessions_already_done + 1) must therefore send:

        input tokens = (sessions_already_done + 1) * messages * tokens_per_exchange

    Per-session cost grows linearly with history; total cost grows quadratically.
    """
    return (
        (
            (sessions_already_done + 1)           # all sessions so far + current one
            *
            num_of_chat_exchanges_per_session      # exchanges per session
            *
            num_of_tokens_per_chat_exchange       # tokens per user+assistant pair
        )
        /
        1_000_000  # per million tokens
        *
        llm_cost_per_m_input  # $ per million input tokens
        # note: output tokens (assistant responses) are identical
        # so they are excluded here — they do not affect the breakeven comparison
    )


def recallr_cost_per_session(
    num_of_chat_exchanges_per_session: int,
    num_of_tokens_per_chat_exchange: int,
    num_of_memories_extracted_per_session: int,
    llm_cost_per_m_input: float,
    llm_cost_per_m_output: float,
    reasoning: bool,
) -> float:
    """
    Cost of one session under Recallr.

    Cost is *constant* — it does not grow as the memory bank grows, because
    the graph DB retrieval is O(n) in time but the LLM only ever sees a fixed
    number of retrieved memories regardless of total bank size.

    Stage 0 — query (during session)
        global system prompt + retrieve relevant memories → send to LLM for the actual conversation
    Stage 1 — memory generation
        after session ends, extract new memories from the raw transcript
    Stage 2 — decision making
        one LLM call per extracted memory to decide add / update / delete
    Stage 3 — decision execution
        apply all changes to the graph DB
    """
    stage_0_total_input_tokens = 0
    for i in range(1, num_of_chat_exchanges_per_session + 1):  # i = turn number (1-indexed)
        stage_0_total_input_tokens += (
            2500                                  # global system prompt (constant size)
            +
            2000                                  # retrieved context
            +
            num_of_tokens_per_chat_exchange * i   # current session conversation so far (grows each turn)
        )
    stage_0 = (
        stage_0_total_input_tokens
        /
        1_000_000  # per million tokens
        *
        llm_cost_per_m_input  # $ per million input tokens
        # note: output tokens (assistant responses) are identical
        # so they are excluded here — they do not affect the breakeven comparison
    )

    stage_1 = (
        # input: memory generation system prompt + raw session transcript
        (
            (
                1500  # memory generation prompt tokens
                +
                (
                    num_of_chat_exchanges_per_session   # number of exchanges
                    *
                    num_of_tokens_per_chat_exchange    # tokens per exchange
                )
            )
            /
            1_000_000  # per million tokens
            *
            llm_cost_per_m_input  # $ per million input tokens
        )
        +
        # output: (reasoning trace +) extracted memory candidates
        (
            (
                3000  # internal reasoning + memory generation output tokens
                if reasoning
                else 300  # no internal reasoning, so fewer output tokens
            )
            /
            1_000_000  # per million tokens
            *
            llm_cost_per_m_output  # $ per million output tokens
        )
    )

    stage_2 = (
        num_of_memories_extracted_per_session  # one LLM call per extracted memory
        *
        (
            # input: decision prompt + one memory candidate + relevant retrieved context
            (
                (
                    4000  # decision making prompt tokens
                    +
                    500   # tokens from each memory + retrieved memories
                )
                /
                1_000_000  # per million tokens
                *
                llm_cost_per_m_input  # $ per million input tokens
            )
            +
            # output: (reasoning trace +) add / update / delete decision
            (
                (
                    4000  # internal reasoning + decision making output tokens
                    if reasoning
                    else 200  # no internal reasoning, so fewer output tokens
                )
                /
                1_000_000  # per million tokens
                *
                llm_cost_per_m_output  # $ per million output tokens
            )
        )
    )

    stage_3 = (
        # input: execution prompt
        (
            2000  # average decision execution prompt tokens
            /
            1_000_000  # per million tokens
            *
            llm_cost_per_m_input  # $ per million input tokens
        )
        +
        # output: (reasoning trace +) write-back confirmation
        (
            (
                2000  # internal reasoning + decision execution output tokens
                if reasoning
                else 200  # no internal reasoning, so fewer output tokens
            )
            /
            1_000_000  # per million tokens
            *
            llm_cost_per_m_output  # $ per million output tokens
        )
    )

    return stage_0 + stage_1 + stage_2 + stage_3


# ─────────────────────────────────────────────────────────────────────────────
#  Simulation — day-by-day engine
#  Builds the time-series arrays needed for plotting.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimulationParams:
    # Global Params
    num_of_chat_exchanges_per_session: int           # turns per session
    num_of_tokens_per_chat_exchange: int            # tokens per user+assistant pair
    num_of_sessions_per_day: int                    # sessions per day
    num_of_days: int                                # simulation window
    llm_cost_per_m_input: float                     # USD per 1M input tokens
    llm_cost_per_m_output: float                    # USD per 1M output tokens
    # Recallr Specific Params
    num_of_memories_extracted_per_session: int      # new memories written per session
    recallr_reasoning: bool                         # enable chain-of-thought internally


@dataclass
class SimulationResult:
    days: list[int]                   # [1, 2, …, num_of_days]
    naive_daily: list[float]          # cost on each individual day — naive
    recallr_daily: list[float]        # cost on each individual day — Recallr
    naive_cumulative: list[float]     # running total — naive
    recallr_cumulative: list[float]   # running total — Recallr
    breakeven_day: Optional[int]      # first day where cumulative Recallr < naive
    total_naive: float                # final cumulative naive cost
    total_recallr: float              # final cumulative Recallr cost


def run_simulation(params: SimulationParams) -> SimulationResult:
    """
    Iterate day by day, calling naive_cost_per_session / recallr_cost_per_session
    for every session so that the quadratic cost growth of the naive approach
    is captured accurately over time.

    sessions_done tracks the global session index so that naive_cost_per_session
    receives the correct sessions_already_done value on each call.
    """
    days:          list[int]   = []
    naive_daily:   list[float] = []
    recallr_daily: list[float] = []
    naive_cum:     list[float] = []
    recallr_cum:   list[float] = []

    total_naive    = 0.0
    total_recallr  = 0.0
    sessions_done  = 0                    # 0-indexed; incremented after every session
    breakeven_day: Optional[int] = None

    for day in range(1, params.num_of_days + 1):
        day_naive   = 0.0
        day_recallr = 0.0

        for _ in range(params.num_of_sessions_per_day):
            day_naive += naive_cost_per_session(
                sessions_already_done=sessions_done,
                num_of_chat_exchanges_per_session=params.num_of_chat_exchanges_per_session,
                num_of_tokens_per_chat_exchange=params.num_of_tokens_per_chat_exchange,
                llm_cost_per_m_input=params.llm_cost_per_m_input,
            )
            day_recallr += recallr_cost_per_session(
                num_of_chat_exchanges_per_session=params.num_of_chat_exchanges_per_session,
                num_of_tokens_per_chat_exchange=params.num_of_tokens_per_chat_exchange,
                num_of_memories_extracted_per_session=params.num_of_memories_extracted_per_session,
                llm_cost_per_m_input=params.llm_cost_per_m_input,
                llm_cost_per_m_output=params.llm_cost_per_m_output,
                reasoning=params.recallr_reasoning,
            )
            sessions_done += 1

        total_naive   += day_naive
        total_recallr += day_recallr

        days.append(day)
        naive_daily.append(day_naive)
        recallr_daily.append(day_recallr)
        naive_cum.append(total_naive)
        recallr_cum.append(total_recallr)

        if breakeven_day is None and total_recallr < total_naive:
            breakeven_day = day

    return SimulationResult(
        days=days,
        naive_daily=naive_daily,
        recallr_daily=recallr_daily,
        naive_cumulative=naive_cum,
        recallr_cumulative=recallr_cum,
        breakeven_day=breakeven_day,
        total_naive=total_naive,
        total_recallr=total_recallr,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Recallr — Cost Breakeven",
    page_icon="📊",
    layout="wide",
)


def fmt_usd(v: float) -> str:
    if v >= 1_000_000: return f"${v / 1_000_000:.2f}M"
    if v >= 1_000:     return f"${v / 1_000:.2f}k"
    return f"${v:.2f}"


with st.sidebar:
    st.title("⚙️ Parameters")

    show_daily = st.toggle("Show per-day cost instead of cumulative", value=False)

    st.subheader("Session")
    messages_per_session = st.slider(
        "Chat messages per session (1 user + 1 assistant)",
        min_value=1, max_value=50, step=1, value=10,
        help="Number of user+assistant exchanges in one session.",
    )
    tokens_per_exchange = st.slider(
        "Tokens per chat exchange (1 user + 1 assistant)",
        min_value=50, max_value=2_000, step=50, value=400,
        help="Token count for one full user+assistant pair. "
             "The naive approach re-sends ALL past sessions' exchanges on every call, "
             "so this directly drives the quadratic cost growth.",
    )
    sessions_per_day = st.slider(
        "Sessions per day",
        min_value=1, max_value=200, step=1, value=5,
    )
    num_days = st.slider(
        "Simulation window (days)",
        min_value=30, max_value=365, step=7, value=60,
    )

    st.subheader("Recallr memory pipeline")
    memories_extracted = st.slider(
        "Memories extracted per session",
        min_value=1, max_value=30, step=1, value=5,
        help="How many new memory records Recallr writes to the graph DB after each session. "
             "Drives stage 2 cost (one LLM call per extracted memory).",
    )
    reasoning = st.toggle(
        "Enable internal reasoning for Recallr (recommended)",
        value=True,
        help="Adds chain-of-thought thinking tokens to all three post-session pipeline stages.",
    )

    st.subheader("LLM pricing (USD / 1M tokens)")
    cost_input = st.slider(
        "Input token cost ($)",
        min_value=0.10, max_value=30.0, step=0.10, value=3.0, format="$%.2f",
    )
    cost_output = st.slider(
        "Output token cost ($)",
        min_value=0.10, max_value=60.0, step=0.10, value=15.0, format="$%.2f",
    )


params = SimulationParams(
    num_of_chat_exchanges_per_session=messages_per_session,
    num_of_tokens_per_chat_exchange=tokens_per_exchange,
    num_of_sessions_per_day=sessions_per_day,
    num_of_days=num_days,
    num_of_memories_extracted_per_session=memories_extracted,
    llm_cost_per_m_input=cost_input,
    llm_cost_per_m_output=cost_output,
    recallr_reasoning=reasoning,
)

result: SimulationResult = run_simulation(params)

st.title("📊 Recallr — Cost Breakeven Analysis")
st.caption(
    "**Naive approach**: every session re-sends the *entire* raw conversation history "
    "as context → cost grows **quadratically** with sessions.  "
    "**Recallr**: fixed pipeline overhead per session regardless of history size → "
    "cost grows **linearly**."
)

savings     = result.total_naive - result.total_recallr
savings_pct = (savings / result.total_naive * 100) if result.total_naive > 0 else 0.0

c1, c2, c3, c4 = st.columns(4)

with c1:
    if result.breakeven_day is not None:
        st.metric("Breakeven", f"Day {result.breakeven_day}")
    else:
        st.metric("Breakeven", "Recallr is Expensive")

with c2:
    st.metric(
        f"Savings at day {num_days}",
        fmt_usd(savings),
        delta=f"{savings_pct:.1f}% cheaper" if savings > 0 else f"{-savings_pct:.1f}% more expensive",
        delta_color="normal" if savings > 0 else "inverse",
    )

with c3:
    st.metric("Naive total", fmt_usd(result.total_naive))

with c4:
    st.metric("Recallr total", fmt_usd(result.total_recallr))

if result.breakeven_day is None and not show_daily:
    st.warning(
        "Recallr hasn't broken even within this window.  "
        "Try more sessions per day or a longer simulation window."
    )

st.divider()

naive_data   = result.naive_daily   if show_daily else result.naive_cumulative
recallr_data = result.recallr_daily if show_daily else result.recallr_cumulative
y_label      = "Cost per day (USD)"  if show_daily else "Cumulative cost (USD)"
chart_title  = "Daily LLM Cost Over Time" if show_daily else "Cumulative LLM Cost Over Time"

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=result.days, y=naive_data,
    name="Naive (full history as context)",
    line=dict(color="red", width=2.5),
    fill="tozeroy",
    fillcolor="rgba(239,68,68,0.08)",
    hovertemplate="Day %{x}<br>$%{y:,.4f}<extra>Naive</extra>",
))

fig.add_trace(go.Scatter(
    x=result.days, y=recallr_data,
    name="Recallr",
    line=dict(color="white", width=2.5),
    fill="tozeroy",
    fillcolor="rgba(34,197,94,0.08)",
    hovertemplate="Day %{x}<br>$%{y:,.4f}<extra>Recallr</extra>",
))

if result.breakeven_day is not None and not show_daily:
    fig.add_vline(
        x=result.breakeven_day,
        line=dict(color="#f59e0b", width=2, dash="dash"),
        annotation_text=f" Breakeven — Day {result.breakeven_day}",
        annotation_position="top right",
        annotation_font_color="#f59e0b",
    )

fig.update_layout(
    title=chart_title,
    xaxis_title="Day",
    yaxis_title=y_label,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template="plotly_dark",
    height=500,
    margin=dict(l=0, r=0, t=60, b=0),
)

st.plotly_chart(fig, use_container_width=True)
