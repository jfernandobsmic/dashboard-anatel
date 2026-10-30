"""
Dashboard - Reclamações Anatel (dataset 2020-2026 + referência do Panorama 2025)

Estrutura esperada:
    dashboard_anatel.py
    dados/fato_principal.parquet
    dados/fato_problema.parquet
    dados/fato_cidade.parquet

Executar:  streamlit run dashboard_anatel.py
Requisitos: pip install "streamlit>=1.50" pandas plotly pyarrow
(Para regerar os parquets a partir do CSV bruto: python preparar_dados.py)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Reclamações Anatel", page_icon="📡", layout="wide")

DADOS = Path(__file__).parent / "dados"
AZUL, AMARELO, VERDE, VERMELHO = "#2A4565", "#FDB022", "#2E9B5A", "#D64545"
PALETA = [AZUL, AMARELO, VERDE, "#E8833A", "#4A90D9", "#8E5BA8", "#E83E9B", "#7CC04B", "#9DB4CE", "#6B6B6B"]
MESES = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]

SERV_MAP = {"SCM": "Banda Larga Fixa", "SMP_Pós": "Celular Pós-Pago", "SMP_Pré": "Celular Pré-Pago",
            "STFC": "Telefone Fixo", "SeAC": "TV por Assinatura", "Outros": "Outros"}
CANAL_MAP = {"Mobile app": "Aplicativo", "Usuário web": "Site", "Call center": "Call center",
             "Whatsapp": "WhatsApp", "Sei": "SEI"}
MARCA_ESPECIAL = {"OI TV (MILETO)": "Oi TV (Mileto)", "NÃO LOCALIZEI A MINHA EMPRESA": "Não localizei a minha empresa",
                  "DEMAIS MARCAS": "Demais marcas", "GIGA+": "Giga+"}
MARCAS_GENERICAS = ["Não localizei a minha empresa", "Demais marcas", "Outros"]


# ----------------------------------------------------------------------------
# CARGA DE DADOS
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Carregando dados...")
def carregar():
    p = pd.read_parquet(DADOS / "fato_principal.parquet")
    q = pd.read_parquet(DADOS / "fato_problema.parquet")
    c = pd.read_parquet(DADOS / "fato_cidade.parquet")
    nomes = {m: MARCA_ESPECIAL.get(m, m.title()) for m in set(p.marca) | set(q.marca)}
    for d in (p, q, c):
        d["servico"] = d["servico"].map(SERV_MAP).fillna(d["servico"])
    for d in (p, q):
        d["marca"] = d["marca"].map(nomes)
    p["canal"] = p["canal"].map(CANAL_MAP).fillna(p["canal"])
    return p, q, c


if not (DADOS / "fato_principal.parquet").exists():
    st.error("Pasta `dados/` não encontrada. Coloque os arquivos .parquet ao lado deste script "
             "(ou execute `python preparar_dados.py`).")
    st.stop()

P, Q, C = carregar()
ANO_MIN, ANO_MAX = int(P.ano.min()), int(P.ano.max())
ULT_MES = int(P[P.ano == ANO_MAX].mes.max())
PARCIAL = ULT_MES < 12


def lbl(y):
    return f"{y} (jan–{MESES[ULT_MES - 1]})" if (y == ANO_MAX and PARCIAL) else str(y)


fmt = lambda v: "—" if pd.isna(v) else f"{v:,.0f}".replace(",", ".")
fpct = lambda v: "—" if pd.isna(v) else f"{v:+.1f}%".replace(".", ",")
fabs = lambda v: "—" if pd.isna(v) else f"{v:+,.0f}".replace(",", ".")


def filtra(df, **kw):
    for col, sel in kw.items():
        if sel and col in df.columns:
            df = df[df[col].isin(sel)]
    return df


def comparar(df, chave, a, b):
    """Tabela com valores em dois anos, variação absoluta e %."""
    p = df[df.ano.isin([a, b])].pivot_table(index=chave, columns="ano", values="n", aggfunc="sum", fill_value=0)
    for y in (a, b):
        if y not in p.columns:
            p[y] = 0
    p = p[[a, b]].copy()
    p["var_abs"] = p[b] - p[a]
    p["var_pct"] = np.where(p[a] > 0, (p[b] / p[a] - 1) * 100, np.nan)
    return p.reset_index()


def estilo(fig, h=400, legenda=True):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=55, b=10), legend_title_text="",
                      legend=dict(orientation="h", y=-0.22) if legenda else None, showlegend=legenda)
    return fig


# ----------------------------------------------------------------------------
# BARRA LATERAL (FILTROS)
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("🔎 Filtros")
    f_tipo = st.multiselect("Tipo de atendimento", sorted(P.tipo.unique()), default=["Reclamação"],
                            help="Vazio = todos os tipos.")
    f_anos = st.slider("Período (anos)", ANO_MIN, ANO_MAX, (2023, ANO_MAX))
    f_serv = st.multiselect("Serviço", [s for s in SERV_MAP.values() if s in set(P.servico)])
    marcas_ord = P.groupby("marca").n.sum().sort_values(ascending=False).index.tolist()
    f_marca = st.multiselect("Marca", marcas_ord)
    f_uf = st.multiselect("UF", sorted(P.uf.dropna().unique()))
    f_canal = st.multiselect("Canal de entrada", sorted(P.canal.unique()))
    f_cond = st.multiselect("Condição", sorted(P.condicao.unique()))
    st.divider()
    st.subheader("Comparação entre anos")
    anos_lista = list(range(ANO_MIN, ANO_MAX + 1))
    ano_a = st.selectbox("Ano base", anos_lista, index=anos_lista.index(2024) if 2024 in anos_lista else 0, format_func=lbl)
    ano_b = st.selectbox("Ano de comparação", anos_lista, index=anos_lista.index(2025) if 2025 in anos_lista else len(anos_lista) - 1, format_func=lbl)
    if ano_a == ano_b:
        st.error("Escolha anos diferentes para comparar.")
        st.stop()
    if PARCIAL:
        st.caption(f"⚠️ {ANO_MAX} é parcial (janeiro a {MESES[ULT_MES - 1]}). Comparações com anos completos são enviesadas; "
                   "use o comparativo 'mesmo período' na aba Visão geral.")

PF = filtra(P, tipo=f_tipo, servico=f_serv, marca=f_marca, uf=f_uf, canal=f_canal, condicao=f_cond)  # sem período
PFp = PF[PF.ano.between(*f_anos)]
QF = filtra(Q, servico=f_serv, marca=f_marca)
CF = filtra(C, servico=f_serv, uf=f_uf)

st.title("📡 Reclamações Anatel")
st.caption(f"Dataset de solicitações registradas no Anatel Consumidor, {ANO_MIN}–{ANO_MAX} "
           f"(últimos dados: {MESES[ULT_MES - 1]}/{ANO_MAX}). Fonte: Anatel · SRC/RCTS.")

if PFp.empty:
    st.warning("Nenhum registro para os filtros selecionados.")
    st.stop()

tabs = st.tabs(["📊 Visão geral", "🗂️ Assuntos e problemas", "🏢 Prestadoras", "🗺️ Geografia",
                "📲 Canais e perfil", "📘 Referência do Panorama 2025"])

# ----------------------------------------------------------------------------
# 1. VISÃO GERAL
# ----------------------------------------------------------------------------
with tabs[0]:
    tot_a, tot_b = PF[PF.ano == ano_a].n.sum(), PF[PF.ano == ano_b].n.sum()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"Total no período ({f_anos[0]}–{f_anos[1]})", fmt(PFp.n.sum()))
    k2.metric(f"{lbl(ano_b)}", fmt(tot_b), f"{fpct((tot_b / tot_a - 1) * 100 if tot_a else np.nan)} vs {lbl(ano_a)}", delta_color="inverse")
    if PARCIAL:
        y_now, y_prev = ANO_MAX, ANO_MAX - 1
        now = PF[(PF.ano == y_now) & (PF.mes <= ULT_MES)].n.sum()
        prev = PF[(PF.ano == y_prev) & (PF.mes <= ULT_MES)].n.sum()
        k3.metric(f"{y_now} vs {y_prev} (jan–{MESES[ULT_MES - 1]})", fmt(now), f"{fpct((now / prev - 1) * 100 if prev else np.nan)}", delta_color="inverse")
    else:
        k3.metric("Média mensal", fmt(PFp.groupby(["ano", "mes"]).n.sum().mean()))
    reab = PFp[PFp.condicao == "Reaberta"].n.sum() / PFp.n.sum() * 100
    k4.metric("Registros reabertos", f"{reab:.1f}%".replace(".", ","))

    c1, c2 = st.columns(2)
    with c1:
        a = PFp.groupby(["ano", "servico"], as_index=False).n.sum()
        a["Ano"] = a.ano.map(lbl)
        fig = px.bar(a, x="Ano", y="n", color="servico", title="Volume anual por serviço",
                     labels={"n": "Solicitações", "servico": "Serviço"}, color_discrete_sequence=PALETA)
        tot = a.groupby("Ano", as_index=False).n.sum()
        fig.add_trace(go.Scatter(x=tot["Ano"], y=tot.n, mode="text", text=tot.n.map(lambda v: f"{v/1e6:.2f} mi"),
                                 textposition="top center", showlegend=False))
        st.plotly_chart(estilo(fig), width="stretch")
    with c2:
        m = PFp.groupby(["ano", "mes"], as_index=False).n.sum()
        m["Ano"] = m.ano.map(lbl)
        fig = px.line(m, x="mes", y="n", color="Ano", markers=True, title="Sazonalidade mensal",
                      labels={"mes": "Mês", "n": "Solicitações"}, color_discrete_sequence=PALETA)
        fig.update_xaxes(tickmode="array", tickvals=list(range(1, 13)), ticktext=MESES)
        st.plotly_chart(estilo(fig), width="stretch")

    serie = PFp.groupby(["ano", "mes"], as_index=False).n.sum()
    serie["data"] = pd.to_datetime(dict(year=serie.ano, month=serie.mes, day=1))
    serie = serie.sort_values("data")
    serie["Média móvel 3m"] = serie.n.rolling(3).mean()
    fig = go.Figure()
    fig.add_bar(x=serie.data, y=serie.n, name="Mensal", marker_color="#9DB4CE")
    fig.add_scatter(x=serie.data, y=serie["Média móvel 3m"], name="Média móvel 3m", line=dict(color=AZUL, width=3))
    fig.update_layout(title="Série mensal")
    st.plotly_chart(estilo(fig, 360), width="stretch")

    st.subheader(f"Variação por serviço: {lbl(ano_a)} → {lbl(ano_b)}")
    t = comparar(PF, "servico", ano_a, ano_b).sort_values(ano_b, ascending=False)
    if PARCIAL:
        ya = PF[(PF.ano == ANO_MAX - 1) & (PF.mes <= ULT_MES)].groupby("servico").n.sum()
        yb = PF[(PF.ano == ANO_MAX) & (PF.mes <= ULT_MES)].groupby("servico").n.sum()
        t[f"{ANO_MAX} vs {ANO_MAX - 1} (mesmo período)"] = t.servico.map(lambda s: fpct((yb.get(s, 0) / ya.get(s, np.nan) - 1) * 100))
    cc1, cc2 = st.columns([3, 2])
    with cc1:
        tv = t.sort_values("var_pct")
        fig = px.bar(tv, x="var_pct", y="servico", orientation="h", title="Variação % por serviço",
                     text=tv.var_pct.map(lambda v: fpct(v)), labels={"var_pct": "Variação %", "servico": ""})
        fig.update_traces(marker_color=[VERDE if v < 0 else VERMELHO for v in tv.var_pct])
        st.plotly_chart(estilo(fig, 340, False), width="stretch")
    with cc2:
        show = t.rename(columns={"servico": "Serviço", "var_abs": "Δ abs.", "var_pct": "Δ %"}).copy()
        show.columns = [lbl(c) if isinstance(c, (int, np.integer)) else c for c in show.columns]
        show["Δ abs."] = show["Δ abs."].map(fabs)
        show["Δ %"] = show["Δ %"].map(fpct)
        for c in show.columns:
            if c[:2] == "20" and "vs" not in c:
                show[c] = show[c].map(fmt)
        st.dataframe(show, hide_index=True, width="stretch")

    st.download_button("⬇️ Baixar série mensal filtrada (CSV)",
                       serie[["ano", "mes", "n"]].rename(columns={"n": "solicitacoes"}).to_csv(index=False, sep=";").encode("utf-8-sig"),
                       "serie_mensal_filtrada.csv", "text/csv")

# ----------------------------------------------------------------------------
# 2. ASSUNTOS E PROBLEMAS
# ----------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Assuntos")
    top_ass = PFp.groupby("assunto").n.sum().sort_values(ascending=False).head(6).index.tolist()
    sel_ass = st.multiselect("Assuntos exibidos", sorted(PFp.assunto.unique()), default=top_ass)
    a = PFp[PFp.assunto.isin(sel_ass)].groupby(["ano", "assunto"], as_index=False).n.sum()
    a["Ano"] = a.ano.map(lbl)
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = px.bar(a, x="Ano", y="n", color="assunto", barmode="group", title="Evolução por assunto",
                     labels={"n": "Solicitações", "assunto": "Assunto"}, color_discrete_sequence=PALETA)
        st.plotly_chart(estilo(fig, 440), width="stretch")
    with c2:
        ta = comparar(PF, "assunto", ano_a, ano_b).sort_values("var_abs")
        ta = pd.concat([ta.head(5), ta.tail(8)]).drop_duplicates("assunto")
        fig = px.bar(ta, x="var_abs", y="assunto", orientation="h", title=f"Maiores variações ({lbl(ano_a)} → {lbl(ano_b)})",
                     labels={"var_abs": "Δ absoluta", "assunto": ""})
        fig.update_traces(marker_color=[VERDE if v < 0 else VERMELHO for v in ta.var_abs])
        st.plotly_chart(estilo(fig, 440, False), width="stretch")

    # participação (heatmap)
    pv = PFp[PFp.assunto.isin(top_ass)].pivot_table(index="assunto", columns="ano", values="n", aggfunc="sum", fill_value=0)
    share = pv / PFp.groupby("ano").n.sum().reindex(pv.columns) * 100
    share.columns = [lbl(c) for c in share.columns]
    fig = px.imshow(share, text_auto=".1f", aspect="auto", color_continuous_scale="Blues",
                    title="Participação de cada assunto no total do ano (%)")
    st.plotly_chart(estilo(fig, 320, False), width="stretch")

    st.subheader("Problemas")
    st.caption("Base de reclamações detalhada por problema (respeita filtros de serviço e marca).")
    ass_opts = ["Todos"] + sorted(QF.assunto.unique())
    ass_p = st.selectbox("Assunto", ass_opts)
    QP = QF if ass_p == "Todos" else QF[QF.assunto == ass_p]
    tp = comparar(QP, "problema", ano_a, ano_b)
    c1, c2 = st.columns(2)
    with c1:
        top = tp.sort_values(ano_b, ascending=False).head(15).sort_values(ano_b)
        fig = px.bar(top, x=ano_b, y="problema", orientation="h", title=f"Top 15 problemas em {lbl(ano_b)}",
                     labels={ano_b: "Reclamações", "problema": ""})
        fig.update_traces(marker_color=AZUL)
        st.plotly_chart(estilo(fig, 520, False), width="stretch")
    with c2:
        v = tp[(tp[[ano_a, ano_b]].max(axis=1) >= 500)].sort_values("var_abs")
        v = pd.concat([v.head(7), v.tail(7)]).drop_duplicates("problema")
        fig = px.bar(v, x="var_abs", y="problema", orientation="h", title=f"Maiores altas e quedas ({lbl(ano_a)} → {lbl(ano_b)})",
                     labels={"var_abs": "Δ absoluta", "problema": ""})
        fig.update_traces(marker_color=[VERDE if x < 0 else VERMELHO for x in v.var_abs])
        st.plotly_chart(estilo(fig, 520, False), width="stretch")

    top5 = QP.groupby("problema").n.sum().sort_values(ascending=False).head(5).index
    ev = QP[(QP.problema.isin(top5)) & QP.ano.between(*f_anos)].groupby(["ano", "problema"], as_index=False).n.sum()
    ev["Ano"] = ev.ano.map(lbl)
    fig = px.line(ev, x="Ano", y="n", color="problema", markers=True, title="Evolução dos 5 problemas mais frequentes",
                  labels={"n": "Reclamações", "problema": "Problema"}, color_discrete_sequence=PALETA)
    st.plotly_chart(estilo(fig, 400), width="stretch")

# ----------------------------------------------------------------------------
# 3. PRESTADORAS
# ----------------------------------------------------------------------------
with tabs[2]:
    ocultar = st.checkbox("Ocultar marcas genéricas (Não localizei a minha empresa, Demais marcas, Outros)", value=True)
    PB = PF[~PF.marca.isin(MARCAS_GENERICAS)] if ocultar else PF
    n_top = st.slider("Quantidade de marcas no ranking", 5, 25, 12)
    tb = comparar(PB, "marca", ano_a, ano_b)
    tb["Participação (%)"] = tb[ano_b] / tb[ano_b].sum() * 100 if tb[ano_b].sum() else np.nan
    tbt = tb.sort_values(ano_b, ascending=False).head(n_top)

    c1, c2 = st.columns(2)
    with c1:
        long = tbt.melt(id_vars="marca", value_vars=[ano_a, ano_b], var_name="Ano", value_name="Solicitações")
        long["Ano"] = long.Ano.map(lbl)
        fig = px.bar(long, x="marca", y="Solicitações", color="Ano", barmode="group", title="Ranking de marcas",
                     color_discrete_sequence=["#9DB4CE", AZUL])
        st.plotly_chart(estilo(fig, 430), width="stretch")
    with c2:
        v = tbt.sort_values("var_abs")
        fig = px.bar(v, x="var_abs", y="marca", orientation="h", title=f"Variação absoluta ({lbl(ano_a)} → {lbl(ano_b)})",
                     text=v.var_abs.map(fabs), labels={"var_abs": "Δ absoluta", "marca": ""})
        fig.update_traces(marker_color=[VERDE if x < 0 else VERMELHO for x in v.var_abs])
        st.plotly_chart(estilo(fig, 430, False), width="stretch")

    padrao = tbt.marca.head(5).tolist()
    sel_m = st.multiselect("Marcas na evolução anual", sorted(PB.marca.unique()), default=padrao)
    ev = PB[PB.marca.isin(sel_m) & PB.ano.between(*f_anos)].groupby(["ano", "marca"], as_index=False).n.sum()
    ev["Ano"] = ev.ano.map(lbl)
    fig = px.line(ev, x="Ano", y="n", color="marca", markers=True, title="Evolução anual por marca",
                  labels={"n": "Solicitações", "marca": "Marca"}, color_discrete_sequence=PALETA)
    st.plotly_chart(estilo(fig, 400), width="stretch")

    # marca x assunto
    if len(sel_m):
        hm = PB[PB.marca.isin(sel_m) & (PB.ano == ano_b)].pivot_table(index="marca", columns="assunto", values="n", aggfunc="sum", fill_value=0)
        hm = hm.div(hm.sum(axis=1), axis=0) * 100
        hm = hm[hm.sum().sort_values(ascending=False).head(8).index]
        fig = px.imshow(hm, text_auto=".0f", aspect="auto", color_continuous_scale="Oranges",
                        title=f"Perfil de assuntos por marca em {lbl(ano_b)} (% das solicitações da marca)")
        st.plotly_chart(estilo(fig, 360, False), width="stretch")

    tab = tbt.rename(columns={"marca": "Marca", "var_abs": "Δ abs.", "var_pct": "Δ %"}).copy()
    tab.columns = [lbl(c) if isinstance(c, (int, np.integer)) else c for c in tab.columns]
    for c in (lbl(ano_a), lbl(ano_b)):
        tab[c] = tab[c].map(fmt)
    tab["Δ abs."], tab["Δ %"] = tab["Δ abs."].map(fabs), tab["Δ %"].map(fpct)
    tab["Participação (%)"] = tab["Participação (%)"].map(lambda x: f"{x:.1f}%".replace(".", ","))
    st.dataframe(tab, hide_index=True, width="stretch")
    st.caption("O dataset não traz a base de acessos, portanto não é possível calcular o IR por marca aqui. "
               "Veja os IRs publicados na aba 'Referência do Panorama 2025'.")

# ----------------------------------------------------------------------------
# 4. GEOGRAFIA
# ----------------------------------------------------------------------------
with tabs[3]:
    g = PFp.groupby("uf", as_index=False).n.sum().sort_values("n")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(g, x="n", y="uf", orientation="h", title=f"Solicitações por UF ({f_anos[0]}–{f_anos[1]})",
                     labels={"n": "Solicitações", "uf": ""})
        fig.update_traces(marker_color=AZUL)
        st.plotly_chart(estilo(fig, 640, False), width="stretch")
    with c2:
        tu = comparar(PF, "uf", ano_a, ano_b)
        tu = tu[tu[ano_a] >= 1000].sort_values("var_pct")
        fig = px.bar(tu, x="var_pct", y="uf", orientation="h", title=f"Variação % por UF ({lbl(ano_a)} → {lbl(ano_b)})",
                     labels={"var_pct": "Variação %", "uf": ""})
        fig.update_traces(marker_color=[VERDE if x < 0 else VERMELHO for x in tu.var_pct])
        st.plotly_chart(estilo(fig, 640, False), width="stretch")

    top_uf = g.sort_values("n", ascending=False).uf.head(5).tolist()
    evu = PFp[PFp.uf.isin(top_uf)].groupby(["ano", "uf"], as_index=False).n.sum()
    evu["Ano"] = evu.ano.map(lbl)
    fig = px.line(evu, x="Ano", y="n", color="uf", markers=True, title="Evolução anual nas 5 maiores UFs",
                  labels={"n": "Solicitações", "uf": "UF"}, color_discrete_sequence=PALETA)
    st.plotly_chart(estilo(fig, 360), width="stretch")

    st.subheader("Municípios (somente reclamações)")
    tc = comparar(CF.assign(cidade_uf=CF.cidade.fillna("N/D") + " / " + CF.uf.fillna("")), "cidade_uf", ano_a, ano_b)
    tcs = tc.sort_values(ano_b, ascending=False).head(25)
    cc1, cc2 = st.columns([3, 2])
    with cc1:
        top15 = tcs.head(15).sort_values(ano_b)
        fig = px.bar(top15, x=ano_b, y="cidade_uf", orientation="h", title=f"Top 15 municípios em {lbl(ano_b)}",
                     labels={ano_b: "Reclamações", "cidade_uf": ""})
        fig.update_traces(marker_color=AZUL)
        st.plotly_chart(estilo(fig, 480, False), width="stretch")
    with cc2:
        s = tcs.rename(columns={"cidade_uf": "Município", "var_abs": "Δ abs.", "var_pct": "Δ %"}).copy()
        s.columns = [lbl(c) if isinstance(c, (int, np.integer)) else c for c in s.columns]
        for c in (lbl(ano_a), lbl(ano_b)):
            s[c] = s[c].map(fmt)
        s["Δ abs."], s["Δ %"] = s["Δ abs."].map(fabs), s["Δ %"].map(fpct)
        st.dataframe(s, hide_index=True, width="stretch", height=480)
    st.caption("Volumes absolutos, sem normalização por população ou número de acessos; grandes municípios tendem a liderar.")

# ----------------------------------------------------------------------------
# 5. CANAIS E PERFIL
# ----------------------------------------------------------------------------
with tabs[4]:
    PC = filtra(P, tipo=f_tipo, servico=f_serv, marca=f_marca, uf=f_uf, condicao=f_cond)
    PC = PC[PC.ano.between(*f_anos)]
    ca = PC.groupby(["ano", "canal"], as_index=False).n.sum()
    ca["%"] = ca.n / ca.groupby("ano").n.transform("sum") * 100
    ca["Ano"] = ca.ano.map(lbl)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(ca, x="Ano", y="%", color="canal", title="Participação dos canais de entrada (%)",
                     labels={"canal": "Canal"}, color_discrete_map={"Site": "#7CC04B", "Aplicativo": "#4A90D9", "Call center": "#E02020",
                                                                    "WhatsApp": "#25D366", "SEI": "#999999"})
        st.plotly_chart(estilo(fig, 420), width="stretch")
    with c2:
        w = PC.groupby(["ano", "mes", "canal"], as_index=False).n.sum()
        w["data"] = pd.to_datetime(dict(year=w.ano, month=w.mes, day=1))
        w = w[w.canal.isin(["WhatsApp", "Aplicativo", "Site", "Call center"]) & (w.ano >= max(f_anos[0], 2023))]
        fig = px.line(w, x="data", y="n", color="canal", title="Canais por mês (a partir de 2023)",
                      labels={"n": "Solicitações", "data": "", "canal": "Canal"},
                      color_discrete_map={"Site": "#7CC04B", "Aplicativo": "#4A90D9", "Call center": "#E02020", "WhatsApp": "#25D366"})
        st.plotly_chart(estilo(fig, 420), width="stretch")

    PD = filtra(P, servico=f_serv, marca=f_marca, uf=f_uf, canal=f_canal, condicao=f_cond)
    PD = PD[PD.ano.between(*f_anos)]
    c3, c4 = st.columns(2)
    with c3:
        t = PD.groupby(["ano", "tipo"], as_index=False).n.sum()
        t["Ano"] = t.ano.map(lbl)
        fig = px.bar(t[t.tipo != "Reclamação"], x="Ano", y="n", color="tipo", title="Demais tipos de atendimento (exceto Reclamação)",
                     labels={"n": "Solicitações", "tipo": "Tipo"}, color_discrete_sequence=PALETA)
        st.plotly_chart(estilo(fig, 380), width="stretch")
    with c4:
        cd = PC.groupby(["ano", "condicao"], as_index=False).n.sum()
        cd["%"] = cd.n / cd.groupby("ano").n.transform("sum") * 100
        cd["Ano"] = cd.ano.map(lbl)
        fig = px.bar(cd, x="Ano", y="%", color="condicao", title="Registros novos vs reabertos (%)",
                     labels={"condicao": "Condição"}, color_discrete_map={"Nova": AZUL, "Reaberta": AMARELO})
        st.plotly_chart(estilo(fig, 380), width="stretch")

    sc = PC.groupby(["servico", "canal"], as_index=False).n.sum()
    sc["%"] = sc.n / sc.groupby("servico").n.transform("sum") * 100
    pv = sc.pivot(index="servico", columns="canal", values="%").fillna(0)
    fig = px.imshow(pv, text_auto=".1f", aspect="auto", color_continuous_scale="Blues",
                    title="Canal de entrada por serviço no período (% do serviço)")
    st.plotly_chart(estilo(fig, 340, False), width="stretch")

# ----------------------------------------------------------------------------
# 6. REFERÊNCIA DO PANORAMA 2025 (PDF)
# ----------------------------------------------------------------------------
with tabs[5]:
    st.markdown("Conteúdo extraído do relatório **Panorama de Reclamações 2025** (Anatel, fev/2026). "
                "O IR depende da base de acessos das prestadoras, que não consta no dataset.")
    pdf_2025 = {"Banda Larga Fixa": 470875, "Celular Pós-Pago": 489968, "Celular Pré-Pago": 194063,
                "Telefone Fixo": 101698, "TV por Assinatura": 83350, "Outros": 14837}
    pdf_2024 = {"Banda Larga Fixa": 442281, "Celular Pós-Pago": 426699, "Celular Pré-Pago": 193655,
                "Telefone Fixo": 120775, "TV por Assinatura": 73993, "Outros": 9786}
    base = P[P.tipo == "Reclamação"].groupby(["ano", "servico"]).n.sum()
    rec = pd.DataFrame({"Serviço": list(pdf_2025)})
    rec["PDF 2024"] = rec["Serviço"].map(pdf_2024)
    rec["Dataset 2024"] = rec["Serviço"].map(lambda s: base.get((2024, s), np.nan))
    rec["PDF 2025"] = rec["Serviço"].map(pdf_2025)
    rec["Dataset 2025"] = rec["Serviço"].map(lambda s: base.get((2025, s), np.nan))
    tot = rec.drop(columns="Serviço").sum()
    rec.loc[len(rec)] = ["Total", *tot.values]
    rec["Dif. 2025 (dataset − PDF)"] = rec["Dataset 2025"] - rec["PDF 2025"]
    st.subheader("Conferência: dataset × PDF (somente reclamações)")
    st.dataframe(rec.assign(**{c: rec[c].map(fmt) for c in rec.columns[1:5]},
                            **{"Dif. 2025 (dataset − PDF)": rec["Dif. 2025 (dataset − PDF)"].map(fabs)}),
                 hide_index=True, width="stretch")
    st.caption("Pequenas diferenças decorrem da data de extração e de reclassificações posteriores.")

    ir_servico = pd.DataFrame({"Serviço": ["Banda Larga Fixa", "Celular Pós-Pago", "Celular Pré-Pago", "Telefone Fixo", "TV por Assinatura"],
                               "2021": [1.18, 0.70, 0.22, 1.09, 0.86], "2025": [0.73, 0.34, 0.17, 0.39, 0.84]})
    ir_marca = {"Banda Larga Fixa": {"Tim": 5.49, "Oi": 3.42, "Sky": 1.38, "Nio": 1.28, "Vivo": 1.14, "Claro": 1.03, "Algar": 0.68,
                                     "Vero": 0.71, "Giga+": 0.64, "Desktop": 0.51, "Brisanet": 0.37, "Brasil Tecpar": 0.34, "Unifique": 0.14},
                "Celular Pós-Pago": {"Tim": 0.56, "Claro": 0.38, "Vivo": 0.22, "Algar": 0.18, "Surf Telecom": 0.06},
                "Celular Pré-Pago": {"Claro": 0.18, "Tim": 0.17, "Vivo": 0.15, "Algar": 0.09, "Surf Telecom": 0.07},
                "Telefone Fixo": {"Vivo": 0.75, "Oi": 0.67, "Claro": 0.25, "Nio": 0.24, "Algar": 0.12},
                "TV por Assinatura": {"Claro": 1.08, "Vivo": 0.58, "Sky": 0.45}}
    k1, k2, k3 = st.columns(3)
    k1.metric("IR do setor 2025 (reclamações por mil acessos)", "0,38", "+0,03 vs 2024", delta_color="inverse")
    k2.metric("IR 2024", "0,35")
    k3.metric("IR 2023", "0,36")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_bar(x=ir_servico["Serviço"], y=ir_servico["2021"], name="2021", marker_color="#9DB4CE")
        fig.add_bar(x=ir_servico["Serviço"], y=ir_servico["2025"], name="2025", marker_color=AZUL)
        fig.update_layout(barmode="group", title="IR por serviço: 2021 vs 2025")
        st.plotly_chart(estilo(fig, 380), width="stretch")
    with c2:
        sv = st.selectbox("Serviço (IR por marca, 2025)", list(ir_marca))
        s = pd.Series(ir_marca[sv]).sort_values(ascending=False).reset_index()
        s.columns = ["Marca", "IR 2025"]
        fig = px.bar(s, x="Marca", y="IR 2025", text="IR 2025", title=f"{sv}: IR 2025 por marca")
        fig.update_traces(marker_color=AZUL, texttemplate="%{text:.2f}")
        st.plotly_chart(estilo(fig, 380, False), width="stretch")
    st.caption("IR por marca lido dos gráficos do relatório; pode haver pequenas imprecisões onde as linhas se sobrepõem.")

    st.markdown("""
**Principais destaques do relatório**
- 2025 teve 1.354.791 reclamações (+6,91% vs 2024), o primeiro crescimento desde 2019.
- **Celular Pós-Pago** (+14,8%, puxado por Cobrança e pela Tim) e **TV por Assinatura** (+12,6%, Claro e Oi/Mileto) lideram as altas; **Telefone Fixo** foi o único em queda (-15,8%).
- Em termos relativos, a **TV por Assinatura** voltou a ser o serviço mais reclamado (IR 0,84), o que não ocorria desde 2018.
- Assuntos com maior alta: Cobrança (+52,2 mil), Cancelamento (+27,9 mil) e Planos/Ofertas (+10,3 mil); Qualidade caiu 24,9 mil.
- Piores IR por serviço: Tim (Banda Larga e Pós-pago), Claro (Pré-pago e TV), Vivo (Telefone Fixo).
""")

st.divider()
st.caption("Dados: dataset de solicitações do Anatel Consumidor (agregado) e Panorama de Reclamações 2025 (Anatel).")
