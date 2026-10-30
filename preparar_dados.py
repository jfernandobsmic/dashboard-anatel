"""
Gera os arquivos agregados em dados/ a partir do CSV bruto (reclamacoes_contexto.csv, separador ';').
Uso: python preparar_dados.py caminho/para/reclamacoes_contexto.csv
Requisitos: pip install duckdb pyarrow
"""
import sys
from pathlib import Path

import duckdb

csv = sys.argv[1] if len(sys.argv) > 1 else "reclamacoes_contexto.csv"
out = Path(__file__).parent / "dados"
out.mkdir(exist_ok=True)

con = duckdb.connect()
con.execute(f"""create view r as select * from read_csv('{csv}', delim=';', header=true,
                encoding='utf-8', all_varchar=true)""")
con.execute("""create table base as select
    cast(Ano as smallint) ano, cast("Mês" as tinyint) mes, UF uf, Cidade cidade, CO_MUNICIPIO cod_mun,
    CanalEntrada canal, "Condição" condicao, TipoAtendimento tipo, Serviço servico,
    upper(trim(Marca)) marca, trim(Assunto) assunto, trim(Problema) problema,
    cast("SOLICITAÇÕES" as int) n from r""")
# 40 marcas com mais reclamações; as demais viram 'DEMAIS MARCAS'
con.execute("""create table topm as select marca from base where tipo='Reclamação'
               group by 1 order by sum(n) desc limit 40""")
con.execute("""create table b2 as select ano, mes, uf, cidade, cod_mun, canal, condicao, tipo, servico,
               case when marca in (select marca from topm) then marca else 'DEMAIS MARCAS' end marca,
               assunto, problema, n from base""")

def salvar(sql, nome):
    con.execute(f"copy ({sql}) to '{out / nome}' (format parquet, compression zstd)")

salvar("select ano,mes,uf,canal,condicao,tipo,servico,marca,assunto,sum(n) n from b2 group by all", "fato_principal.parquet")
salvar("select ano,servico,marca,assunto,problema,sum(n) n from b2 where tipo='Reclamação' group by all", "fato_problema.parquet")
salvar("select ano,uf,cidade,cod_mun,servico,sum(n) n from b2 where tipo='Reclamação' group by all", "fato_cidade.parquet")
print("Arquivos gerados em", out)
