import pandas as pd
import numpy as np
import os
from github import Github, Auth
from io import StringIO
import requests
from scipy.stats import poisson
import difflib
from datetime import datetime, timedelta

# --- CONFIGURAÇÕES ---
GITHUB_TOKEN = os.getenv('GH_TOKEN')
NOME_REPO = "marcioklipper/futebol-preditivo-v2"
ARQUIVO_JOGOS = "historico_jogos.csv"
ARQUIVO_PREVISOES = "analise_preditiva.csv"
ARQUIVO_HIST_RECENTE = "historico_10j_times.csv"

# --- DICIONÁRIO DE TRADUÇÃO BLINDADO ---
MAPA_TIMES = {
    "Leverkusen": "Bayer Leverkusen",
    "Bayer 04 Leverkusen": "Bayer Leverkusen",
    "M'Gladbach": "Borussia Monchengladbach",
    "Mönchengladbach": "Borussia Monchengladbach",
    "Frankfurt": "Eintracht Frankfurt",
    "Ein Frankfurt": "Eintracht Frankfurt",
    "Eintracht Frankfurt": "Eintracht Frankfurt",
    "Stuttgart": "VfB Stuttgart",
    "Wolfsburg": "VfL Wolfsburg",
    "Bochum": "VfL Bochum",
    "Heidenheim": "FC Heidenheim",
    "1. FC Heidenheim 1846": "FC Heidenheim",
    "Darmstadt 98": "SV Darmstadt 98",
    "Köln": "FC Cologne",
    "1. FC Köln": "FC Cologne",
    "HSV": "Hamburger SV",
    "Hamburger SV": "Hamburger SV",
    "Bremen": "Werder Bremen",
    "Werder Bremen": "Werder Bremen",
    "Augsburg": "FC Augsburg",
    "Hoffenheim": "TSG Hoffenheim",
    "Bayern Munich": "Bayern Munich",
    "Dortmund": "Borussia Dortmund",
    "RB Leipzig": "RB Leipzig",
    "Freiburg": "SC Freiburg",
    "Union Berlin": "Union Berlin",
    "Mainz 05": "FSV Mainz 05",
    "1. FSV Mainz 05": "FSV Mainz 05"
}

def limpar_datas_e_nomes(df):
    """Padroniza nomes ANTES de qualquer cálculo para evitar duplicatas invisíveis"""
    if df.empty: return df
    df['Mandante'] = df['Mandante'].replace(MAPA_TIMES)
    df['Visitante'] = df['Visitante'].replace(MAPA_TIMES)
    return df

def faxina_temporal(df):
    """A GUILHOTINA: Converte tudo para data real, corta o futuro e devolve como texto limpo"""
    if df.empty: return df
    
    # Converte para objeto de tempo (entende tanto 17/01/2026 quanto 2026-01-17)
    df['Data_Temp'] = pd.to_datetime(df['Data'], format='mixed', dayfirst=True)
    
    # Corta fora qualquer linha onde a Data seja MAIOR que hoje
    hoje = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
    df = df[df['Data_Temp'] <= hoje].copy()
    
    # Converte de volta para texto padrão (YYYY-MM-DD) e apaga a coluna temporária
    df['Data'] = df['Data_Temp'].dt.strftime('%Y-%m-%d')
    df = df.drop(columns=['Data_Temp'])
    return df

# --- 1. ATUALIZADOR INFALÍVEL (FBREF) ---
def atualizar_historico(df_historico_atual):
    print("--- 1. BUSCANDO RESULTADOS NO FBREF ---")
    url = "https://fbref.com/en/comps/20/schedule/Bundesliga-Scores-and-Fixtures"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        r = requests.get(url, headers=headers)
        dfs = pd.read_html(StringIO(r.text), flavor='lxml')
        df_fbref = pd.DataFrame()
        
        for df in dfs:
            if 'Home' in df.columns and 'Score' in df.columns:
                df = df.dropna(subset=['Home', 'Away', 'Score', 'Date'])
                
                for _, row in df.iterrows():
                    try:
                        data_jogo_bruta = str(row['Date'])
                        placar_str = str(row['Score']).strip()
                        
                        # Trava primária: ignora se não tiver um traço no placar
                        if not placar_str or placar_str == 'nan' or '-' not in placar_str.replace('–', '-'):
                            continue
                            
                        placar = placar_str.replace('–', '-').split('-')
                        
                        if len(placar) == 2:
                            gm_str = ''.join(filter(str.isdigit, placar[0]))
                            gv_str = ''.join(filter(str.isdigit, placar[1]))
                            
                            if gm_str and gv_str: 
                                df_fbref = pd.concat([df_fbref, pd.DataFrame([{
                                    'Data': data_jogo_bruta,
                                    'Mandante': str(row['Home']),
                                    'Visitante': str(row['Away']),
                                    'Liga': 'Bundesliga',
                                    'Gols_Mandante': int(gm_str),
                                    'Gols_Visitante': int(gv_str)
                                }])])
                    except Exception as e:
                        pass
                break

        # Limpeza Inicial de Nomes
        if not df_fbref.empty:
            df_fbref = limpar_datas_e_nomes(df_fbref)
            
        df_historico_atual = limpar_datas_e_nomes(df_historico_atual)
        
        # Junta o passado com o presente
        df_final = pd.concat([df_fbref, df_historico_atual], ignore_index=True)
        
        # --- A FAXINA TEMPORAL (O Exterminador do Futuro) ---
        df_final = faxina_temporal(df_final)
        
        # Remove duplicatas (agora que as datas estão no mesmo formato universal)
        df_final = df_final.drop_duplicates(subset=['Data', 'Mandante', 'Visitante'], keep='last')
        
        print(f"Base atualizada com sucesso! Total de {len(df_final)} jogos reais confirmados.")
        return df_final
        
    except Exception as e:
        print(f"Erro ao buscar resultados no FBref: {e}")
        return df_historico_atual

def obter_proxima_rodada():
    print("--- 2. BUSCANDO PRÓXIMOS JOGOS (ESPN) ---")
    hoje = datetime.now()
    futuro = hoje + timedelta(days=30)
    data_inicio = hoje.strftime('%Y%m%d')
    data_fim = futuro.strftime('%Y%m%d')
    
    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/ger.1/scoreboard?dates={data_inicio}-{data_fim}"
    
    try:
        r = requests.get(url, timeout=15)
        dados = r.json()
        jogos = []
        for evento in dados.get('events', []):
            comp = evento['competitions'][0]
            if comp['status']['type']['state'] == 'pre':
                jogos.append({
                    'Data': evento['date'][:10],
                    'Mandante': comp['competitors'][0]['team']['displayName'],
                    'Visitante': comp['competitors'][1]['team']['displayName'],
                    'Liga': 'Bundesliga'
                })
        
        df_futuros = pd.DataFrame(jogos)
        if not df_futuros.empty:
            df_futuros = limpar_datas_e_nomes(df_futuros)
            df_futuros = faxina_temporal(df_futuros) # Formata a data bonitinho
            df_futuros = df_futuros.head(9)
            print(f"Encontrados {len(df_futuros)} próximos jogos!")
            return df_futuros
        return pd.DataFrame()
    except Exception as e:
        print(f"Erro ao conectar na ESPN: {e}")
        return pd.DataFrame()

def gerar_analise(df_treino, df_prever):
    print("--- 3. CÉREBRO PREDITIVO E EXTRATOR DE HISTÓRICO ---")
    if df_treino.empty or df_prever.empty: return pd.DataFrame(), pd.DataFrame()

    df_treino = df_treino.dropna(subset=['Gols_Mandante', 'Gols_Visitante']).copy()
    print(f"Treinando o modelo com {len(df_treino)} jogos...")

    df_treino['Total_Gols'] = df_treino['Gols_Mandante'] + df_treino['Gols_Visitante']
    df_treino['Over15'] = np.where(df_treino['Total_Gols'] > 1.5, 1, 0)
    m_liga_mand = df_treino['Gols_Mandante'].mean()
    m_liga_visit = df_treino['Gols_Visitante'].mean()

    stats_casa = df_treino.groupby('Mandante').agg(Media_Feitos_Casa=('Gols_Mandante', 'mean'), Media_Sofridos_Casa=('Gols_Visitante', 'mean'), Taxa_Over_Casa=('Over15', 'mean')).reset_index().rename(columns={'Mandante': 'Time'})
    stats_fora = df_treino.groupby('Visitante').agg(Media_Feitos_Fora=('Gols_Visitante', 'mean'), Media_Sofridos_Fora=('Gols_Mandante', 'mean'), Taxa_Over_Fora=('Over15', 'mean')).reset_index().rename(columns={'Visitante': 'Time'})
    stats = pd.merge(stats_casa, stats_fora, on='Time', how='outer').fillna(0)

    def calcular_fator_forma(time_oficial):
        ultimos_10 = df_treino[(df_treino['Mandante'] == time_oficial) | (df_treino['Visitante'] == time_oficial)].sort_values('Data', ascending=False).head(10)
        if ultimos_10.empty: return 1.0
        taxa_recente = ultimos_10['Over15'].mean()
        taxa_historica = (stats[stats['Time'] == time_oficial]['Taxa_Over_Casa'].values[0] + stats[stats['Time'] == time_oficial]['Taxa_Over_Fora'].values[0]) / 2
        return (taxa_recente / taxa_historica) if taxa_historica > 0 else 1.0

    stats['Fator_10_Jogos'] = stats['Time'].apply(calcular_fator_forma)
    stats['Ataque_Casa'] = stats['Media_Feitos_Casa'] / m_liga_mand
    stats['Defesa_Casa'] = stats['Media_Sofridos_Casa'] / m_liga_visit
    stats['Ataque_Fora'] = stats['Media_Feitos_Fora'] / m_liga_visit
    stats['Defesa_Fora'] = stats['Media_Sofridos_Fora'] / m_liga_mand

    previsoes = []
    tabela_historico = []
    times_processados = set()

    print("--- 4. CALCULANDO PROBABILIDADES ---")
    for idx, row in df_prever.iterrows():
        try:
            home = str(row['Mandante']).strip()
            away = str(row['Visitante']).strip()

            d_home = stats[stats['Time'] == home]
            d_away = stats[stats['Time'] == away]
            if d_home.empty or d_away.empty: continue

            for time_foco in [home, away]:
                if time_foco not in times_processados:
                    jogos_time = df_treino[(df_treino['Mandante'] == time_foco) | (df_treino['Visitante'] == time_foco)].sort_values('Data', ascending=False).head(10)
                    for _, jogo in jogos_time.iterrows():
                        gm = int(jogo['Gols_Mandante'])
                        gv = int(jogo['Gols_Visitante'])
                        total_gols = gm + gv
                        mando = 'Casa' if jogo['Mandante'] == time_foco else 'Fora'
                        tabela_historico.append({
                            'Time_Foco': time_foco,
                            'Data_Jogo': jogo['Data'],
                            'Mando': mando,
                            'Adversario': jogo['Visitante'] if mando == 'Casa' else jogo['Mandante'],
                            'Placar': f"{gm} x {gv}",
                            'Total_Gols': total_gols,
                            'Bateu_Over_1_5': 'Sim' if total_gols > 1 else 'Não'
                        })
                    times_processados.add(time_foco)

            lamb_home = d_home['Ataque_Casa'].values[0] * d_away['Defesa_Fora'].values[0] * m_liga_mand
            lamb_away = d_away['Ataque_Fora'].values[0] * d_home['Defesa_Casa'].values[0] * m_liga_visit
            prob_under = (poisson.pmf(0, lamb_home)*poisson.pmf(0, lamb_away)) + (poisson.pmf(1, lamb_home)*poisson.pmf(0, lamb_away)) + (poisson.pmf(0, lamb_home)*poisson.pmf(1, lamb_away))
            prob_over_pura = (1 - prob_under) * 100

            fator_h = d_home['Fator_10_Jogos'].values[0]
            fator_a = d_away['Fator_10_Jogos'].values[0]
            fator_combinado = (fator_h + fator_a) / 2
            prob_ajustada = prob_over_pura * fator_combinado
            prob_final = min(max(prob_ajustada, 5.0), 95.0) 

            previsoes.append({
                'Data': row['Data'], 'Liga': row['Liga'], 'Mandante': home, 'Visitante': away,
                'xG_Mandante': round(lamb_home, 2), 'xG_Visitante': round(lamb_away, 2),
                'Media_Gols_Esperada': round(lamb_home + lamb_away, 2),
                'Prob_Poisson_Pura': round(prob_over_pura, 1), 'Fator_Forma_10j': round(fator_combinado, 2),
                'Prob_Over_Final': round(prob_final, 1)
            })
            print(f"✅ Gerado Previsão + Histórico Oficial: {home} vs {away}")
        except Exception as e: continue
        
    return pd.DataFrame(previsoes), pd.DataFrame(tabela_historico)

def salvar_no_github(repo, nome_arquivo, df_dados, mensagem):
    csv_string = df_dados.to_csv(index=False)
    try:
        contents = repo.get_contents(nome_arquivo)
        repo.update_file(contents.path, f"Update {mensagem}", csv_string, contents.sha)
        print(f"SUCESSO: {nome_arquivo} Atualizado!")
    except Exception as e:
        try:
            repo.create_file(nome_arquivo, f"Create {mensagem}", csv_string)
            print(f"SUCESSO: {nome_arquivo} Criado pela primeira vez!")
        except Exception as e_create:
            print(f"Erro Crítico ao criar {nome_arquivo}: {e_create}")

def main():
    if not GITHUB_TOKEN:
        print("Erro Crítico: GITHUB_TOKEN não configurado no Secrets do repositório.")
        return
        
    auth = Auth.Token(GITHUB_TOKEN)
    g = Github(auth=auth)
    repo = g.get_repo(NOME_REPO)

    print("--- LENDO HISTÓRICO BASE ---")
    try:
        conteudo = repo.get_contents(ARQUIVO_JOGOS)
        df_historico_base = pd.read_csv(StringIO(conteudo.decoded_content.decode('utf-8')))
        print("Base principal carregada com sucesso!")
    except Exception as e: 
        print(f"Falha ao ler o histórico base. Abortando. Erro: {e}")
        return

    df_historico_atualizado = atualizar_historico(df_historico_base)
    
    salvar_no_github(repo, ARQUIVO_JOGOS, df_historico_atualizado, "Historico Exterminador do Futuro")

    df_novos = obter_proxima_rodada()
    
    if not df_novos.empty:
        df_analise, df_hist_recente = gerar_analise(df_historico_atualizado, df_novos)
        
        if not df_analise.empty:
            salvar_no_github(repo, ARQUIVO_PREVISOES, df_analise, "Previsoes")
            salvar_no_github(repo, ARQUIVO_HIST_RECENTE, df_hist_recente, "Historico Detalhado 10 Jogos")
        else: 
            print("Aviso: Falha matemática ao gerar as tabelas de previsão.")
    else: 
        print("Aviso: Sem jogos mapeados no futuro para prever hoje.")

if __name__ == "__main__":
    main()
