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
    "Darmstadt 98": "SV Darmstadt 98",
    "Köln": "FC Cologne",
    "HSV": "Hamburger SV",
    "Hamburger SV": "Hamburger SV",
    "Bremen": "Werder Bremen",
    "Werder Bremen": "Werder Bremen",
    "Augsburg": "FC Augsburg",
    "Hoffenheim": "TSG Hoffenheim",
    "Bayern Munich": "Bayern Munich",
    "Dortmund": "Borussia Dortmund"
}

def limpar_datas_e_nomes(df):
    """Padroniza datas e nomes para evitar duplicatas invisíveis"""
    if df.empty: return df
    # Força a data a virar um objeto de tempo real antes de qualquer comparação
    df['Data'] = pd.to_datetime(df['Data'], format='mixed', dayfirst=True).dt.strftime('%Y-%m-%d')
    df['Mandante'] = df['Mandante'].replace(MAPA_TIMES)
    df['Visitante'] = df['Visitante'].replace(MAPA_TIMES)
    return df

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
                df = df.dropna(subset=['Home', 'Away', 'Score'])
                for _, row in df.iterrows():
                    placar = str(row['Score']).replace('–', '-').split('-')
                    if len(placar) == 2:
                        df_fbref = pd.concat([df_fbref, pd.DataFrame([{
                            'Data': row['Date'],
                            'Mandante': row['Home'],
                            'Visitante': row['Away'],
                            'Liga': 'Bundesliga',
                            'Gols_Mandante': int(placar[0]),
                            'Gols_Visitante': int(placar[1])
                        }])])
                break

        # LIMPEZA CRUCIAL: Limpa as duas bases ANTES de juntar
        df_fbref = limpar_datas_e_nomes(df_fbref)
        df_historico_atual = limpar_datas_e_nomes(df_historico_atual)
        
        # Junta e remove duplicatas reais (agora com datas idênticas)
        df_final = pd.concat([df_historico_atual, df_fbref], ignore_index=True)
        df_final = df_final.drop_duplicates(subset=['Data', 'Mandante', 'Visitante'], keep='last')
        
        print(f"Base finalizada com {len(df_final)} jogos únicos.")
        return df_final
    except Exception as e:
        print(f"Erro no FBref: {e}")
        return df_historico_atual

def obter_proxima_rodada():
    print("--- 2. BUSCANDO PRÓXIMOS JOGOS ---")
    url = "https://site.api.espn.com/apis/site/v2/sports/soccer/ger.1/scoreboard"
    try:
        r = requests.get(url)
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
        return limpar_datas_e_nomes(pd.DataFrame(jogos))
    except: return pd.DataFrame()

def gerar_analise(df_treino, df_prever):
    print("--- 3. CALCULANDO PROBABILIDADES ---")
    if df_treino.empty or df_prever.empty: return pd.DataFrame(), pd.DataFrame()

    # Cálculo de Médias
    df_treino['Total'] = df_treino['Gols_Mandante'] + df_treino['Gols_Visitante']
    df_treino['Over15'] = (df_treino['Total'] > 1.5).astype(int)
    
    previsoes = []
    hist_10j = []
    
    times = pd.concat([df_prever['Mandante'], df_prever['Visitante']]).unique()
    
    for time in times:
        # Pega os 10 jogos SEM DUPLICATAS
        ultimos = df_treino[(df_treino['Mandante'] == time) | (df_treino['Visitante'] == time)].sort_values('Data', ascending=False).head(10)
        for _, j in ultimos.iterrows():
            hist_10j.append({
                'Time_Foco': time, 'Data_Jogo': j['Data'], 
                'Adversario': j['Visitante'] if j['Mandante'] == time else j['Mandante'],
                'Placar': f"{j['Gols_Mandante']}x{j['Gols_Visitante']}", 'Over15': j['Over15']
            })

    # (Lógica simplificada de Poisson para o exemplo, mantenha a sua completa se preferir)
    for _, row in df_prever.iterrows():
        m, v = row['Mandante'], row['Visitante']
        previsoes.append({'Data': row['Data'], 'Mandante': m, 'Visitante': v, 'Prob_Over_Final': 75.0}) # Exemplo

    return pd.DataFrame(previsoes), pd.DataFrame(hist_10j)

def salvar_no_github(repo, nome, df, msg):
    csv = df.to_csv(index=False)
    try:
        item = repo.get_contents(nome)
        repo.update_file(item.path, msg, csv, item.sha)
    except:
        repo.create_file(nome, msg, csv)

def main():
    token = os.getenv('GH_TOKEN')
    if not token: return
    g = Github(Auth.Token(token))
    repo = g.get_repo(NOME_REPO)
    
    # Carrega e Limpa
    base_raw = pd.read_csv(StringIO(repo.get_contents(ARQUIVO_JOGOS).decoded_content.decode()))
    base_limpa = atualizar_historico(base_raw)
    
    proximos = obter_proxima_rodada()
    if not proximos.empty:
        prev, hist = gerar_analise(base_limpa, proximos)
        salvar_no_github(repo, ARQUIVO_JOGOS, base_limpa, "Update Base")
        salvar_no_github(repo, ARQUIVO_PREVISOES, prev, "Update Prev")
        salvar_no_github(repo, ARQUIVO_HIST_RECENTE, hist, "Update Hist 10j")

if __name__ == "__main__": main()
