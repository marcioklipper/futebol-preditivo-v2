import pandas as pd
import numpy as np
import os
from github import Github
from io import StringIO
import requests
from scipy.stats import poisson
from datetime import datetime

# --- CONFIGURAÇÕES ---
GITHUB_TOKEN = os.getenv('GH_TOKEN')
NOME_REPO = "marcioklipper/futebol-preditivo-v2" # Certifique-se que o nome do repo é este mesmo
ARQUIVO_JOGOS = "base_europa_unificada (1).csv"
ARQUIVO_PREVISOES = "analise_preditiva.csv"

urls_ligas = {
    'Bundesliga': 'https://www.espn.com.br/futebol/calendario/_/liga/GER.1'
}

def extrair_jogos_espn():
    print("--- 1. VERIFICANDO JOGOS REAIS (ESPN) ---")
    lista_dfs = []
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    for liga, url in urls_ligas.items():
        try:
            r = requests.get(url, headers=headers)
            tabelas = pd.read_html(StringIO(r.text), flavor=['lxml', 'bs4'])
            for df in tabelas:
                if len(df) < 2: continue
                col_partida = next((c for c in df.columns if df[c].astype(str).str.contains(' vs ', na=False).any()), None)
                df_temp = df.copy()
                if col_partida:
                    try:
                        div = df_temp[col_partida].str.split(' vs ', expand=True)
                        if len(div.columns) >= 2:
                            df_temp['Mandante'] = div[0].str.strip()
                            df_temp['Visitante'] = div[1].str.strip()
                        else: continue
                    except: continue
                elif len(df.columns) >= 4:
                    df_temp['Mandante'] = df_temp.iloc[:, 0]
                    df_temp['Visitante'] = df_temp.iloc[:, 1]
                else: continue

                df_temp['Liga'] = liga
                df_temp['Data'] = datetime.today().strftime('%Y-%m-%d')
                
                for c in ['Gols_Mandante', 'Gols_Visitante']:
                    if c not in df_temp.columns: df_temp[c] = np.nan
                
                cols = ['Data', 'Mandante', 'Visitante', 'Liga', 'Gols_Mandante', 'Gols_Visitante']
                for c in cols:
                    if c not in df_temp.columns: df_temp[c] = np.nan
                
                lista_dfs.append(df_temp[cols])
        except: pass
    
    if lista_dfs:
        df = pd.concat(lista_dfs, ignore_index=True)
        for c in ['Mandante', 'Visitante']:
            df[c] = df[c].astype(str).str.replace(' logo', '', regex=False)
        return df
    return pd.DataFrame()

def gerar_analise_focada(df_completo):
    print("--- 2. CÉREBRO PREDITIVO (MAX HISTÓRICO) ---")
    
    df_foco = df_completo[df_completo['Liga'] == 'Bundesliga'].copy()
    df_foco['Gols_Mandante'] = pd.to_numeric(df_foco['Gols_Mandante'], errors='coerce')
    df_foco['Gols_Visitante'] = pd.to_numeric(df_foco['Gols_Visitante'], errors='coerce')

    df_futuro_real = df_foco[df_foco['Gols_Mandante'].isna()].copy()
    df_historico_base = df_foco.dropna(subset=['Gols_Mandante', 'Gols_Visitante']).copy()

    if not df_futuro_real.empty:
        print(f"Encontrados {len(df_futuro_real)} jogos futuros reais.")
        df_para_prever = df_futuro_real
        df_treino = df_historico_base
    else:
        print(">>> MODO SIMULAÇÃO: Prevendo a última rodada do arquivo <<<")
        df_sorted = df_historico_base.sort_values(by=['Data', 'Mandante'])
        
        ultima_data = df_sorted['Data'].max()
        print(f"Data Alvo da Simulação: {ultima_data}")
        
        df_para_prever = df_sorted[df_sorted['Data'] == ultima_data].copy()
        if len(df_para_prever) < 2:
             df_para_prever = df_sorted.tail(9).copy()
             
        df_para_prever = df_para_prever.drop_duplicates(subset=['Data', 'Mandante', 'Visitante'])
        ids_prever = df_para_prever.index
        df_treino = df_historico_base.drop(ids_prever)
        
        df_para_prever['Gols_Mandante'] = np.nan
        df_para_prever['Gols_Visitante'] = np.nan

    print(f"Base de Treino: {len(df_treino)} jogos | Jogos a Prever: {len(df_para_prever)}")
    if df_treino.empty: return pd.DataFrame()

    df_treino['Total_Gols'] = df_treino['Gols_Mandante'] + df_treino['Gols_Visitante']
    df_treino['Over15'] = np.where(df_treino['Total_Gols'] > 1.5, 1, 0)

    m_liga_mand = df_treino['Gols_Mandante'].mean()
    m_liga_visit = df_treino['Gols_Visitante'].mean()

    stats_casa = df_treino.groupby('Mandante').agg(
        Media_Feitos_Casa=('Gols_Mandante', 'mean'),
        Media_Sofridos_Casa=('Gols_Visitante', 'mean'),
        Taxa_Over_Casa=('Over15', 'mean')
    ).reset_index().rename(columns={'Mandante': 'Time'})

    stats_fora = df_treino.groupby('Visitante').agg(
        Media_Feitos_Fora=('Gols_Visitante', 'mean'),
        Media_Sofridos_Fora=('Gols_Mandante', 'mean'),
        Taxa_Over_Fora=('Over15', 'mean')
    ).reset_index().rename(columns={'Visitante': 'Time'})

    stats = pd.merge(stats_casa, stats_fora, on='Time', how='outer').fillna(0)
    stats['Ataque_Casa'] = stats['Media_Feitos_Casa'] / m_liga_mand
    stats['Defesa_Casa'] = stats['Media_Sofridos_Casa'] / m_liga_visit
    stats['Ataque_Fora'] = stats['Media_Feitos_Fora'] / m_liga_visit
    stats['Defesa_Fora'] = stats['Media_Sofridos_Fora'] / m_liga_mand

    def get_momento(time):
        jogos = df_treino[(df_treino['Mandante'] == time) | (df_treino['Visitante'] == time)].sort_values('Data', ascending=False).head(5)
        if len(jogos) == 0: return 0
        return jogos['Over15'].mean()
    
    stats['Momento_Over'] = stats['Time'].apply(get_momento)

    previsoes = []
    for idx, row in df_para_prever.iterrows():
        try:
            home, away = row['Mandante'], row['Visitante']
            d_home = stats[stats['Time'] == home]
            d_away = stats[stats['Time'] == away]
            
            if d_home.empty or d_away.empty: continue

            lamb_home = d_home['Ataque_Casa'].values[0] * d_away['Defesa_Fora'].values[0] * m_liga_mand
            lamb_away = d_away['Ataque_Fora'].values[0] * d_home['Defesa_Casa'].values[0] * m_liga_visit
            
            prob_under = (poisson.pmf(0, lamb_home)*poisson.pmf(0, lamb_away)) + \
                         (poisson.pmf(1, lamb_home)*poisson.pmf(0, lamb_away)) + \
                         (poisson.pmf(0, lamb_home)*poisson.pmf(1, lamb_away))
            prob_over = 1 - prob_under

            hist_over = (d_home['Taxa_Over_Casa'].values[0] + d_away['Taxa_Over_Fora'].values[0]) / 2
            momento = (d_home['Momento_Over'].values[0] + d_away['Momento_Over'].values[0]) / 2

            previsoes.append({
                'Data': row['Data'],
                'Liga': 'Bundesliga',
                'Mandante': home,
                'Visitante': away,
                'xG_Mandante': round(lamb_home, 2),
                'xG_Visitante': round(lamb_away, 2),
                'Prob_Over_1_5': round(prob_over * 100, 1),
                'Media_Gols_Esperada': round(lamb_home + lamb_away, 2),
                'Historico_Over': round(hist_over * 100, 1),
                'Momento_Recente': round(momento * 100, 1)
            })
        except: continue

    return pd.DataFrame(previsoes)

def main():
    g = Github(GITHUB_TOKEN)
    repo = g.get_repo(NOME_REPO)

    print("--- LENDO HISTÓRICO ---")
    try:
        # Lendo de forma autenticada direto do GitHub! Adeus erro 404.
        conteudo_arquivo = repo.get_contents(ARQUIVO_JOGOS)
        df_historico = pd.read_csv(StringIO(conteudo_arquivo.decoded_content.decode('utf-8')))
        print(f"Base carregada com sucesso: {len(df_historico)} linhas.")
    except Exception as e:
        print(f"Erro ao ler CSV do GitHub: {e}")
        return

    df_novos = extrair_jogos_espn()
    df_final = df_historico.copy()
    if not df_novos.empty:
        df_final = pd.concat([df_historico, df_novos], ignore_index=True)
        df_final = df_final.drop_duplicates(subset=['Data', 'Mandante', 'Visitante'], keep='last')
        try:
            repo.update_file(ARQUIVO_JOGOS, "Update Jogos", df_final.to_csv(index=False), repo.get_contents(ARQUIVO_JOGOS).sha)
        except: pass

    df_analise = gerar_analise_focada(df_final)
    if not df_analise.empty:
        csv_analise = df_analise.to_csv(index=False)
        try:
            try:
                contents = repo.get_contents(ARQUIVO_PREVISOES)
                repo.update_file(contents.path, "Update Analise", csv_analise, contents.sha)
                print("SUCESSO: analise_preditiva.csv Atualizado!")
            except:
                repo.create_file(ARQUIVO_PREVISOES, "Create Analise", csv_analise)
                print("SUCESSO: analise_preditiva.csv Criado!")
        except Exception as e:
            print(f"Erro ao salvar no GitHub: {e}")
    else:
        print("Aviso: Nenhuma previsão gerada.")

if __name__ == "__main__":
    main()
