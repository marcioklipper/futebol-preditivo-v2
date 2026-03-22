import pandas as pd
import numpy as np
import os
from github import Github, Auth
from io import StringIO
import requests
from scipy.stats import poisson
import difflib

# --- CONFIGURAÇÕES ---
GITHUB_TOKEN = os.getenv('GH_TOKEN')
NOME_REPO = "marcioklipper/futebol-preditivo-v2"
ARQUIVO_JOGOS = "historico_jogos.csv"
ARQUIVO_PREVISOES = "analise_preditiva.csv"

def obter_proxima_rodada():
    print("--- 1. BUSCANDO PRÓXIMOS JOGOS (FBREF) ---")
    # Tabela oficial e limpa do FBref
    url = "https://fbref.com/en/comps/20/schedule/Bundesliga-Scores-and-Fixtures"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        r = requests.get(url, headers=headers)
        dfs = pd.read_html(StringIO(r.text))
        for df in dfs:
            if 'Home' in df.columns and 'Away' in df.columns and 'Score' in df.columns:
                df = df.dropna(subset=['Home', 'Away'])
                
                # O Pulo do Gato: Jogos que ainda não aconteceram NÃO têm Score
                df_futuros = df[df['Score'].isna() | (df['Score'] == '')].copy()
                
                if not df_futuros.empty:
                    df_futuros = df_futuros.rename(columns={'Home': 'Mandante', 'Away': 'Visitante', 'Date': 'Data'})
                    df_futuros['Liga'] = 'Bundesliga'
                    df_futuros = df_futuros.dropna(subset=['Data'])
                    
                    # Pega exatamente os próximos 9 jogos (1 rodada completa da Bundesliga)
                    proximos = df_futuros.head(9)[['Data', 'Mandante', 'Visitante', 'Liga']]
                    print(f"Encontrados {len(proximos)} jogos reais da próxima rodada!")
                    return proximos
    except Exception as e:
        print(f"Erro ao buscar na web: {e}")
    
    print("Aviso: Não foi possível baixar a próxima rodada.")
    return pd.DataFrame()

def gerar_analise(df_treino, df_prever):
    print("--- 2. CÉREBRO PREDITIVO (POISSON) ---")
    
    # Faxina: Remove os "Fantasmas" (jogos antigos sem gols que corrompiam a base)
    df_treino['Gols_Mandante'] = pd.to_numeric(df_treino['Gols_Mandante'], errors='coerce')
    df_treino['Gols_Visitante'] = pd.to_numeric(df_treino['Gols_Visitante'], errors='coerce')
    df_treino = df_treino.dropna(subset=['Gols_Mandante', 'Gols_Visitante'])

    if df_treino.empty or df_prever.empty:
        return pd.DataFrame()

    print(f"Treinando o modelo com {len(df_treino)} jogos passados válidos...")

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

    times_conhecidos = stats['Time'].unique().tolist()
    previsoes = []

    print("--- 3. CALCULANDO PROBABILIDADES ---")
    for idx, row in df_prever.iterrows():
        try:
            home_original = str(row['Mandante']).strip()
            away_original = str(row['Visitante']).strip()
            
            # Tradutor de nomes (O mesmo que salvou nossa vida no passo anterior)
            match_home = difflib.get_close_matches(home_original, times_conhecidos, n=1, cutoff=0.45)
            match_away = difflib.get_close_matches(away_original, times_conhecidos, n=1, cutoff=0.45)
            
            home = match_home[0] if match_home else home_original
            away = match_away[0] if match_away else away_original

            d_home = stats[stats['Time'] == home]
            d_away = stats[stats['Time'] == away]
            
            if d_home.empty or d_away.empty: 
                continue

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
                'Liga': row['Liga'],
                'Mandante': home,
                'Visitante': away,
                'xG_Mandante': round(lamb_home, 2),
                'xG_Visitante': round(lamb_away, 2),
                'Prob_Over_1_5': round(prob_over * 100, 1),
                'Media_Gols_Esperada': round(lamb_home + lamb_away, 2),
                'Historico_Over': round(hist_over * 100, 1),
                'Momento_Recente': round(momento * 100, 1)
            })
            print(f"✅ Gerado: {home} vs {away} | Data: {row['Data']}")
        except Exception as e: 
            continue

    return pd.DataFrame(previsoes)

def main():
    auth = Auth.Token(GITHUB_TOKEN)
    g = Github(auth=auth)
    repo = g.get_repo(NOME_REPO)

    print("--- LENDO HISTÓRICO ---")
    try:
        conteudo = repo.get_contents(ARQUIVO_JOGOS)
        df_historico = pd.read_csv(StringIO(conteudo.decoded_content.decode('utf-8')))
        print("Base carregada com sucesso!")
    except Exception as e:
        print(f"Erro ao ler CSV do GitHub: {e}")
        return

    # 1. Pega os jogos do futuro
    df_novos = obter_proxima_rodada()
    
    # 2. Gera a previsão cruzando passado e futuro
    if not df_novos.empty:
        df_analise = gerar_analise(df_historico, df_novos)
        
        # 3. Salva SÓ o arquivo de previsões
        if not df_analise.empty:
            csv_analise = df_analise.to_csv(index=False)
            try:
                try:
                    contents = repo.get_contents(ARQUIVO_PREVISOES)
                    repo.update_file(contents.path, "Update Analise", csv_analise, contents.sha)
                    print("SUCESSO: analise_preditiva.csv Atualizado na Nuvem!")
                except:
                    repo.create_file(ARQUIVO_PREVISOES, "Create Analise", csv_analise)
                    print("SUCESSO: analise_preditiva.csv Criado na Nuvem!")
            except Exception as e:
                print(f"Erro ao salvar no GitHub: {e}")
        else:
            print("Aviso: Falha ao gerar tabela de previsões.")
    else:
        print("Aviso: Sem jogos futuros para prever hoje.")

if __name__ == "__main__":
    main()
