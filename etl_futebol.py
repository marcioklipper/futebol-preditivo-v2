import pandas as pd
import numpy as np
import os
from github import Github, Auth
from io import StringIO
import requests
from scipy.stats import poisson
import difflib
from bs4 import BeautifulSoup # <-- Certifique-se de ter esta importação

# ... (Configurações e outras partes do código ficam iguais) ...

def obter_proxima_rodada():
    print("--- 1. BUSCANDO PRÓXIMOS JOGOS (ESPN CALENDÁRIO) ---")
    # Agora apontamos direto para a página de calendário da Bundesliga (como você fez no navegador)
    url = "https://www.espn.com.br/futebol/calendario/_/liga/GER.1"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    try:
        r = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(r.content, 'html.parser')
        
        jogos = []
        
        # O HTML da ESPN agrupa os jogos por dias, então precisamos achar as tabelas de calendário
        tabelas = soup.find_all('table', class_='Table')
        
        if not tabelas:
             print("Aviso: Nenhuma tabela de jogos encontrada no calendário da ESPN.")
             return pd.DataFrame()

        for tabela in tabelas:
            linhas = tabela.find_all('tr', class_='Table__TR')
            for linha in linhas:
                 colunas = linha.find_all('td')
                 
                 # Uma linha válida de jogo futuro costuma ter pelo menos 3 colunas (Mandante, Placar/Hora, Visitante)
                 if len(colunas) >= 3:
                     # A coluna 0 geralmente tem o time mandante, a 1 o placar/horário, e a 2 o visitante.
                     try:
                         # Extrai os nomes dos times. O span com a classe indica o nome do time.
                         mandante = colunas[0].find_all('a')[1].text.strip() if len(colunas[0].find_all('a')) > 1 else colunas[0].text.strip()
                         visitante = colunas[2].find_all('a')[1].text.strip() if len(colunas[2].find_all('a')) > 1 else colunas[2].text.strip()
                         
                         # Se o "placar" for um horário (ex: 10:30) ou 'v', é um jogo futuro
                         placar_ou_hora = colunas[1].text.strip()
                         if ':' in placar_ou_hora or 'v' in placar_ou_hora.lower():
                             # Pega a data da seção (ou usa a data atual como placeholder, já que não precisamos do dia exato para prever, só saber quem joga)
                             
                             jogos.append({
                                 'Data': '2025-03-29', # Simplificando: Assumimos a data da próxima rodada com base na sua imagem
                                 'Mandante': mandante,
                                 'Visitante': visitante,
                                 'Liga': 'Bundesliga'
                             })
                     except Exception as e:
                         pass # Ignora linhas de cabeçalho ou mal formatadas
        
        df_futuros = pd.DataFrame(jogos)
        
        if not df_futuros.empty:
            # Pega apenas os primeiros 9 jogos únicos (uma rodada completa)
            df_futuros = df_futuros.drop_duplicates(subset=['Mandante', 'Visitante']).head(9)
            print(f"Encontrados {len(df_futuros)} jogos futuros da próxima rodada no calendário!")
            return df_futuros
        else:
            print("Aviso: Não encontrou jogos futuros na raspagem da página.")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Erro ao conectar na ESPN: {e}")
        return pd.DataFrame()

def gerar_analise(df_treino, df_prever):
    print("--- 2. CÉREBRO PREDITIVO (POISSON) ---")
    
    # Faxina: Remove os "Fantasmas" (jogos antigos sem gols)
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
            
            # O Tradutor Automático de Nomes entra em ação novamente
            match_home = difflib.get_close_matches(home_original, times_conhecidos, n=1, cutoff=0.45)
            match_away = difflib.get_close_matches(away_original, times_conhecidos, n=1, cutoff=0.45)
            
            home = match_home[0] if match_home else home_original
            away = match_away[0] if match_away else away_original

            d_home = stats[stats['Time'] == home]
            d_away = stats[stats['Time'] == away]
            
            if d_home.empty or d_away.empty: 
                print(f"❌ Ignorado (Sem histórico na base): {home} vs {away}")
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

    # 1. Tenta pegar os jogos futuros via API
    df_novos = obter_proxima_rodada()
    
    # --- O TRUQUE PARA TESTAR O POWER BI HOJE ---
    if df_novos.empty:
        print("\n⚠️ Como não há jogos na ESPN, ativando MODO SIMULAÇÃO para gerar arquivo de teste...")
        df_sorted = df_historico.sort_values(by=['Data', 'Mandante'])
        
        # Pega a última rodada (9 jogos) do histórico e finge que é o futuro
        df_novos = df_sorted.tail(9).copy()
        df_novos['Gols_Mandante'] = np.nan
        df_novos['Gols_Visitante'] = np.nan
        
        # Tira esses jogos do histórico de treino para o robô não "colar" na prova
        df_historico = df_historico.drop(df_novos.index)
        print(f"Simulando {len(df_novos)} jogos de teste...\n")
    # --------------------------------------------

    # 2. Gera a previsão
    df_analise = gerar_analise(df_historico, df_novos)
    
    # 3. Salva
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

if __name__ == "__main__":
    main()
