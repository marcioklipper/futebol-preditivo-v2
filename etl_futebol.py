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

def obter_proxima_rodada():
    print("--- 1. BUSCANDO PRÓXIMOS JOGOS (ESPN API 30 DIAS) ---")
    
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
            status = comp['status']['type']['state']
            
            if status == 'pre':
                data_jogo = evento['date'][:10]
                times = comp['competitors']
                
                mandante = next(t['team']['displayName'] for t in times if t['homeAway'] == 'home')
                visitante = next(t['team']['displayName'] for t in times if t['homeAway'] == 'away')
                
                jogos.append({
                    'Data': data_jogo,
                    'Mandante': mandante,
                    'Visitante': visitante,
                    'Liga': 'Bundesliga'
                })
        
        df_futuros = pd.DataFrame(jogos)
        if not df_futuros.empty:
            df_futuros = df_futuros.head(9)
            print(f"Encontrados {len(df_futuros)} jogos agendados na API!")
            return df_futuros
        else:
            print("Aviso: A API não retornou jogos para os próximos 30 dias.")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Erro ao conectar na API da ESPN: {e}")
        return pd.DataFrame()

def gerar_analise(df_treino, df_prever):
    print("--- 2. CÉREBRO PREDITIVO (POISSON CALIBRADO) ---")
    
    df_treino = df_treino.dropna(subset=['Gols_Mandante', 'Gols_Visitante']).copy()

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
    
    # --- NOVA VARIÁVEL: CALIBRAÇÃO DOS ÚLTIMOS 10 JOGOS ---
    def calcular_fator_forma(time):
        ultimos_10 = df_treino[(df_treino['Mandante'] == time) | (df_treino['Visitante'] == time)].sort_values('Data', ascending=False).head(10)
        if ultimos_10.empty: return 1.0
        taxa_recente = ultimos_10['Over15'].mean()
        taxa_historica = (stats[stats['Time'] == time]['Taxa_Over_Casa'].values[0] + stats[stats['Time'] == time]['Taxa_Over_Fora'].values[0]) / 2
        
        # Se a taxa histórica for 0, evita erro de divisão. Retorna o fator de ajuste.
        return (taxa_recente / taxa_historica) if taxa_historica > 0 else 1.0

    stats['Fator_10_Jogos'] = stats['Time'].apply(calcular_fator_forma)

    stats['Ataque_Casa'] = stats['Media_Feitos_Casa'] / m_liga_mand
    stats['Defesa_Casa'] = stats['Media_Sofridos_Casa'] / m_liga_visit
    stats['Ataque_Fora'] = stats['Media_Feitos_Fora'] / m_liga_visit
    stats['Defesa_Fora'] = stats['Media_Sofridos_Fora'] / m_liga_mand

    times_conhecidos = stats['Time'].unique().tolist()
    previsoes = []

    print("--- 3. CALCULANDO PROBABILIDADES ---")
    for idx, row in df_prever.iterrows():
        try:
            home_original = str(row['Mandante']).strip()
            away_original = str(row['Visitante']).strip()
            
            match_home = difflib.get_close_matches(home_original, times_conhecidos, n=1, cutoff=0.45)
            match_away = difflib.get_close_matches(away_original, times_conhecidos, n=1, cutoff=0.45)
            
            home = match_home[0] if match_home else home_original
            away = match_away[0] if match_away else away_original

            d_home = stats[stats['Time'] == home]
            d_away = stats[stats['Time'] == away]
            
            if d_home.empty or d_away.empty: 
                print(f"❌ Ignorado (Sem histórico na base): {home} vs {away}")
                continue

            # Força de Ataque e Defesa
            lamb_home = d_home['Ataque_Casa'].values[0] * d_away['Defesa_Fora'].values[0] * m_liga_mand
            lamb_away = d_away['Ataque_Fora'].values[0] * d_home['Defesa_Casa'].values[0] * m_liga_visit
            
            # Probabilidade Matemática (Poisson Puro)
            prob_under = (poisson.pmf(0, lamb_home)*poisson.pmf(0, lamb_away)) + \
                         (poisson.pmf(1, lamb_home)*poisson.pmf(0, lamb_away)) + \
                         (poisson.pmf(0, lamb_home)*poisson.pmf(1, lamb_away))
            prob_over_pura = (1 - prob_under) * 100

            # --- APLICAÇÃO DO REAJUSTE DE FORMA ---
            fator_h = d_home['Fator_10_Jogos'].values[0]
            fator_a = d_away['Fator_10_Jogos'].values[0]
            fator_combinado = (fator_h + fator_a) / 2
            
            # Multiplica a probabilidade pura pelo momento dos times
            prob_ajustada = prob_over_pura * fator_combinado
            # Cria uma trava de segurança para não passar de 95% nem cair abaixo de 5%
            prob_final = min(max(prob_ajustada, 5.0), 95.0) 

            previsoes.append({
                'Data': row['Data'],
                'Liga': row['Liga'],
                'Mandante': home,
                'Visitante': away,
                'xG_Mandante': round(lamb_home, 2),
                'xG_Visitante': round(lamb_away, 2),
                'Media_Gols_Esperada': round(lamb_home + lamb_away, 2),
                'Prob_Poisson_Pura': round(prob_over_pura, 1),
                'Fator_Forma_10j': round(fator_combinado, 2),
                'Prob_Over_Final': round(prob_final, 1)
            })
            print(f"✅ Gerado: {home} vs {away} | Poisson: {round(prob_over_pura,1)}% -> Ajustado: {round(prob_final,1)}%")
        except Exception as e: 
            continue

    return pd.DataFrame(previsoes)

def main():
    if not GITHUB_TOKEN:
        print("Erro: GITHUB_TOKEN não encontrado. Verifique os secrets do repositório.")
        return
        
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

    df_novos = obter_proxima_rodada()
    
    if not df_novos.empty:
        df_analise = gerar_analise(df_historico, df_novos)
        
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
