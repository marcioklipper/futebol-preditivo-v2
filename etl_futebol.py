import pandas as pd
import numpy as np
import os
from github import Github, Auth
from io import StringIO
import requests
from scipy.stats import poisson
import difflib
from datetime import datetime, timedelta
import html5lib

# --- CONFIGURAÇÕES ---
GITHUB_TOKEN = os.getenv('GH_TOKEN')
NOME_REPO = "marcioklipper/futebol-preditivo-v2"
ARQUIVO_JOGOS = "historico_jogos.csv"
ARQUIVO_PREVISOES = "analise_preditiva.csv"
ARQUIVO_HIST_RECENTE = "historico_10j_times.csv"

# --- 1. ATUALIZADOR INFALÍVEL (FBREF) ---
def atualizar_historico(df_historico_atual):
    print("--- 1. BUSCANDO RESULTADOS PASSADOS (FBREF) ---")
    
    # URL da tabela completa da temporada atual da Bundesliga no FBref
    url = "https://fbref.com/en/comps/20/schedule/Bundesliga-Scores-and-Fixtures"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    try:
        r = requests.get(url, headers=headers)
        # O FBref esconde algumas tabelas em comentários HTML, essa função do Pandas é poderosa
        dfs = pd.read_html(StringIO(r.text), flavor='lxml')
        
        jogos_novos = []
        for df in dfs:
            if 'Home' in df.columns and 'Away' in df.columns and 'Score' in df.columns:
                # Remove linhas de divisória/cabeçalho da tabela
                df = df.dropna(subset=['Home', 'Away', 'Date'])
                
                # Filtra apenas jogos que já têm um placar preenchido (que já aconteceram)
                df_encerrados = df[df['Score'].notna() & (df['Score'] != '')].copy()
                
                for _, row in df_encerrados.iterrows():
                    try:
                        data_jogo = str(row['Date'])
                        mandante = str(row['Home'])
                        visitante = str(row['Away'])
                        placar = str(row['Score'])
                        
                        # O placar vem no formato "2–1" (pode usar um traço diferente, por isso a limpeza)
                        placar = placar.replace('–', '-').replace('—', '-')
                        partes = placar.split('-')
                        
                        if len(partes) == 2:
                            gm = int(partes[0].strip())
                            gv = int(partes[1].strip())
                            
                            jogos_novos.append({
                                'Data': data_jogo,
                                'Mandante': mandante,
                                'Visitante': visitante,
                                'Liga': 'Bundesliga',
                                'Gols_Mandante': gm,
                                'Gols_Visitante': gv
                            })
                    except Exception as e:
                        pass # Ignora linhas mal formatadas
                
                break # Achou a tabela principal, pode parar o loop
        
        df_novos = pd.DataFrame(jogos_novos)
        
        if not df_novos.empty:
            df_atualizado = pd.concat([df_historico_atual, df_novos], ignore_index=True)
            # Remove duplicatas. O pulo do gato: mantém os dados do FBref (que são mais limpos) caso haja conflito.
            df_atualizado = df_atualizado.drop_duplicates(subset=['Data', 'Mandante', 'Visitante'], keep='last')
            
            jogos_adicionados = len(df_atualizado) - len(df_historico_atual)
            print(f"Base atualizada com sucesso pelo FBref! Foram adicionados/corrigidos {jogos_adicionados} jogos.")
            return df_atualizado
        else:
            print("Nenhum jogo novo encontrado para adicionar.")
            return df_historico_atual
            
    except Exception as e:
        print(f"Erro ao buscar resultados passados no FBref: {e}")
        return df_historico_atual

def obter_proxima_rodada():
    print("--- 2. BUSCANDO PRÓXIMOS JOGOS (ESPN API 30 DIAS) ---")
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
                jogos.append({'Data': data_jogo, 'Mandante': mandante, 'Visitante': visitante, 'Liga': 'Bundesliga'})
        
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
    print("--- 3. CÉREBRO PREDITIVO E EXTRATOR DE HISTÓRICO ---")
    df_treino = df_treino.dropna(subset=['Gols_Mandante', 'Gols_Visitante']).copy()
    if df_treino.empty or df_prever.empty: return pd.DataFrame(), pd.DataFrame()
    print(f"Treinando o modelo com {len(df_treino)} jogos passados válidos...")

    df_treino['Total_Gols'] = df_treino['Gols_Mandante'] + df_treino['Gols_Visitante']
    df_treino['Over15'] = np.where(df_treino['Total_Gols'] > 1.5, 1, 0)
    m_liga_mand = df_treino['Gols_Mandante'].mean()
    m_liga_visit = df_treino['Gols_Visitante'].mean()

    stats_casa = df_treino.groupby('Mandante').agg(Media_Feitos_Casa=('Gols_Mandante', 'mean'), Media_Sofridos_Casa=('Gols_Visitante', 'mean'), Taxa_Over_Casa=('Over15', 'mean')).reset_index().rename(columns={'Mandante': 'Time'})
    stats_fora = df_treino.groupby('Visitante').agg(Media_Feitos_Fora=('Gols_Visitante', 'mean'), Media_Sofridos_Fora=('Gols_Mandante', 'mean'), Taxa_Over_Fora=('Over15', 'mean')).reset_index().rename(columns={'Visitante': 'Time'})
    stats = pd.merge(stats_casa, stats_fora, on='Time', how='outer').fillna(0)
    
    def calcular_fator_forma(time):
        ultimos_10 = df_treino[(df_treino['Mandante'] == time) | (df_treino['Visitante'] == time)].sort_values('Data', ascending=False).head(10)
        if ultimos_10.empty: return 1.0
        taxa_recente = ultimos_10['Over15'].mean()
        taxa_historica = (stats[stats['Time'] == time]['Taxa_Over_Casa'].values[0] + stats[stats['Time'] == time]['Taxa_Over_Fora'].values[0]) / 2
        return (taxa_recente / taxa_historica) if taxa_historica > 0 else 1.0

    stats['Fator_10_Jogos'] = stats['Time'].apply(calcular_fator_forma)
    stats['Ataque_Casa'] = stats['Media_Feitos_Casa'] / m_liga_mand
    stats['Defesa_Casa'] = stats['Media_Sofridos_Casa'] / m_liga_visit
    stats['Ataque_Fora'] = stats['Media_Feitos_Fora'] / m_liga_visit
    stats['Defesa_Fora'] = stats['Media_Sofridos_Fora'] / m_liga_mand

    times_conhecidos = stats['Time'].unique().tolist()
    previsoes = []
    tabela_historico = []
    times_processados = set()

    print("--- 4. CALCULANDO PROBABILIDADES ---")
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
            print(f"✅ Gerado Previsão + Histórico: {home} vs {away}")
        except Exception as e: continue
    return pd.DataFrame(previsoes), pd.DataFrame(tabela_historico)

def salvar_no_github(repo, nome_arquivo, df_dados, mensagem):
    csv_string = df_dados.to_csv(index=False)
    try:
        try:
            contents = repo.get_contents(nome_arquivo)
            repo.update_file(contents.path, f"Update {mensagem}", csv_string, contents.sha)
            print(f"SUCESSO: {nome_arquivo} Atualizado!")
        except:
            repo.create_file(nome_arquivo, f"Create {mensagem}", csv_string)
            print(f"SUCESSO: {nome_arquivo} Criado!")
    except Exception as e: print(f"Erro ao salvar {nome_arquivo}: {e}")

def main():
    if not GITHUB_TOKEN: return
    auth = Auth.Token(GITHUB_TOKEN)
    g = Github(auth=auth)
    repo = g.get_repo(NOME_REPO)

    print("--- LENDO HISTÓRICO ---")
    try:
        conteudo = repo.get_contents(ARQUIVO_JOGOS)
        df_historico_base = pd.read_csv(StringIO(conteudo.decoded_content.decode('utf-8')))
        print("Base carregada com sucesso!")
    except Exception as e: return

    df_historico_atualizado = atualizar_historico(df_historico_base)
    
    if len(df_historico_atualizado) > len(df_historico_base):
        salvar_no_github(repo, ARQUIVO_JOGOS, df_historico_atualizado, "Historico de Jogos Realizados")

    df_novos = obter_proxima_rodada()
    
    if not df_novos.empty:
        df_analise, df_hist_recente = gerar_analise(df_historico_atualizado, df_novos)
        if not df_analise.empty:
            salvar_no_github(repo, ARQUIVO_PREVISOES, df_analise, "Previsoes")
            salvar_no_github(repo, ARQUIVO_HIST_RECENTE, df_hist_recente, "Historico Detalhado")
        else: print("Aviso: Falha ao gerar tabelas.")
    else: print("Aviso: Sem jogos futuros para prever hoje.")

if __name__ == "__main__":
    main()
