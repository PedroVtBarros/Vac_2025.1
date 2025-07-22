# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t, chi2, f
import imageio.v2 as imageio
# %%
def criar_gif(parametros, funcao_plot, nome_arquivo, fps=2):
    """Cria um GIF animado a partir de vários pngs gerados."""
    nomes_arquivos_frames = []
    print(f"Gerando frames para {nome_arquivo}...")
    
    for i, p in enumerate(parametros):
        funcao_plot(p)
        nome_frame = f"frame_{i:03d}.png"
        plt.savefig(nome_frame)
        plt.close()
        nomes_arquivos_frames.append(nome_frame)
    
    print(f"Montando o GIF: {nome_arquivo}...")
    with imageio.get_writer(nome_arquivo, mode='I', fps=fps) as writer:
        for nome_frame in nomes_arquivos_frames:
            imagem = imageio.imread(nome_frame)
            writer.append_data(imagem)
    
    for nome_frame in nomes_arquivos_frames:
         os.remove(nome_frame)
    
    print(f"'{nome_arquivo}' salvo com sucesso!")
# %%
# 1. Distribuição Normal
def plot_normal(desvio_padrao):
    media = 10
    x = np.linspace(-20, 40, 1000)
    y = norm.pdf(x, loc=media, scale=desvio_padrao)
    
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, color='blue', linewidth=3)
    plt.xlim(-20, 40)
    plt.ylim(0, 0.4)
    plt.xlabel('x', fontsize=14)
    plt.ylabel('f(x)', fontsize=14)
    plt.title('Distribuição Normal', fontsize=16)
    plt.text(25, 0.35, r'Parâmetros', fontsize=12, weight='bold')
    plt.text(25, 0.30, f'$\\mu = {media}$\n$\\sigma = {desvio_padrao:.2f}$', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

params_normal = np.arange(1, 15.1, 1)
criar_gif(params_normal, plot_normal, 'fdp_Normal.gif')
# %%
# 2. Distribuição t de Student
def plot_t(graus_liberdade):
    x = np.linspace(-10, 10, 1000)
    y = t.pdf(x, df=graus_liberdade)
    
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, color='green', linewidth=3)
    plt.xlim(-10, 10)
    plt.ylim(0, 0.4)
    plt.xlabel('t', fontsize=14)
    plt.ylabel('f(t)', fontsize=14)
    plt.title('Distribuição t de Student', fontsize=16)
    y_norm = norm.pdf(x, loc=0, scale=1)
    plt.plot(x, y_norm, color='black', linestyle='--', label='Normal Padrão (ref)')
    plt.text(6, 0.35, r'Parâmetros', fontsize=12, weight='bold')
    plt.text(6, 0.30, f'$\\phi = {graus_liberdade}$', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

params_t = np.arange(1, 30.1, 1)
criar_gif(params_t, plot_t, 'fdp_Student.gif')
# %%
# 3. Distribuição Qui-quadrado
def plot_chi2(graus_liberdade):
    x = np.linspace(0.01, 80, 1000)
    y = chi2.pdf(x, df=graus_liberdade)
    
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, color='red', linewidth=3)
    plt.xlim(0, 80)
    plt.ylim(0, 0.5)
    plt.xlabel(r'$\chi^2$', fontsize=14)
    plt.ylabel(r'$f(\chi^2)$', fontsize=14)
    plt.title('Distribuição Qui-Quadrado', fontsize=16)
    plt.text(60, 0.4, r'Parâmetros', fontsize=12, weight='bold')
    plt.text(60, 0.35, f'$\\varphi = {graus_liberdade}$', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

params_chi2 = np.arange(1, 40.1, 1)
criar_gif(params_chi2, plot_chi2, 'fdp_Qui-Quadrado.gif')
# %%
# 4. Distribuição F
def plot_f(df1):
    df2 = 15  # Graus de liberdade do denominador (fixo)
    x = np.linspace(0.01, 10, 1000)
    y = f.pdf(x, dfn=df1, dfd=df2)
    
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, color='purple', linewidth=3)
    plt.xlim(0, 10)
    plt.ylim(0, 1.2)
    plt.xlabel('F', fontsize=14)
    plt.ylabel('f(F)', fontsize=14)
    plt.title('Distribuição F', fontsize=16)
    plt.text(7, 0.9, r'Parâmetros', fontsize=12, weight='bold')
    plt.text(7, 0.8, f'$\\varphi_1 = {df1}$\n$\\varphi_2 = {df2}$', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

params_f = np.arange(1, 50.1, 1)
criar_gif(params_f, plot_f, 'fdp_F.gif')

print("\nProcesso finalizado! Todos os GIFs foram gerados.")
# %%
