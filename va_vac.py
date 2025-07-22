# %%
#-----------------------------------------------------------------------
# Universidade Estadual de Santa Cruz - UESC
# Título: Variáveis aleatórias contínuas - VAC
# Curso : Estatistica e probabilidade
# Grupo: Pedro Vitor Moura de Araujo Barros
#        Rodrigo Almeida Piropo
#-----------------------------------------------------------------------
# Objetivos:
# a) Compreensão das funções de densidade de probabilidades
# b) Cálculo de probabilidades via integrais dessas funções
#-----------------------------------------------------------------------

import numpy as np # Para operações vetoriais
import matplotlib.pyplot as plt # Gráficos
from scipy import integrate # Integração numérica 
from scipy.stats import norm, t, chi2, f # Distribuições estatísticas
from scipy.special import gamma # Função gama, usada em distribuições como t, qui-quadrado e F
import math # Funções matemáticas básicas

#-----------------------------------------------------------------------
# Exemplos - FDP (Função de Densidade de Probabilidade)
#-----------------------------------------------------------------------

# %%
#--------------------
#.. Exemplo 0
#--------------------
# Definindo uma FDP
def fdp_x0(x):

    return np.where((x >= 0) & (x <= 2), 3/8 * (4*x - 2*x**2), 0)

tes = np.linspace(-1, 3, 1000)

# Gráfico da função
plt.figure(figsize=(10, 6))
plt.plot(tes, fdp_x0(tes), color='red', linewidth=2, label='f(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.xlim(-1, 3)
plt.title('Função de Densidade de Probabilidade - Exemplo 0')

#... Verificando se fdp_x0 é uma FDP
# Condição I: f(x) >= 0
print("Exemplo 0: Todos os valores >= 0?", np.all(fdp_x0(np.arange(-1, 3.1, 0.1)) >= 0))

# Condição II: Integral total deve ser 1
intg_x, _ = integrate.quad(fdp_x0, 0, 2)
print(f"Exemplo 0: Integral de 0 a 2: {intg_x:.5f}")

# P(X > 1)
# Gráfico da área
x_fill = np.linspace(1, 2, 100)
plt.fill_between(x_fill, fdp_x0(x_fill), color='gray', alpha=0.5)

# Valor da probabilidade
p_x_gt_1, _ = integrate.quad(fdp_x0, 1, np.inf)
print(f"Exemplo 0: P(X > 1): {p_x_gt_1:.5f}")

# Adicionando texto ao gráfico
plt.text(1.5, 0.1, f'P(X > 1) = {p_x_gt_1:.1f}', fontsize=12, color='blue')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()


# %%
#--------------------
#.. Exemplo 1
#--------------------
def fdp_y1(y):
    return np.where((y >= 0) & (y <= 1), 2*y, 0)

tes = np.linspace(-1, 2, 1000)

# Gráfico da função
plt.figure(figsize=(10, 6))
plt.plot(tes, fdp_y1(tes), color='red', linewidth=2, label='f(y)')
plt.xlabel('y')
plt.ylabel('f(y)')
plt.xlim(-1, 2)
plt.title('Função de Densidade de Probabilidade - Exemplo 1')

#... Verificando se fdp_y1 é uma FDP
print("\nExemplo 1: Todos os valores >= 0?", np.all(fdp_y1(np.arange(-1, 3.1, 0.1)) >= 0))
intg_y, _ = integrate.quad(fdp_y1, 0, 1)
print(f"Exemplo 1: Integral de 0 a 1: {intg_y:.5f}")

# Visualizando algumas probabilidades
# Entre 0 e 0.5
x_fill1 = np.linspace(0, 0.5, 100)
plt.fill_between(x_fill1, fdp_y1(x_fill1), color='gray', alpha=0.8)

# Entre 0.5 e 1
x_fill2 = np.linspace(0.5, 1, 100)
plt.fill_between(x_fill2, fdp_y1(x_fill2), color='darkgray', alpha=0.6)

# Calculando e mostrando probabilidades no gráfico
p_0_05, _ = integrate.quad(fdp_y1, 0, 0.5)
p_05_1, _ = integrate.quad(fdp_y1, 0.5, 1)

plt.text(0.1, 0.2, f'P(0 < Y < 0.5)\n= {p_0_05:.2f}', fontsize=10, color='blue')
plt.text(0.7, 0.5, f'P(0.5 < Y < 1)\n= {p_05_1:.2f}', fontsize=10, color='blue')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()


# %%
#--------------------
#.. Exemplo 2
#--------------------
def fdp_x2(x):
    return np.where(x < 0, 0, 2 * np.exp(-2*x))

tes = np.linspace(-1, 4, 1000)

# Gráfico da função
plt.figure(figsize=(10, 6))
plt.plot(tes, fdp_x2(tes), color='red', linewidth=2, label='f(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.xlim(-1, 4)
plt.title('Função de Densidade de Probabilidade - Exemplo 2 (Exponencial)')

#... Verificando se fdp_x2 é uma FDP
print("\nExemplo 2: Todos os valores >= 0?", np.all(fdp_x2(np.arange(-1, 100, 0.1)) >= 0))
intg_x2, _ = integrate.quad(fdp_x2, 0, np.inf)
print(f"Exemplo 2: Integral de 0 a infinito: {intg_x2:.5f}")

# Visualizando probabilidades
# Área a: P(X > 1)
x_fill_a = np.linspace(1, 5, 100)
plt.fill_between(x_fill_a, fdp_x2(x_fill_a), color='yellow', alpha=0.5)

# Área b: P(0.0 < X < 1.0)
x_fill_b = np.linspace(0.0, 1.0, 100)
plt.fill_between(x_fill_b, fdp_x2(x_fill_b), color='darkgray', alpha=0.8)

# Calculando e mostrando probabilidades
p_a, _ = integrate.quad(fdp_x2, 1, np.inf)
p_b, _ = integrate.quad(fdp_x2, 0.00, 1.00)

plt.text(1.6, 0.3, f'Pa = P(X>1)\n= {p_a:.2f}', fontsize=10, color='blue')
plt.text(0.3, 1.0, f'Pb = P(0.0<X<1.0)\n= {p_b:.2f}', fontsize=10, color='blue')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()

# %%
#--------------------------------------------------------------------------
#. PRINCIPAIS DISTRIBUIÇÕES
# - Mostrar a igualdade entre a equação teórica e a implementada pelo Python (Scipy)
# - Apresentar as funções do Scipy para cada distribuição
#   - Densidade de probabilidade      (.pdf)
#   - Probabilidade acumulada         (.cdf)
#   - Quantil (inversa da cdf)        (.ppf)
#   - Amostras aleatórias             (.rvs)
#--------------------------------------------------------------------------

# %%
#-----------------------------------
#.. Distribuição Normal
#-----------------------------------
print("\n--- Distribuição Normal ---")

# Função de densidade da Normal implementada manualmente
# x: valores para os quais queremos calcular a densidade
# m: média da distribuição
# s: desvio padrão
def fdp_norm_manual(x, m, s):
    return 1/(s * np.sqrt(2 * np.pi)) * np.exp(-1/2 * ((x - m) / s)**2)

# Gera 1000 pontos igualmente espaçados entre 0 e 20
tes = np.linspace(0, 20, 1000)

# Cria gráfico com matplotlib
plt.figure(figsize=(10, 6))
plt.plot(tes, fdp_norm_manual(tes, 10, 2), linewidth=3, label='Nossa Implementação')  # curva normal manual
plt.plot(tes, norm.pdf(tes, loc=10, scale=2), 'r--', linewidth=2, label='Scipy: norm.pdf')  # função do scipy
plt.title('Distribuição Normal: Implementação Manual vs. Scipy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Calcula a integral da função de densidade de -∞ a ∞ (deve dar 1)
integral_total, _ = integrate.quad(fdp_norm_manual, -np.inf, np.inf, args=(10, 2))
print(f"Integral de -inf a inf: {integral_total}")

# Probabilidade de X ≤ 10 usando scipy (área até a média)
prob_ate_media, _ = integrate.quad(norm.pdf, -np.inf, 10, args=(10, 2))
print(f"Probabilidade até a média (10): {prob_ate_media}")

# Exemplo com uma distribuição N(100, 10)
print("\nExemplo com Scipy: X ~ N(100, 10)")

# a) Probabilidade de X < 95
p_a = norm.cdf(95, loc=100, scale=10)
print(f"P(X < 95) = {p_a:.4f}")

# b) Probabilidade de X entre 90 e 110
p_b = norm.cdf(110, 100, 10) - norm.cdf(90, 100, 10)
print(f"P(90 < X < 110) = {p_b:.4f}")

# c) Probabilidade de X > 95 (duas formas de calcular)
p_c = 1 - norm.cdf(95, 100, 10)
p_c_sf = norm.sf(95, 100, 10)  # survival function
print(f"P(X > 95) = {p_c:.4f} (usando 1-cdf)")
print(f"P(X > 95) = {p_c_sf:.4f} (usando sf)")

# Quantis: qual valor x tem probabilidade acumulada igual à de X=10
prob_x = norm.cdf(10, loc=10, scale=2)
quantil = norm.ppf(prob_x, loc=10, scale=2)
print(f"\nProbabilidade P(X<=10) é {prob_x:.2f}. O quantil para essa prob. é: {quantil:.2f}")

# Quantil para 40% dos menores valores
quantil_40 = norm.ppf(0.40, loc=10, scale=2)
print(f"Quantil que delimita 40% dos menores valores: {quantil_40:.4f}")

# Geração de 10.000 amostras aleatórias de uma normal com média 10 e desvio 2
am_n = norm.rvs(loc=10, scale=2, size=10000)

# Histograma da amostra vs. curva teórica
plt.figure(figsize=(10, 6))
plt.hist(am_n, bins=50, density=True, alpha=0.7, label='Histograma da Amostra')
x_plot = np.linspace(am_n.min(), am_n.max(), 200)
plt.plot(x_plot, norm.pdf(x_plot, loc=10, scale=2), 'r-', lw=2, label='PDF Teórica')
plt.title('Histograma de Amostra Normal vs. PDF Teórica')
plt.legend()
plt.show()

# Plot de várias curvas normais com mesma média e diferentes desvios
plt.figure(figsize=(10, 6))
nc = np.arange(1, 10.5, 0.5)  # lista de desvios
colors = plt.cm.rainbow(np.linspace(0, 1, len(nc)))
for i, c in zip(nc, colors):
    x_vals = np.linspace(-20, 40, 1000)
    plt.plot(x_vals, norm.pdf(x_vals, loc=10, scale=i), color=c, label=f'σ={i}')
plt.xlim(-20, 40); plt.ylim(0, 0.45); plt.xlabel('x'); plt.ylabel('f(x)')
plt.title('Várias Distribuições Normais (μ=10, σ variável)')
plt.legend()
plt.show()


# %%
#-----------------------------------
#.. Distribuição t de Student
#-----------------------------------
print("\n--- Distribuição t de Student ---")
# df: graus de liberdade (degrees of freedom)
def fdp_t_manual(x, df):
    num = gamma((df + 1) / 2)
    den = np.sqrt(df * np.pi) * gamma(df / 2)
    return (num / den) * (1 + x**2 / df)**(- (df + 1) / 2)

tes = np.linspace(-10, 10, 1000)
plt.figure(figsize=(10, 6))
plt.plot(tes, fdp_t_manual(tes, 10), linewidth=3, label='Nossa Implementação')
plt.plot(tes, t.pdf(tes, df=10), 'r--', linewidth=2, label='Scipy: t.pdf')
plt.title('Distribuição t (df=10): Manual vs. Scipy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Comparando integral com a CDF do Scipy
prob_ate_0, _ = integrate.quad(fdp_t_manual, -np.inf, 0, args=(10,))
prob_ate_0_scipy = t.cdf(0, df=10)
print(f"Integral de -inf a 0 (manual): {prob_ate_0:.4f}")
print(f"Scipy t.cdf(0, df=10): {prob_ate_0_scipy:.4f}")
print(f"São aproximadamente iguais? {np.isclose(prob_ate_0, prob_ate_0_scipy)}")

# Quantil que delimita 15% dos menores valores
quantil_15_t = t.ppf(0.15, df=10)
print(f"Quantil t que delimita 15% dos menores valores (df=10): {quantil_15_t:.4f}")

# Amostra aleatória
am_t = t.rvs(df=10, size=10000)
plt.figure(figsize=(10, 6))
plt.hist(am_t, bins=50, density=True, alpha=0.7, label='Histograma da Amostra t')
x_plot = np.linspace(am_t.min(), am_t.max(), 200)
plt.plot(x_plot, t.pdf(x_plot, df=10), 'r-', lw=2, label='PDF Teórica t(10)')
plt.title('Histograma de Amostra t vs. PDF Teórica')
plt.legend()
plt.show()

# Várias curvas t sobrepostas e a Normal Padrão
plt.figure(figsize=(10, 6))
nc = np.arange(1, 21, 1)
colors = plt.cm.viridis(np.linspace(0, 1, len(nc)))
for i, c in zip(nc, colors):
    x_vals = np.linspace(-10, 10, 1000)
    plt.plot(x_vals, t.pdf(x_vals, df=i), color=c, label=f'df={i}')
# Sobrepondo uma Normal Padrão para comparação
plt.plot(x_vals, norm.pdf(x_vals), 'k--', lw=2, label='Normal Padrão (Ref.)')
plt.xlim(-5, 5); plt.ylim(0, 0.45); plt.xlabel('t'); plt.ylabel('f(t)')
plt.title('Distribuições t (df variável) vs. Normal Padrão')
plt.legend()
plt.show()


# %%
#-----------------------------------
#.. Distribuição Qui-quadrado
#-----------------------------------
print("\n--- Distribuição Qui-quadrado ---")
def fdp_chisq_manual(x, df):
    # a função só é válida para x > 0
    x = np.asarray(x)
    val = np.zeros_like(x, dtype=float)
    idx = x > 0
    val[idx] = (1 / (2**(df/2) * gamma(df/2))) * x[idx]**(df/2 - 1) * np.exp(-x[idx]/2)
    return val

tes = np.linspace(0, 50, 1000)
plt.figure(figsize=(10, 6))
plt.plot(tes, fdp_chisq_manual(tes, 10), linewidth=3, label='Nossa Implementação')
plt.plot(tes, chi2.pdf(tes, df=10), 'r--', linewidth=2, label='Scipy: chi2.pdf')
plt.title('Distribuição Qui-quadrado (df=10): Manual vs. Scipy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Comparando integral com a CDF do Scipy
p1_manual, _ = integrate.quad(fdp_chisq_manual, 0, 20, args=(10,))
p1_scipy = chi2.cdf(20, df=10)
print(f"P(X < 20) com df=10 (manual): {p1_manual:.4f}")
print(f"P(X < 20) com df=10 (scipy): {p1_scipy:.4f}")

# Quantil que delimita 50% dos menores valores (mediana)
quantil_50_chi2 = chi2.ppf(0.50, df=10)
print(f"Quantil Qui-quadrado que delimita 50% (mediana) (df=10): {quantil_50_chi2:.4f}")

# Amostra aleatória
am_chi2 = chi2.rvs(df=10, size=10000)
plt.figure(figsize=(10, 6))
plt.hist(am_chi2, bins=50, density=True, alpha=0.7, label='Histograma da Amostra')
x_plot = np.linspace(am_chi2.min(), am_chi2.max(), 200)
plt.plot(x_plot, chi2.pdf(x_plot, df=10), 'r-', lw=2, label='PDF Teórica $\chi^2(10)$')
plt.title('Histograma de Amostra Qui-quadrado vs. PDF Teórica')
plt.legend()
plt.show()


# %%
#-----------------------------------
#.. Distribuição F
#-----------------------------------
print("\n--- Distribuição F ---")
# df1: graus de liberdade do numerador, df2: graus de liberdade do denominador
def fdp_f_manual(x, df1, df2):
    x = np.asarray(x)
    val = np.zeros_like(x, dtype=float)
    idx = x > 0
    
    term1 = gamma((df1 + df2) / 2) / (gamma(df1 / 2) * gamma(df2 / 2))
    term2 = (df1 / df2)**(df1 / 2)
    term3 = x[idx]**(df1 / 2 - 1)
    term4 = (1 + (df1 / df2) * x[idx])**(- (df1 + df2) / 2)
    
    val[idx] = term1 * term2 * term3 * term4
    return val
    
tes = np.linspace(0, 10, 1000)
df1, df2 = 5, 15
plt.figure(figsize=(10, 6))
plt.plot(tes, fdp_f_manual(tes, df1, df2), linewidth=3, label='Nossa Implementação')
plt.plot(tes, f.pdf(tes, dfn=df1, dfd=df2), 'r--', linewidth=2, label='Scipy: f.pdf')
plt.title(f'Distribuição F (df1={df1}, df2={df2}): Manual vs. Scipy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Comparando integral com a CDF do Scipy
p1_manual, _ = integrate.quad(fdp_f_manual, 0, 3, args=(df1, df2))
p1_scipy = f.cdf(3, dfn=df1, dfd=df2)
print(f"P(X < 3) com df1={df1}, df2={df2} (manual): {p1_manual:.4f}")
print(f"P(X < 3) com df1={df1}, df2={df2} (scipy): {p1_scipy:.4f}")

# Quantil que delimita 60% dos menores valores
quantil_60_f = f.ppf(0.60, dfn=df1, dfd=df2)
print(f"Quantil F que delimita 60% dos menores valores: {quantil_60_f:.4f}")

# Amostra aleatória
am_f = f.rvs(dfn=df1, dfd=df2, size=10000)
plt.figure(figsize=(10, 6))
plt.hist(am_f, bins=50, density=True, alpha=0.7, label='Histograma da Amostra')
x_plot = np.linspace(am_f.min(), am_f.max(), 200)
plt.plot(x_plot, f.pdf(x_plot, dfn=df1, dfd=df2), 'r-', lw=2, label=f'PDF Teórica F({df1},{df2})')
plt.title('Histograma de Amostra F vs. PDF Teórica')
plt.legend()
plt.show()
# %%
