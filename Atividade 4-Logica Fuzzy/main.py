import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Definindo as variáveis fuzzy
# ENTRADA IMC varia de 10 a 50
imc = ctrl.Antecedent(np.arange(10, 40, 1), "IMC")  
atividade_fisica = ctrl.Antecedent(np.arange(0, 180, 1), "Atividade Física")  # Tempo de atividade em minutos
# SAÍDA OBESIDADE
obesidade = ctrl.Consequent(np.arange(0, 3, 1), "Obesidade")  # Níveis de obesidade de 0 a 3

imc.automf(names=["baixo", "normal", "alto"])

# Fuzzyficando a variável IMC (conjuntos fuzzy)
imc['baixo_peso'] = fuzz.trimf(imc.universe, [10, 17, 18.5])
imc['peso_normal'] = fuzz.trimf(imc.universe, [18.5, 22, 24.9])
imc['alto_peso'] = fuzz.trimf(imc.universe, [24.9, 30, 40])
imc.view()

# Fuzzyficando a variável de obesidade (conjuntos fuzzy)
obesidade['baixo'] = fuzz.trimf(obesidade.universe, [0, 2, 4])
obesidade['moderada'] = fuzz.trimf(obesidade.universe, [4, 6, 8])
obesidade['grave'] = fuzz.trimf(obesidade.universe, [8, 10, 10])
obesidade.view()

#atividade física

atividade_fisica['pouca_atividade'] = fuzz.trapmf(atividade_fisica.universe, [0, 0, 30, 45])
atividade_fisica['media_atividade'] = fuzz.trapmf(atividade_fisica.universe, [30, 45, 60, 75])
atividade_fisica['muita_atividade'] = fuzz.trapmf(atividade_fisica.universe, [60, 75, 120, 180])

# Definindo as regras fuzzy
regra1 = ctrl.Rule(imc['baixo_peso'] & atividade_fisica['muita_atividade'], obesidade['baixo'])
regra2 = ctrl.Rule(imc['peso_normal'] & atividade_fisica['media_atividade'], obesidade['moderada'])
regra3 = ctrl.Rule(imc['alto_peso'] & atividade_fisica['pouca_atividade'], obesidade['grave'])


# Criando o sistema de controle fuzzy
regras_obesidade = ctrl.ControlSystem([regra1, regra2, regra3])
obesidade_calculo = ctrl.ControlSystemSimulation(regras_obesidade)


obesidade_calculo.input['IMC'] = 29
obesidade_calculo.input['Atividade Física'] = 31
# Calculando a saída
obesidade_calculo.compute()

# Exibindo o resultado
#print(obesidade_calculo.output['Obesidade'])  # Corrigido: uso de 'Obesidade' ao invés de 'obesidade'
imc.view(sim= obesidade_calculo)
obesidade.view(sim= obesidade_calculo)

plt.show()