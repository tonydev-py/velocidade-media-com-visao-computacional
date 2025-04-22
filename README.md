# Velocidade Média com Visão Computacional

🚗 Projeto de detecção e cálculo de velocidade média de veículos utilizando **YOLOv5** e **visão computacional** com Python.

## 📌 Objetivo

Este projeto tem como objetivo utilizar técnicas de **visão computacional** para identificar veículos em movimento em um vídeo e calcular a **velocidade média** deles, com base na distância entre dois pontos fixos no ambiente.

## 🛠️ Tecnologias utilizadas

- **Python 3**
- **OpenCV** - Para processamento de vídeo e imagens
- **YOLOv5** - Para detecção de objetos (veículos)
- **PyTorch** - Framework para deep learning
- **NumPy** - Manipulação de arrays
- **Matplotlib** - Para visualização gráfica (se aplicável)
- **Garbage Collection (gc)** - Para otimização de memória
- **Time** - Para medição de tempo e controle de execução

## 🎯 Funcionalidades

- Detecta veículos em um vídeo ou webcam usando YOLOv5
- Marca os pontos de entrada e saída do trajeto
- Calcula a velocidade média (convertendo de m/s para km/h)
- Exibe a velocidade de cada veículo detectado sobre a imagem
