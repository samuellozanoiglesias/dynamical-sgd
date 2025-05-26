# El aprendizaje automático como sistema físico fuera del equilibrio

> En lugar de utilizar una función de coste que evoluciona dinámicamente, seguimos un enfoque diferente. En la práctica, no se utiliza todo el conjunto de datos a la vez para optimizar, principalmente porque no es eficiente computacionalmente. En su lugar, se seleccionan aleatoriamente pequeños subconjuntos del conjunto de datos en cada iteración, lo que se conoce como descenso de gradiente estocástico (SGD). El enfoque que adoptamos aquí combina esta idea con la pérdida dinámica: el subconjunto de datos (o lote) que tomamos en cada iteración incluye ejemplos aleatorios de distintas clases, pero en proporciones cíclicas que varían en el tiempo.

> Utilizando este método, nos centramos en entender cómo se reconfigura internamente una red neuronal entrenada mediante lotes de composición dinámica. Nuestro objetivo es estudiar si esta dinámica induce una reorganización progresiva de los parámetros, similar a la resolución secuencial de frustraciones observada en redes físicas. Para ello, analizaremos desde métricas globales -como la distancia $L_2$ al origen y la evolución macroscópica de las fronteras de decisión- hasta aspectos que llamaremos microscópicos -como los gradientes principales, la evolución de los parámetros capa por capa y la geometría de sus distribuciones. En conjunto, estos experimentos buscan identificar patrones que emerjan de forma espontánea y permitan entender cómo se coordinan las distintas partes de la red para resolver distintas tareas sin interferencias destructivas.

![Training visualization](dynamical-loss.png)
# Autor
Este código fue escrito por [Nicolas Ratier Werbin](mailto:nicolasratierwerbin@gmail.com).
<!-- # Figuras Adicionales
## Análisis macroscópico de la dinámica de aprendizaje
### Resultados con amplitud $A = 70$
![AGV_vUdcsz0arOtjC137aWONzxckbqmDTWWqs2jM7RN36V6-r2vGJteTQ0j3XEqZcJ_mtKl71ieSfFA1TA9jKURYUAIn2IVdeozAk-ZlImBhFuxiIg5wUx0uaLxx](https://github.com/user-attachments/assets/c2e781a0-cc0e-424d-bf9e-cc385f3c65f0)

### Resultados con amplitud $A=10$
![IMG_6F23003FE771-1](https://github.com/user-attachments/assets/ab07b6b7-e9ef-44f4-afff-93fdb356c9e9)

### Resultados sin oscilaciones, $A=1$
![image](https://github.com/user-attachments/assets/f258106a-a847-4ddf-9c93-79b0ad688c5b)

## Análisis microscópico de la reconfiguración interna

### Capa oculta de tamaño $N=500$
![AGV_vUdr-N_VUTTAIUcbcn2i_-ymVrGowy_J24re4nVHk9YsNMSVsgwLSfvXpZk0Z3qVV7UWq6SOdfgjpSeCYKlPv-fceU8M9d1Ciz2dhmD-oGIpPvIkAF107OWd](https://github.com/user-attachments/assets/37c24fe2-b039-4f51-90cb-4cc9e7a56524)

### Capa oculta de tamaño $N=50$
![AGV_vUeLwTdneJsi483dALA7e_GR1kliiXC5aciok4tHrhQ7JYVzaE8Y7XhLwJggiie0rlQhMVxu8vx0Wm-mLaaG9Y3euIltPVvETWQpp33Awri-Fge64ozFkzVf](https://github.com/user-attachments/assets/675ce624-58e8-47e7-962a-e5ab9ec62ec8)

![unknown](https://github.com/user-attachments/assets/ee38c88d-1c20-4a93-8fc7-659053b8bb1f)

![unknown](https://github.com/user-attachments/assets/4409db89-d34d-4e6e-b9cb-d4fbda76235e)

![AGV_vUeon5xmIGY87uBFntoxm831fgHXFWh_NfUCcyibkRPcJKL6PgIL_XJRRe8B2AA51IhgsoSAv8tD6j0VD8XPA9TedcEpfOz1pQk72H_9FPUFLV4BeyOz3UKT](https://github.com/user-attachments/assets/352f3363-38fc-4b9e-8826-9bdb3105291a)

![unknown](https://github.com/user-attachments/assets/e56ebbce-2a49-4d76-a2c1-bd99528712e4)

## Material Adicional

<p align="left">
  <img src="https://github.com/user-attachments/assets/6be44635-da1a-4ba9-b7b0-5e343ff21189" width="450"/>
</p>


https://github.com/user-attachments/assets/3de7c7ff-c5a7-4c0b-a021-2531b6a3acba


https://github.com/user-attachments/assets/d7a8b3bf-d558-4ed9-b2b0-370461d25568

# Autor
Este código fue escrito por [Nicolas Ratier Werbin](mailto:nicolasratierwerbin@gmail.com).

