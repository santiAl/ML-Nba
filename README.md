# 🏀 ML-NBA: Predicción de Resultados en la NBA

¡Bienvenido/a al modelo **ML-NBA**! 🎉  
Con este proyecto podrás hacer **predicciones para partidos de la NBA** utilizando estadísticas pasadas. 📊✨  

---

## 📚 ¿Qué hace este modelo?
Actualmente, el modelo utiliza **Random Forest o MLP** 📈 para clasificar un partido como **ganado** o **perdido** según estadísticas históricas.  
Tu puedes elegir que modelo utilizar.

### 🚀 Características destacadas:
- **Precisión actual**: 64% ✅  
- En constante mejora: Planeo seguir actualizando los datos y optimizando el modelo. 🔧  
- **Puesta en producción**: Implementado en **AWS EC2** 🌐  

Puedes probar el modelo accediendo a la siguiente IP:  
👉 [http://3.133.153.4:5000](http://3.133.153.4:5000)

---
## 📊 Variables Predictoras:
A continuación, te detallo las variables utilizadas para las predicciones:  

1. **Mean field goal last 10 matches**: Media de porcentaje de tiros de campo en los últimos 10 partidos. 🏹 
2. **Mean blocks last 10 matches**: Media de tapones en los últimos 10 partidos. 🛡️  
3. **Mean oponent blocks last 10 matches**: Media de tapones del rival en los últimos 10 partidos. 🛡️ 
4. **Mean oponent field goal last 10 matchest**: Media de porcentaje de tiros de campo del rival en los últimos 10 partidos. 🏀   
5. **Mean recieved field goal % last 10 matches**: Media de porcentajes de tiros de campo recibidos en los últimos 10 partidos. ❌ 
6. **Mean opponent turnovers last 10 matches**: Media de pérdidas por partido del oponente en los últimos 10 partidos. ❌
7. **Mean oponet recieved points last 10 matches**: Media de puntos recibidos por el rival en los últimos 10 partidos. ❌
8. **Mean score last 10 matches**: Media de puntos anotados por el equipo 1 en los ultimos 10 partidos. 🏀 
9. **Mean opoent recieved field goal % last 10 matches**: Media de porcentajes de tiros de campo recibidos por el equipo 2 en los últimos 10 partidos. ❌
10. **Mean turnovers last 10 matches**: Media de pérdidas del equipo 1 en los últimos 10 partidos. ❌  
11. **Mean oponet recieved points last 10 matches**: Media de puntos recibidos por el equipo 2 en los últimos 10 partidos.
12. **Mean oponent score last 10 matches**: Media de puntos anotados por el equipo 2 en los ultimos 10 partidos. 🏀 
13. **Oponent - Matches won - Matches lost**: Diferencia entre partidos ganados y perdidos por el equipo 2 en la temporada. 🏆
14. **Home game?**: ¿Es un partido de local para el equipo 1? 🏠  
   - `1` si es local.  
   - `0` si es visitante.  
15. **Matches won - Matches lost**: Diferencia entre partidos ganados y perdidos por el equipo 1 en la temporada. 🏆
---

### 🌟 Agradecimiento
Gracias por tomarte el tiempo de revisar este proyecto.  
¡Espero que te sea útil y emocionante! 😊  

