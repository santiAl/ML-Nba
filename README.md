# 🏀 ML-NBA: Predicción de Resultados en la NBA

¡Bienvenido/a al modelo **ML-NBA**! 🎉  
Con este proyecto podrás hacer **predicciones para partidos de la NBA** utilizando estadísticas pasadas. 📊✨  

---

## 📚 ¿Qué hace este modelo?
Actualmente, el modelo utiliza **Ridge Regression** 📈 para clasificar un partido como **ganado** o **perdido** según estadísticas históricas.  

### 🚀 Características destacadas:
- **Precisión actual**: 70% ✅  
- En constante mejora: Planeo seguir actualizando los datos y optimizando el modelo. 🔧  
- **Puesta en producción**: Implementado en **AWS EC2** 🌐  

Puedes probar el modelo accediendo a la siguiente IP:  
👉 [http://3.133.153.4:5000](http://3.133.153.4:5000)

---

## 📊 Variables Predictoras:
A continuación, te detallo las variables utilizadas para las predicciones:  

1. **Mean assists last 10 matches**: Media de asistencias en los últimos 10 partidos. 🏀  
2. **Mean blocks last 10 matches**: Media de tapones en los últimos 10 partidos. 🛡️  
3. **Mean offensive rebounds last 10 matches**: Media de rebotes ofensivos en los últimos 10 partidos. 🔄  
4. **Mean three pointers made last 10 matches**: Media de tiros de 3 puntos convertidos en los últimos 10 partidos. 🎯  
5. **Mean turnovers last 10 matches**: Media de pérdidas en los últimos 10 partidos. ❌  
6. **Mean field goal last 10 matches**: Media de porcentajes de tiros de campo en los últimos 10 partidos. 🏹  
7. **Home game?**: ¿Es un partido de local? 🏠  
   - `1` si es local.  
   - `0` si es visitante.  
8. **Mean opponent blocks last 10 matches**: Media de tapones por partido del oponente en los últimos 10 partidos. 🛡️  
9. **Mean opponent turnovers last 10 matches**: Media de pérdidas por partido del oponente en los últimos 10 partidos. ❌  
10. **Matches won - Matches lost**: Diferencia entre partidos ganados y perdidos por el equipo en la temporada. 🏆  

---

### 🌟 Agradecimiento
Gracias por tomarte el tiempo de revisar este proyecto.  
¡Espero que te sea útil y emocionante! 😊  

