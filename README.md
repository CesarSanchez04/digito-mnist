---
title: Digito
emoji: 🔢
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

Servicio FastAPI que recibe una foto (multipart, campo `file`) en `POST /predict/`
y devuelve el dígito manuscrito reconocido por una CNN entrenada con MNIST:

```json
{"prediction": ["7"], "confianza": 0.99}
```
