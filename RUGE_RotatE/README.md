# RUGE avec RotatE

Cette variante est isolée dans `RUGE_RotatE/` : le projet `RUGE/` d’origine et son implémentation ComplEx ne sont pas modifiés.

## Représentation

Les entités sont des vecteurs complexes (deux matrices : partie réelle et partie imaginaire). Chaque dimension de relation est un angle \(\theta_r\), et l’opération de relation est la rotation complexe \(e^{i\theta_r}\). Le score utilisé est :

```text
score(h, r, t) = margin - sum(|Re(h * r - t)| + |Im(h * r - t)|)
```

Le score élevé correspond à un triplet plausible. L’entraînement logistique, la corruption tête/queue et la régularisation des règles RUGE utilisent ce score. Les phases de relation sont ramenées dans \([-\pi,\pi]\) après chaque mise à jour.

## Compilation

Depuis PowerShell, dans ce dossier (la commande contourne la stratégie d’exécution uniquement pour ce script local) :

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\build.ps1
```

Cela compile les sources et crée `RUGE_RotatE.jar`. Java et `javac` doivent être disponibles dans le `PATH`.

## Entraînement

Les fichiers de triplets et de règles doivent être prétraités au même format numérique que pour le projet d’origine. Exemple pour YAGO37, lancé depuis `RUGE_RotatE/` :

```powershell
java -jar .\RUGE_RotatE.jar `
  -train ..\datasets\yago37\train.txt `
  -valid ..\datasets\yago37\valid.txt `
  -test ..\datasets\yago37\test.txt `
  -rule ..\datasets\yago37\groundings.txt `
  -m 37 -n 123189 -k 150 -d 0.003 -c 0.01 -ne 10 `
  -ge 1.0 -gr 0.5 -gamma 12 -# 1000 -skip 50 -eval 10
```

`-gamma` désigne la marge RotatE (par défaut `12`). `-k` est le nombre de dimensions complexes. `-eval` règle le nombre de triplets de validation évalués à chaque point de contrôle (10 par défaut pour accélérer les essais ; augmenter cette valeur pour des métriques plus représentatives). Les autres paramètres RUGE conservent leurs rôles habituels.

Les meilleurs embeddings sont écrits dans le dossier courant :

- `MatrixE.real.best`
- `MatrixE.imag.best`
- `MatrixR.phase.best`
- `result.log`

## Évaluation sur le jeu de test

```powershell
java -cp .\bin rotate.test.EvalRotatE `
  123189 37 150 `
  .\MatrixE.real.best .\MatrixE.imag.best .\MatrixR.phase.best `
  ..\datasets\yago37\train.txt ..\datasets\yago37\valid.txt `
  ..\datasets\yago37\test.txt 12
```

## Test ciblé

Après compilation :

```powershell
java -cp .\bin rotate.test.RotatEModelTest
```

Ce test vérifie une rotation de \(90^\circ\) et compare les gradients analytiques aux différences finies.
