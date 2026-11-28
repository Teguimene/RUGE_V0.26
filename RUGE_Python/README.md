# RUGE en Python (PyTorch)

Cette implémentation est développée séparément de `RUGE/` et `RUGE_RotatE/`.
Elle porte les deux modèles RUGE, ComplEx et RotatE, utilise CUDA lorsqu'il est
disponible et conserve les fichiers numériques d'entrée ainsi que les matrices
de sortie Java.

## Installation

Kaggle fournit normalement déjà PyTorch. Depuis la racine du dépôt :

```bash
python -m pip install -e ./RUGE_Python
```

Tests (run from `RUGE_Python/`):

```bash
python -m unittest discover -s tests -v
```

## Entraînement ComplEx avec les règles RUGE

```bash
python -m ruge_python --model complex \
  -train datasets/yago37/train.txt \
  -valid datasets/yago37/valid.txt \
  -test datasets/yago37/test.txt \
  -rule datasets/yago37/groundings.txt \
  -m 37 -n 123189 -k 150 -d 0.003 -c 0.01 -ne 10 \
  -ge 1.0 -gr 0.5 -# 1000 -skip 50 -eval 1000
```

## Entraînement RotatE avec les règles RUGE

```bash
python -m ruge_python --model rotate \
  -train datasets/yago37/train.txt \
  -valid datasets/yago37/valid.txt \
  -test datasets/yago37/test.txt \
  -rule datasets/yago37/groundings.txt \
  -m 37 -n 123189 -k 150 -d 0.003 -c 0.01 -ne 10 \
  -ge 1.0 -gr 0.5 -gamma 12 -# 1000 -skip 50 -eval 1000
```

`--device auto` sélectionne CUDA si disponible, sinon le CPU. `--precision
float64` est le défaut pour se rapprocher des calculs Java en double précision;
`--precision float32` permet un entraînement GPU plus rapide.

L'entraînement affiche une barre `tqdm` mise à jour à chaque mini-batch, et les
évaluations de validation et de test affichent leur progression par triplet.
Les barres indiquent le temps écoulé et une estimation du temps restant.

Les matrices sont écrites dans le format texte de Java, avec les noms
`MatrixE.real.best`, `MatrixE.imag.best`, et les matrices de relation propres à
chaque modèle. Un checkpoint PyTorch est aussi enregistré sous
`ruge_python_best.pt`.

## Évaluation filtrée sur le test

Après installation, un checkpoint PyTorch peut être évalué ainsi :

```bash
python -m ruge_python.evaluate --model rotate \
  --checkpoint ruge_python_best.pt \
  -train datasets/yago37/train.txt \
  -valid datasets/yago37/valid.txt \
  -test datasets/yago37/test.txt -m 37 -n 123189 -k 150
```

Pour lire les fichiers texte compatibles avec Java, remplace `--checkpoint`
par `--matrix-dir .` et précise `-gamma 12` pour RotatE.

## Produire des prédictions top-K

Pour compléter un sujet et une relation avec les objets les mieux classés :

```bash
python -m ruge_python.predict \
  --checkpoint ruge_python_best.pt \
  --entity-map datasets/yago37/entityid.txt \
  --relation-map datasets/yago37/relationid.txt \
  --train datasets/yago37/train.txt \
  --subject "<Ibrahim_Sekagya>" \
  --relation "<playsFor>" \
  --top-k 10
```

Pour compléter une relation et un objet avec les sujets les mieux classés,
remplace `--subject` par `--object "<nom-de-l-entite>"`. Les faits déjà présents
dans le train sont filtrés. Les scores servent à classer les candidats; ce ne
sont pas des probabilités. Les triplets retournés pourront ensuite être fournis
à l'étape d'explication par règles.

Ajoute `--candidate "<nom-de-l-entite>"` pour afficher aussi le rang filtré
et le score d'un candidat précis, même s'il n'est pas dans le top-K. Cela permet
de comparer le même triplet avec deux checkpoints, par exemple avec et sans
groundings de règles négatives.

## Prétraitement

Les commandes Python `convert` et `ground` remplacent les deux utilitaires Java.
Les fichiers de mapping existants sont réutilisés pour ne pas renuméroter les
entités et relations :

```bash
python -m ruge_python.preprocess convert \
  --entity-map datasets/yago37/entityid.txt \
  --relation-map datasets/yago37/relationid.txt \
  --raw-train datasets/yago37/yago37_triples.train \
  --raw-valid datasets/yago37/yago37_triples.valid \
  --raw-test datasets/yago37/yago37_triples.test \
  --train-out datasets/yago37/train.txt \
  --valid-out datasets/yago37/valid.txt \
  --test-out datasets/yago37/test.txt

python -m ruge_python.preprocess ground \
  --relation-map datasets/yago37/relationid.txt \
  --rules datasets/yago37/yago37_rule \
  --train datasets/yago37/train.txt \
  --output datasets/yago37/groundings.txt
```

Sans fichiers de mapping existants, la conversion crée des IDs déterministes
triés. Cette situation ne reproduit pas l'ordre non spécifié des `HashMap`
Java; les mappings existants sont donc à conserver pour des comparaisons.

Les groundings historiques, sans marqueur de polarité, sont lus comme positifs.
Le chargeur accepte aussi des lignes où `positive` ou `negative` suit l'index
`unseen`, par exemple `2 negative (0 1 2) (2 0 1) 0.9`. À cette étape, le
trainer réduit la probabilité courante d'une cible négative finale de
`rule_weight × confidence × probabilité(prémisses)`, puis borne le résultat
entre 0 et 1. Une cible déjà connue positive dans train, validation ou test
provoque une erreur au lieu d'être entraînée avec une étiquette contradictoire.
Après audit des groundings, ajoute `--negative-rule
datasets/yago37/inverse_exclusion_groundings.txt` à la commande d'entraînement
pour les charger en plus du fichier donné par `-rule`.

Pour auditer les candidats de règle inverse `r(x,y) ⇒ ¬r(y,x)` à partir du
train, sans les appliquer à l'entraînement :

```bash
python -m ruge_python.preprocess mine-inverse \
  --relation-map datasets/yago37/relationid.txt \
  --train datasets/yago37/train.txt \
  --output datasets/yago37/inverse_exclusion_report.tsv \
  --min-support 20 --min-confidence 0.8
```

Le rapport contient toutes les relations et indique celles qui passent les
seuils. La confiance lissée utilise `((support - violations) + 1) /
(support + 2)`. Elle mesure la cohérence observée dans le train, pas la
probabilité que les triplets absents soient réellement faux.

Pour l'expérience initiale, les groundings de toutes les relations qui passent
les seuils sont écrits séparément. Les positifs connus fournis sont exclus des
cibles négatives; la confiance, elle, est toujours calculée uniquement sur le
train. Cela inclut notamment `hasGender`; `--relation` peut servir plus tard à
limiter l'expérience à une sélection de relations :

```bash
python -m ruge_python.preprocess ground-inverse \
  --relation-map datasets/yago37/relationid.txt \
  --train datasets/yago37/train.txt \
  --known-positive datasets/yago37/valid.txt datasets/yago37/test.txt \
  --output datasets/yago37/inverse_exclusion_groundings.txt \
  --min-support 20 --min-confidence 0.8
```

Sans `--relation`, toutes les relations retenues par les seuils sont générées.
Répéter `--relation "<nom>"` pour limiter les groundings à plusieurs relations.

Le fichier généré peut être fourni au trainer avec `--negative-rule`, en plus
des règles positives passées à `-rule`. Il faut d'abord examiner le rapport et
les groundings générés.
