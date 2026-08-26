# Détection des deepfakes audio

Ce dépôt contient le travail réalisé dans le cadre de notre stage sur la **détection des deepfakes audio**.

Le projet porte principalement sur l'étude et l'implémentation de méthodes de **détection de spoofing audio**, avec une attention particulière portée à **AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks)** ainsi qu'à une approche de référence basée sur un **CNN 2D** utilisant des spectrogrammes.

---

## Objectifs du projet

Les principaux objectifs du projet sont :

- Comprendre les bases de la détection des deepfakes audio.
- Comprendre les caractéristiques du signal audio utilisées pour la détection.
- Étudier le dataset et le protocole **ASVspoof**.
- Comprendre l'architecture et le fonctionnement d'**AASIST**.
- Implémenter et analyser les différentes étapes d'AASIST.
- Développer une approche de référence basée sur un **CNN 2D** appliqué à des spectrogrammes.
- Évaluer les deux approches avec des métriques adaptées.
- Comparer leurs performances et leurs caractéristiques.

---

## 🔍 Problématique

Les deepfakes audio sont des voix artificiellement générées ou manipulées qui peuvent imiter une voix humaine réelle.

Dans le cadre de la détection de spoofing audio, deux types importants d'attaques sont notamment étudiés :

- **Text-to-Speech (TTS)** : génération d'une parole à partir d'un texte.
- **Voice Conversion (VC)** : transformation d'une voix afin qu'elle ressemble à celle d'un autre locuteur.

L'objectif d'un système d'anti-spoofing audio est de déterminer si un fichier audio correspond à :

- **Bona-fide** : parole réelle et authentique.
- **Spoof** : parole synthétique ou manipulée.

---

## Approches étudiées

### 1. AASIST

**AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks)** est un modèle d'anti-spoofing audio qui prend directement en entrée la forme d'onde audio.

Le fonctionnement général peut être résumé ainsi :

```text
Audio brut
    ↓
Sinc-Convolution
    ↓
Blocs résiduels
    ↓
Représentation des caractéristiques
    ↓
Graphes spectral et temporel
    ↓
Attention sur graphes
    ↓
Max Graph Operation
    ↓
Readout
    ↓
Classification
    ↓
Bona-fide / Spoof