#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-assets/qwen36-spark-benchmark-card.png}"
FONT="/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD="/System/Library/Fonts/Supplemental/Arial Bold.ttf"

magick -size 1600x900 xc:'#070b12' \
  -fill '#0b1220' -draw 'circle 1420,120 1700,120' \
  -fill '#052e2b' -draw 'circle 160,820 500,820' \
  -fill '#0f172a' -stroke '#263449' -strokewidth 3 -draw 'roundrectangle 55,50 1545,850 44,44' \
  -stroke none -font "$BOLD" -fill '#f8fafc' -pointsize 56 -annotate +92+132 'Qwen3.6 35B NVFP4 on 1× DGX Spark' \
  -stroke none -font "$FONT" -fill '#94a3b8' -pointsize 29 -annotate +94+184 'Native vLLM 0.26 · 262K context · FP8 KV · DFlash K=7 · no custom container' \
  \
  -fill '#020617' -stroke '#253449' -strokewidth 2 -draw 'roundrectangle 92,222 493,338 28,28' \
  -stroke none -font "$FONT" -fill '#94a3b8' -pointsize 24 -annotate +122+268 'single stream' \
  -stroke none -font "$BOLD" -fill '#f8fafc' -pointsize 58 -annotate +122+320 '73.4' \
  -stroke none -font "$FONT" -fill '#94a3b8' -pointsize 30 -annotate +286+318 'tok/s' \
  \
  -fill '#020617' -stroke '#253449' -strokewidth 2 -draw 'roundrectangle 522,222 972,338 28,28' \
  -stroke none -font "$FONT" -fill '#94a3b8' -pointsize 24 -annotate +552+268 '8-session aggregate' \
  -stroke none -font "$BOLD" -fill '#f8fafc' -pointsize 58 -annotate +552+320 '338.4' \
  -stroke none -font "$FONT" -fill '#94a3b8' -pointsize 30 -annotate +755+318 'tok/s' \
  \
  -fill '#111827' -stroke '#334155' -strokewidth 2 -draw 'roundrectangle 998,222 1456,338 28,28' \
  -stroke none -font "$FONT" -fill '#94a3b8' -pointsize 24 -annotate +1028+268 'model shape' \
  -stroke none -font "$BOLD" -fill '#f8fafc' -pointsize 36 -annotate +1028+306 '35B-A3B MoE' \
  -stroke none -font "$FONT" -fill '#94a3b8' -pointsize 21 -annotate +1028+330 'large feel, small active slice' \
  \
  -stroke none -font "$BOLD" -fill '#f8fafc' -pointsize 38 -annotate +98+415 'Concurrency sweep' \
  -stroke none -font "$FONT" -fill '#94a3b8' -pointsize 24 -annotate +98+453 'Exact 500-token code-shaped generations · OpenAI-compatible endpoint' \
  \
  -stroke '#1e293b' -strokewidth 2 -draw 'line 150,730 1450,730' \
  -draw 'line 150,640 1450,640' \
  -draw 'line 150,550 1450,550' \
  -draw 'line 150,460 1450,460' \
  -stroke none -font "$FONT" -fill '#94a3b8' -pointsize 18 -annotate +105+736 '0' \
  -annotate +92+646 '100' \
  -annotate +92+556 '200' \
  -annotate +92+466 '300' \
  \
  -fill '#22c55e' -draw 'roundrectangle 228,667 378,730 16,16' \
  -fill '#38bdf8' -draw 'rectangle 228,667 378,698' \
  -stroke none -font "$BOLD" -fill '#f8fafc' -pointsize 27 -annotate +221+650 '73.4' \
  -stroke none -font "$FONT" -fill '#94a3b8' -pointsize 25 -annotate +281+774 '1×' \
  \
  -fill '#22c55e' -draw 'roundrectangle 468,604 618,730 16,16' \
  -fill '#38bdf8' -draw 'rectangle 468,604 618,667' \
  -stroke none -font "$BOLD" -fill '#f8fafc' -pointsize 27 -annotate +450+587 '147.4' \
  -stroke none -font "$FONT" -fill '#94a3b8' -pointsize 25 -annotate +521+774 '2×' \
  \
  -fill '#22c55e' -draw 'roundrectangle 708,526 858,730 16,16' \
  -fill '#38bdf8' -draw 'rectangle 708,526 858,628' \
  -stroke none -font "$BOLD" -fill '#f8fafc' -pointsize 27 -annotate +690+509 '238.0' \
  -stroke none -font "$FONT" -fill '#94a3b8' -pointsize 25 -annotate +761+774 '4×' \
  \
  -fill '#22c55e' -draw 'roundrectangle 948,450 1098,730 16,16' \
  -fill '#38bdf8' -draw 'rectangle 948,450 1098,590' \
  -stroke none -font "$BOLD" -fill '#f8fafc' -pointsize 27 -annotate +930+433 '327.0' \
  -stroke none -font "$FONT" -fill '#94a3b8' -pointsize 25 -annotate +1001+774 '6×' \
  \
  -fill '#22c55e' -draw 'roundrectangle 1188,440 1338,730 16,16' \
  -fill '#38bdf8' -draw 'rectangle 1188,440 1338,585' \
  -stroke none -font "$BOLD" -fill '#f8fafc' -pointsize 27 -annotate +1170+423 '338.4' \
  -stroke none -font "$FONT" -fill '#94a3b8' -pointsize 25 -annotate +1241+774 '8×' \
  \
  -stroke '#f97316' -strokewidth 7 -fill none -draw 'path \"M 303,667 C 470,625 520,610 543,604 C 676,555 750,532 783,526 C 890,485 980,462 1023,450 C 1112,443 1225,438 1263,440\"' \
  \
  -fill '#020617' -stroke '#334155' -strokewidth 2 -draw 'roundrectangle 1370,454 1498,730 24,24' \
  -stroke none -font "$FONT" -fill '#94a3b8' -pointsize 21 -annotate +1390+500 'startup' \
  -stroke none -font "$BOLD" -fill '#f8fafc' -pointsize 31 -annotate +1390+545 '647K' \
  -stroke none -font "$FONT" -fill '#94a3b8' -pointsize 19 -annotate +1390+582 'KV tokens' \
  -stroke none -font "$FONT" -fill '#94a3b8' -pointsize 21 -annotate +1390+650 'request' \
  -stroke none -font "$BOLD" -fill '#f8fafc' -pointsize 27 -annotate +1390+692 '262K ctx' \
  \
  -stroke none -font "$FONT" -fill '#94a3b8' -pointsize 22 -annotate +92+825 'github.com/sojufx/Qwen3.6-35B-NVFP4-DGX-Spark-Recipe' \
  -stroke none -font "$BOLD" -fill '#f8fafc' -pointsize 25 -annotate +1120+825 'One Spark. Native vLLM. Real numbers.' \
  "$OUT"
