#!/usr/bin/env bash
# Non-interactive build with a time limit; prints errors.
cd /home/karik/lectures/lecture_i2dl/lecture_advanced_ai_ml/session-09-transformers-modernized
timeout 400 latexmk -pdf -interaction=nonstopmode -halt-on-error slides-transformers-modernized.tex >/dev/null 2>&1; rc=$?
grep -n -A3 "^! " slides-transformers-modernized.log | head -12; echo "build rc=$rc"
