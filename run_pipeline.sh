#!/bin/bash
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

LOG_FILE="tcc_log_$(date +%Y%m%d_%H%M).log"

echo -e "${BLUE}====================================================${NC}"
echo -e "${BLUE}   PIPELINE DE PRIVACIDADE DIFERENCIAL - TSE 2024   ${NC}"
echo -e "${BLUE}====================================================${NC}"

run_step() {
    echo -e "\n${BLUE}👉 Passo: $2 ($1)${NC}"
    # O grep -v filtra os alertas de dgl/Goggle que poluem o log
    python3 "$1" 2> >(grep -v -E "CRITICAL|dgl|Goggle|module disabled" >&2) | tee -a $LOG_FILE
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo -e "${GREEN}✅ $2 concluído.${NC}"
    else
        echo -e "${RED}❌ Erro em $2. Abortando.${NC}"
        exit 1
    fi
}

run_step "01_data_prep.py" "Preparação de Dados"
run_step "02_synthesizer.py" "Síntese AIM"
run_step "03_utility_evaluator.py" "Avaliação TSTR"
run_step "05_privacy_auditor.py" "Auditoria de Privacidade"

echo -e "\n${GREEN}✨ Experimento finalizado! Log salvo em: $LOG_FILE${NC}"